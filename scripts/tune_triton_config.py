#!/usr/bin/env python3
"""
Step 5A: Test Triton with different config variations locally.

Generates multiple config.pbtxt files, restarts Triton with each,
and benchmarks to find optimal settings.

Variations tested:
  1. max_queue_delay_microseconds: 1000, 5000, 10000, 20000, 50000
  2. preferred_batch_size combos
  3. instance_group count (1 vs 2 CPU instances)

Usage:
  # Run all variations (requires local Docker + Triton container)
  python scripts/tune_triton_config.py

  # Test specific queue delays only
  python scripts/tune_triton_config.py --queue-delays 1000,5000,20000

  # Custom Triton URL
  python scripts/tune_triton_config.py --triton-url http://localhost:8003

  # Dry run (just generate configs, don't benchmark)
  python scripts/tune_triton_config.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_TEMPLATE = """# Auto-generated config for tuning experiment
# Variation: {description}

name: "openclip_vit_b32"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }}
]

output [
  {{
    name: "embedding"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_sizes} ]
  max_queue_delay_microseconds: {max_queue_delay}
}}

instance_group [
  {{
    count: {instance_count}
    kind: KIND_CPU
  }}
]

version_policy: {{ specific {{ versions: 1 }} }}
"""

GPU_CONFIG_TEMPLATE = """# Auto-generated config for GPU tuning experiment
# Variation: {description}

name: "openclip_vit_b32"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }}
]

output [
  {{
    name: "embedding"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_sizes} ]
  max_queue_delay_microseconds: {max_queue_delay}
}}

instance_group [
  {{
    count: {instance_count}
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

version_policy: {{ specific {{ versions: 1 }} }}

optimization {{
  cuda {{
    graphs: {cuda_graphs}
  }}
}}
"""


def generate_config_variations(queue_delays: List[int], for_gpu: bool = False) -> List[Dict]:
    """Generate a list of config variations to test."""
    variations = []

    # 1. Queue delay variations (main experiment)
    for delay in queue_delays:
        desc = f"queue_delay_{delay}us"
        variations.append({
            "name": desc,
            "description": f"Queue delay = {delay}µs ({delay/1000:.0f}ms)",
            "max_batch_size": 32,
            "preferred_batch_sizes": "4, 8, 16, 32",
            "max_queue_delay": delay,
            "instance_count": 1,
            "cuda_graphs": "true" if for_gpu else "false",
        })

    # 2. Instance count variation
    for count in [1, 2]:
        desc = f"instances_{count}"
        variations.append({
            "name": desc,
            "description": f"Instance count = {count}, queue delay = 5ms",
            "max_batch_size": 32,
            "preferred_batch_sizes": "4, 8, 16, 32",
            "max_queue_delay": 5000,
            "instance_count": count,
            "cuda_graphs": "true" if for_gpu else "false",
        })

    # 3. Batch size preference variations
    batch_combos = [
        ("1, 2, 4, 8", "small_batches"),
        ("4, 8, 16, 32", "default_batches"),
        ("8, 16, 32, 64", "large_batches"),
    ]
    for sizes, label in batch_combos:
        max_bs = int(sizes.split(",")[-1].strip())
        desc = f"batch_pref_{label}"
        variations.append({
            "name": desc,
            "description": f"Preferred batch sizes = [{sizes}]",
            "max_batch_size": max_bs,
            "preferred_batch_sizes": sizes,
            "max_queue_delay": 5000,
            "instance_count": 1,
            "cuda_graphs": "true" if for_gpu else "false",
        })

    # 4. CUDA graphs on vs off (GPU only)
    if for_gpu:
        for graphs in ["true", "false"]:
            desc = f"cuda_graphs_{graphs}"
            variations.append({
                "name": desc,
                "description": f"CUDA graphs = {graphs}",
                "max_batch_size": 32,
                "preferred_batch_sizes": "4, 8, 16, 32",
                "max_queue_delay": 5000,
                "instance_count": 1,
                "cuda_graphs": graphs,
            })

    return variations


def write_config(variation: Dict, config_dir: Path, for_gpu: bool = False):
    """Write a config.pbtxt for a given variation."""
    template = GPU_CONFIG_TEMPLATE if for_gpu else CONFIG_TEMPLATE
    config_text = template.format(**variation)
    config_path = config_dir / "config.pbtxt"
    config_path.write_text(config_text)
    return config_path


def restart_triton_container(container_name: str = "triton-inference-service",
                             image: str = "photo-duplicate-triton:latest",
                             timeout: int = 90) -> bool:
    """Stop, remove, and restart the Triton container."""
    # Stop and remove
    subprocess.run(["docker", "rm", "-f", container_name],
                   capture_output=True, timeout=10)
    time.sleep(1)

    # Start new container
    result = subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "-p", "8003:8000",
        "-p", "8004:8001",
        "-p", "8005:8002",
        image,
    ], capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f"  ✗ Failed to start container: {result.stderr}")
        return False

    # Wait for health
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://localhost:8003/v2/health/ready", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)

    print(f"  ✗ Triton did not become ready within {timeout}s")
    return False


def quick_benchmark(triton_url: str, iterations: int = 20, concurrency: int = 8,
                    concurrent_requests: int = 100) -> Dict:
    """Run a quick benchmark against the current Triton instance."""
    from scripts.benchmark_backends import (
        bench_single_triton,
        bench_batch_triton,
        bench_concurrent_triton,
        percentile_stats,
    )

    results = {}

    # Single image
    lats = bench_single_triton(triton_url, iterations)
    if lats:
        results["single"] = percentile_stats(lats)

    # Batch 8
    lats = bench_batch_triton(triton_url, 8, iterations)
    if lats:
        results["batch_8"] = percentile_stats(lats)

    # Batch 32
    lats = bench_batch_triton(triton_url, 32, iterations)
    if lats:
        results["batch_32"] = percentile_stats(lats)

    # Concurrent
    results["concurrent"] = bench_concurrent_triton(triton_url, concurrent_requests, concurrency)

    # Triton metrics
    try:
        r = requests.get("http://localhost:8005/metrics", timeout=5)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if "nv_inference_request_success" in line and "openclip_vit_b32" in line and not line.startswith("#"):
                    results["triton_total_requests"] = float(line.split()[-1])
                if "nv_inference_queue_duration_us" in line and "openclip_vit_b32" in line and not line.startswith("#"):
                    results["triton_queue_duration_us"] = float(line.split()[-1])
    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Tune Triton config variations (Step 5A)")
    parser.add_argument("--triton-url", type=str, default="http://localhost:8003")
    parser.add_argument("--queue-delays", type=str, default="1000,5000,10000,20000,50000",
                        help="Comma-separated queue delays in microseconds")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--concurrent-requests", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only, don't benchmark")
    parser.add_argument("--gpu", action="store_true", help="Generate GPU configs instead of CPU")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    queue_delays = [int(x) for x in args.queue_delays.split(",")]
    config_dir = PROJECT_ROOT / "model_repository" / "openclip_vit_b32"

    variations = generate_config_variations(queue_delays, for_gpu=args.gpu)

    print("=" * 60)
    print(f"Triton Config Tuning — {len(variations)} variations")
    print("=" * 60)
    print(f"Queue delays: {queue_delays}")
    print(f"GPU mode: {args.gpu}")
    print(f"Iterations: {args.iterations}")
    print(f"Concurrency: {args.concurrency}")
    print()

    if args.dry_run:
        # Just generate and show configs
        output_dir = PROJECT_ROOT / "benchmark_results" / "config_variations"
        output_dir.mkdir(parents=True, exist_ok=True)
        for var in variations:
            config_text = (GPU_CONFIG_TEMPLATE if args.gpu else CONFIG_TEMPLATE).format(**var)
            out_file = output_dir / f"config_{var['name']}.pbtxt"
            out_file.write_text(config_text)
            print(f"  ✓ Generated: {out_file.name} — {var['description']}")
        print(f"\nConfigs saved to: {output_dir}")
        print("Review them, then re-run without --dry-run to benchmark.")
        return

    # Backup original config
    original_config = (config_dir / "config.pbtxt").read_text()

    all_results = {}
    try:
        for i, var in enumerate(variations, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(variations)}] {var['description']}")
            print(f"{'='*60}")

            # Write new config
            write_config(var, config_dir, for_gpu=args.gpu)

            # Rebuild container with new config
            print("  Building Triton image with new config...")
            build_result = subprocess.run(
                ["docker", "build", "-f", "Dockerfile.triton", "-t",
                 "photo-duplicate-triton:latest", "."],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300,
            )
            if build_result.returncode != 0:
                print(f"  ✗ Build failed: {build_result.stderr[-200:]}")
                continue

            # Restart container
            print("  Restarting Triton container...")
            if not restart_triton_container():
                print("  ✗ Container failed to start, skipping")
                continue

            # Wait a bit for model to warm up
            time.sleep(3)

            # Benchmark
            print("  Running benchmark...")
            bench_results = quick_benchmark(
                args.triton_url, args.iterations, args.concurrency, args.concurrent_requests
            )
            all_results[var["name"]] = {
                "config": var,
                "results": bench_results,
            }

            # Print summary
            if "single" in bench_results:
                print(f"  Single:     {bench_results['single']['mean_ms']:.1f}ms mean")
            if "batch_32" in bench_results:
                print(f"  Batch-32:   {bench_results['batch_32']['mean_ms']:.1f}ms mean")
            if "concurrent" in bench_results:
                print(f"  Throughput: {bench_results['concurrent'].get('throughput_rps', 0):.1f} req/s")

    finally:
        # Restore original config
        (config_dir / "config.pbtxt").write_text(original_config)
        print(f"\n✓ Restored original config.pbtxt")

    # Save results
    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"triton_tuning_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("TUNING SUMMARY")
    print("=" * 80)
    print(f"{'Variation':<30} {'Single(ms)':<12} {'Batch32(ms)':<14} {'Throughput':<12}")
    print("-" * 80)
    for name, data in all_results.items():
        r = data["results"]
        single = r.get("single", {}).get("mean_ms", "-")
        batch32 = r.get("batch_32", {}).get("mean_ms", "-")
        tput = r.get("concurrent", {}).get("throughput_rps", "-")
        single_str = f"{single:.1f}" if isinstance(single, float) else single
        batch32_str = f"{batch32:.1f}" if isinstance(batch32, float) else batch32
        tput_str = f"{tput:.1f}" if isinstance(tput, float) else tput
        print(f"  {name:<28} {single_str:<12} {batch32_str:<14} {tput_str:<12}")


if __name__ == "__main__":
    main()
