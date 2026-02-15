#!/usr/bin/env python3
"""
Step 5A: Profile ONNX Runtime for slow ops, CPU fallbacks, and serialization overhead.

This script runs locally (no GPU required) and measures:
  1. ONNX Runtime session creation time
  2. Per-operator profiling (which ops are slow?)
  3. CPU vs GPU provider comparison (if available)
  4. Serialization overhead: JSON .tolist() vs binary encoding
  5. Preprocessing overhead: numpy → tensor conversion
  6. Batch size scaling efficiency

Usage:
  python scripts/profile_onnx.py
  python scripts/profile_onnx.py --iterations 100
  python scripts/profile_onnx.py --model-path model_repository/openclip_vit_b32/1/model.onnx
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_onnx_model(model_path: Path) -> Dict:
    """Inspect ONNX model structure: ops, graph nodes, I/O shapes."""
    import onnx

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    graph = model.graph

    # Count op types
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    # I/O info
    inputs = [
        {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim],
         "dtype": inp.type.tensor_type.elem_type}
        for inp in graph.input
    ]
    outputs = [
        {"name": out.name, "shape": [d.dim_value for d in out.type.tensor_type.shape.dim],
         "dtype": out.type.tensor_type.elem_type}
        for out in graph.output
    ]

    info = {
        "opset_version": model.opset_import[0].version,
        "total_nodes": len(graph.node),
        "op_counts": dict(sorted(op_counts.items(), key=lambda x: -x[1])),
        "inputs": inputs,
        "outputs": outputs,
        "file_size_mb": round(model_path.stat().st_size / 1024 / 1024, 2),
    }
    return info


def profile_session_creation(model_path: Path, iterations: int = 5) -> Dict:
    """Measure ONNX Runtime session creation time (cold-start proxy)."""
    import onnxruntime as ort

    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        _ = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": round(np.mean(times) * 1000, 2),
        "min_ms": round(np.min(times) * 1000, 2),
        "max_ms": round(np.max(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "iterations": iterations,
    }


def profile_inference(model_path: Path, batch_sizes: List[int], iterations: int) -> Dict:
    """Profile inference latency across batch sizes with operator-level profiling."""
    import onnxruntime as ort

    # Create session with profiling enabled
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.log_severity_level = 3  # Suppress verbose logs

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    results = {}
    for bs in batch_sizes:
        dummy = np.random.rand(bs, 3, 224, 224).astype(np.float32)

        # Warm-up
        for _ in range(3):
            session.run(None, {"image": dummy})

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            session.run(None, {"image": dummy})
            latencies.append(time.perf_counter() - t0)

        arr = np.array(latencies) * 1000  # to ms
        results[f"batch_{bs}"] = {
            "mean_ms": round(float(np.mean(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "images_per_sec": round(bs / float(np.mean(latencies)), 2),
        }
        print(f"  batch={bs:>3}: {results[f'batch_{bs}']['mean_ms']:>8.2f}ms mean, "
              f"{results[f'batch_{bs}']['images_per_sec']:>8.2f} img/s")

    # Get profiling results
    profile_file = session.end_profiling()
    profile_data = _parse_ort_profile(profile_file)
    results["operator_profile"] = profile_data

    return results


def _parse_ort_profile(profile_path: str) -> Dict:
    """Parse ONNX Runtime profiling JSON and summarize top operators."""
    try:
        with open(profile_path) as f:
            events = json.load(f)

        # Aggregate by op_name
        op_times = {}
        for event in events:
            if event.get("cat") == "Node":
                name = event.get("name", "unknown")
                # Extract op type from name (format: "op_type_N")
                dur_us = event.get("dur", 0)
                op_type = event.get("args", {}).get("op_name", name.split("_")[0])
                if op_type not in op_times:
                    op_times[op_type] = {"total_us": 0, "count": 0}
                op_times[op_type]["total_us"] += dur_us
                op_times[op_type]["count"] += 1

        # Sort by total time
        sorted_ops = sorted(op_times.items(), key=lambda x: -x[1]["total_us"])
        total_us = sum(v["total_us"] for _, v in sorted_ops)

        top_ops = []
        for op_name, stats in sorted_ops[:15]:
            pct = (stats["total_us"] / total_us * 100) if total_us > 0 else 0
            top_ops.append({
                "op": op_name,
                "total_ms": round(stats["total_us"] / 1000, 3),
                "count": stats["count"],
                "pct_of_total": round(pct, 1),
            })

        # Clean up profile file
        Path(profile_path).unlink(missing_ok=True)

        return {
            "total_compute_ms": round(total_us / 1000, 3),
            "top_operators": top_ops,
        }
    except Exception as e:
        return {"error": str(e)}


def profile_serialization_overhead(batch_sizes: List[int], iterations: int) -> Dict:
    """
    Measure the cost of serializing numpy arrays for Triton HTTP API.

    This is a KEY finding: the benchmark sends data as JSON .tolist(),
    which converts every float to a string. For batch=32 of 3x224x224,
    that's 32 * 3 * 224 * 224 = 4,816,896 floats serialized as strings.
    """
    results = {}
    for bs in batch_sizes:
        data = np.random.rand(bs, 3, 224, 224).astype(np.float32)

        # Method 1: JSON .tolist() (current benchmark approach)
        json_times = []
        json_sizes = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            payload = {
                "inputs": [{
                    "name": "image",
                    "shape": list(data.shape),
                    "datatype": "FP32",
                    "data": data.flatten().tolist(),
                }],
                "outputs": [{"name": "embedding"}],
            }
            json_str = json.dumps(payload)
            json_times.append(time.perf_counter() - t0)
            json_sizes.append(len(json_str))

        # Method 2: Binary encoding (Triton supports this via HTTP)
        binary_times = []
        binary_sizes = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            binary_data = data.tobytes()
            binary_times.append(time.perf_counter() - t0)
            binary_sizes.append(len(binary_data))

        # Method 3: Just the numpy .tolist() conversion (no JSON)
        tolist_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = data.flatten().tolist()
            tolist_times.append(time.perf_counter() - t0)

        results[f"batch_{bs}"] = {
            "num_floats": int(np.prod(data.shape)),
            "json_tolist": {
                "mean_ms": round(np.mean(json_times) * 1000, 2),
                "payload_size_kb": round(np.mean(json_sizes) / 1024, 1),
            },
            "binary": {
                "mean_ms": round(np.mean(binary_times) * 1000, 2),
                "payload_size_kb": round(np.mean(binary_sizes) / 1024, 1),
            },
            "tolist_only": {
                "mean_ms": round(np.mean(tolist_times) * 1000, 2),
            },
            "json_vs_binary_speedup": round(np.mean(json_times) / max(np.mean(binary_times), 1e-9), 1),
        }

        print(f"  batch={bs:>3}: JSON={results[f'batch_{bs}']['json_tolist']['mean_ms']:>8.2f}ms "
              f"({results[f'batch_{bs}']['json_tolist']['payload_size_kb']:>8.1f}KB), "
              f"Binary={results[f'batch_{bs}']['binary']['mean_ms']:>6.2f}ms "
              f"({results[f'batch_{bs}']['binary']['payload_size_kb']:>8.1f}KB), "
              f"Speedup={results[f'batch_{bs}']['json_vs_binary_speedup']}x")

    return results


def profile_preprocessing(iterations: int) -> Dict:
    """Measure preprocessing overhead (HWC uint8 → CHW float32)."""
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Method 1: Simple normalization + transpose (current approach)
    times_simple = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        t = img.astype(np.float32) / 255.0
        t = np.transpose(t, (2, 0, 1))
        times_simple.append(time.perf_counter() - t0)

    # Method 2: With OpenCLIP preprocessing (full pipeline)
    try:
        import open_clip
        from PIL import Image as PILImage
        _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        pil_img = PILImage.fromarray(img)

        times_full = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = preprocess(pil_img)
            times_full.append(time.perf_counter() - t0)
        full_preprocess_ms = round(np.mean(times_full) * 1000, 3)
    except ImportError:
        full_preprocess_ms = None

    return {
        "simple_normalize_transpose_ms": round(np.mean(times_simple) * 1000, 3),
        "openclip_full_preprocess_ms": full_preprocess_ms,
        "note": "Triton receives raw tensors (already preprocessed); PyTorch server does preprocessing internally",
    }


def main():
    parser = argparse.ArgumentParser(description="Profile ONNX Runtime for Step 5A optimization")
    parser.add_argument("--model-path", type=Path,
                        default=PROJECT_ROOT / "model_repository" / "openclip_vit_b32" / "1" / "model.onnx")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if not args.model_path.exists():
        print(f"✗ ONNX model not found: {args.model_path}")
        print("  Run: python scripts/export_to_onnx.py --test")
        sys.exit(1)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. Model inspection
    print("=" * 60)
    print("1. ONNX Model Inspection")
    print("=" * 60)
    model_info = check_onnx_model(args.model_path)
    results["model_info"] = model_info
    print(f"  File size:    {model_info['file_size_mb']} MB")
    print(f"  Opset:        {model_info['opset_version']}")
    print(f"  Total nodes:  {model_info['total_nodes']}")
    print(f"  Top ops:      {', '.join(f'{k}({v})' for k, v in list(model_info['op_counts'].items())[:8])}")

    # 2. Session creation time
    print("\n" + "=" * 60)
    print("2. Session Creation Time (cold-start proxy)")
    print("=" * 60)
    session_stats = profile_session_creation(args.model_path)
    results["session_creation"] = session_stats
    print(f"  Mean: {session_stats['mean_ms']:.1f}ms  "
          f"Min: {session_stats['min_ms']:.1f}ms  "
          f"Max: {session_stats['max_ms']:.1f}ms")

    # 3. Inference profiling
    print("\n" + "=" * 60)
    print("3. Inference Latency by Batch Size (CPU)")
    print("=" * 60)
    inference_stats = profile_inference(args.model_path, batch_sizes, args.iterations)
    results["inference"] = inference_stats

    # Show top operators
    if "operator_profile" in inference_stats:
        prof = inference_stats["operator_profile"]
        print(f"\n  Top operators (total compute: {prof.get('total_compute_ms', '?')}ms):")
        for op in prof.get("top_operators", [])[:10]:
            print(f"    {op['op']:>25s}: {op['total_ms']:>8.1f}ms ({op['pct_of_total']:>5.1f}%) x{op['count']}")

    # 4. Serialization overhead
    print("\n" + "=" * 60)
    print("4. Serialization Overhead (JSON .tolist() vs Binary)")
    print("=" * 60)
    serial_stats = profile_serialization_overhead(batch_sizes, min(args.iterations, 20))
    results["serialization"] = serial_stats

    # 5. Preprocessing overhead
    print("\n" + "=" * 60)
    print("5. Preprocessing Overhead")
    print("=" * 60)
    preprocess_stats = profile_preprocessing(args.iterations)
    results["preprocessing"] = preprocess_stats
    print(f"  Simple (normalize+transpose): {preprocess_stats['simple_normalize_transpose_ms']:.3f}ms")
    if preprocess_stats["openclip_full_preprocess_ms"]:
        print(f"  OpenCLIP full preprocess:     {preprocess_stats['openclip_full_preprocess_ms']:.3f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Key Findings")
    print("=" * 60)

    # Serialization is the big one
    if "batch_32" in serial_stats:
        s = serial_stats["batch_32"]
        print(f"\n  ⚠ SERIALIZATION BOTTLENECK (batch=32):")
        print(f"    JSON .tolist() takes {s['json_tolist']['mean_ms']:.1f}ms "
              f"and produces {s['json_tolist']['payload_size_kb']:.0f}KB payload")
        print(f"    Binary encoding takes {s['binary']['mean_ms']:.2f}ms "
              f"and produces {s['binary']['payload_size_kb']:.0f}KB payload")
        print(f"    → JSON is {s['json_vs_binary_speedup']}x SLOWER than binary")
        print(f"    → This overhead is added to EVERY Triton request in the benchmark")

    # Inference scaling
    if "batch_1" in inference_stats and "batch_32" in inference_stats:
        b1 = inference_stats["batch_1"]["mean_ms"]
        b32 = inference_stats["batch_32"]["mean_ms"]
        scaling = b32 / b1
        print(f"\n  Inference scaling (CPU):")
        print(f"    batch=1:  {b1:.1f}ms")
        print(f"    batch=32: {b32:.1f}ms ({scaling:.1f}x, ideal=32x)")
        print(f"    → {'Good' if scaling < 20 else 'Poor'} batch scaling efficiency")

    # Save results
    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"onnx_profile_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
