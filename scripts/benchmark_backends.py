#!/usr/bin/env python3
"""
Benchmark: PyTorch FastAPI vs. NVIDIA Triton Inference Server

Measures and compares:
  1. Model loading / cold-start latency
  2. Single-image inference latency (p50, p95, p99)
  3. Batch inference latency (varying batch sizes)
  4. Concurrent-request throughput (dynamic batching impact)
  5. GPU utilization snapshot (nvidia-smi, if available)
  6. Triton Prometheus metrics (queue time, compute time)

Usage:
  # Both backends (default)
  python scripts/benchmark_backends.py

  # Only PyTorch
  python scripts/benchmark_backends.py --backend pytorch

  # Only Triton
  python scripts/benchmark_backends.py --backend triton

  # Custom URLs
  python scripts/benchmark_backends.py \
      --pytorch-url http://localhost:8002 \
      --triton-url  http://localhost:8003

  # Adjust concurrency and iterations
  python scripts/benchmark_backends.py --iterations 50 --concurrency 16

    # Save results to JSON (also prints to stdout)
    python scripts/benchmark_backends.py --output benchmark_results.json

    # By default, it ALSO saves a timestamped JSON file under ./benchmark_results/
    python scripts/benchmark_backends.py

Prerequisites:
  - PyTorch backend running on port 8002 (python -m src.inference_service.server)
  - Triton backend running on port 8003 (./build_triton_local.sh or remote)
"""

import argparse
import io
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_images(count: int, size: int = 224) -> List[np.ndarray]:
    """Create synthetic test images (deterministic for reproducibility)."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(count)]


def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    import base64
    pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def image_to_triton_tensor(img_array: np.ndarray) -> np.ndarray:
    """Preprocess image for Triton (CHW float32, [0,1])."""
    t = img_array.astype(np.float32) / 255.0
    return np.transpose(t, (2, 0, 1))  # HWC -> CHW


def percentile_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute p50, p95, p99, mean, min, max from a list of latencies."""
    a = np.array(latencies)
    return {
        "mean_ms": float(np.mean(a) * 1000),
        "p50_ms": float(np.percentile(a, 50) * 1000),
        "p95_ms": float(np.percentile(a, 95) * 1000),
        "p99_ms": float(np.percentile(a, 99) * 1000),
        "min_ms": float(np.min(a) * 1000),
        "max_ms": float(np.max(a) * 1000),
        "count": len(latencies),
    }


def is_server_up(url: str, timeout: float = 2.0) -> bool:
    """Check if a server is reachable."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cold-start measurement
# ---------------------------------------------------------------------------

def measure_cold_start_pytorch(url: str) -> Optional[Dict]:
    """
    Measure PyTorch cold-start by timing the first /embed/base64 call.

    In practice the model is loaded at server startup, so this measures
    "first-request" latency which may include JIT warm-up.
    """
    img = create_test_images(1)[0]
    b64 = image_to_base64(img)

    start = time.perf_counter()
    try:
        r = requests.post(f"{url}/embed/base64", json={"images": [b64]}, timeout=120)
        elapsed = time.perf_counter() - start
        if r.status_code == 200:
            return {"first_request_ms": round(elapsed * 1000, 2)}
    except Exception as e:
        print(f"  ✗ cold-start request failed: {e}")
    return None


def measure_cold_start_triton(url: str) -> Optional[Dict]:
    """
    Measure Triton cold-start by timing the first /v2/models/.../infer call.
    """
    img = create_test_images(1)[0]
    tensor = image_to_triton_tensor(img)
    batch = np.expand_dims(tensor, 0)

    payload = {
        "inputs": [{"name": "image", "shape": list(batch.shape), "datatype": "FP32",
                     "data": batch.flatten().tolist()}],
        "outputs": [{"name": "embedding"}],
    }

    start = time.perf_counter()
    try:
        r = requests.post(f"{url}/v2/models/openclip_vit_b32/infer", json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        if r.status_code == 200:
            return {"first_request_ms": round(elapsed * 1000, 2)}
    except Exception as e:
        print(f"  ✗ cold-start request failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Single-image latency
# ---------------------------------------------------------------------------

def bench_single_pytorch(url: str, iterations: int) -> List[float]:
    """Benchmark single-image inference on PyTorch backend."""
    img = create_test_images(1)[0]
    b64 = image_to_base64(img)
    latencies = []

    # Warm-up
    for _ in range(3):
        requests.post(f"{url}/embed/base64", json={"images": [b64]})

    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/embed/base64", json={"images": [b64]})
        latencies.append(time.perf_counter() - t0)
        if r.status_code != 200:
            latencies.pop()
    return latencies


def bench_single_triton(url: str, iterations: int) -> List[float]:
    """Benchmark single-image inference on Triton backend."""
    img = create_test_images(1)[0]
    tensor = image_to_triton_tensor(img)
    batch = np.expand_dims(tensor, 0)
    payload = {
        "inputs": [{"name": "image", "shape": list(batch.shape), "datatype": "FP32",
                     "data": batch.flatten().tolist()}],
        "outputs": [{"name": "embedding"}],
    }
    latencies = []

    # Warm-up
    for _ in range(3):
        requests.post(f"{url}/v2/models/openclip_vit_b32/infer", json=payload)

    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/v2/models/openclip_vit_b32/infer", json=payload)
        latencies.append(time.perf_counter() - t0)
        if r.status_code != 200:
            latencies.pop()
    return latencies


# ---------------------------------------------------------------------------
# Batch inference latency
# ---------------------------------------------------------------------------

def bench_batch_pytorch(url: str, batch_size: int, iterations: int) -> List[float]:
    """Benchmark batch inference on PyTorch backend."""
    imgs = create_test_images(batch_size)
    b64s = [image_to_base64(im) for im in imgs]
    latencies = []

    # Warm-up
    requests.post(f"{url}/embed/base64", json={"images": b64s})

    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/embed/base64", json={"images": b64s})
        latencies.append(time.perf_counter() - t0)
        if r.status_code != 200:
            latencies.pop()
    return latencies


def bench_batch_triton(url: str, batch_size: int, iterations: int) -> List[float]:
    """Benchmark batch inference on Triton backend."""
    imgs = create_test_images(batch_size)
    tensors = np.array([image_to_triton_tensor(im) for im in imgs], dtype=np.float32)
    payload = {
        "inputs": [{"name": "image", "shape": list(tensors.shape), "datatype": "FP32",
                     "data": tensors.flatten().tolist()}],
        "outputs": [{"name": "embedding"}],
    }
    latencies = []

    # Warm-up
    requests.post(f"{url}/v2/models/openclip_vit_b32/infer", json=payload)

    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/v2/models/openclip_vit_b32/infer", json=payload)
        latencies.append(time.perf_counter() - t0)
        if r.status_code != 200:
            latencies.pop()
    return latencies


# ---------------------------------------------------------------------------
# Concurrent throughput (tests dynamic batching impact)
# ---------------------------------------------------------------------------

def bench_concurrent_pytorch(url: str, total_requests: int, concurrency: int) -> Dict:
    """Fire concurrent single-image requests at PyTorch backend."""
    img = create_test_images(1)[0]
    b64 = image_to_base64(img)

    # Warm-up
    requests.post(f"{url}/embed/base64", json={"images": [b64]})

    latencies = []
    start_wall = time.perf_counter()

    def _call():
        t0 = time.perf_counter()
        r = requests.post(f"{url}/embed/base64", json={"images": [b64]})
        return time.perf_counter() - t0, r.status_code

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_call) for _ in range(total_requests)]
        for f in as_completed(futures):
            elapsed, status = f.result()
            if status == 200:
                latencies.append(elapsed)

    wall_time = time.perf_counter() - start_wall
    stats = percentile_stats(latencies) if latencies else {}
    stats["wall_time_s"] = round(wall_time, 3)
    stats["throughput_rps"] = round(len(latencies) / wall_time, 2) if wall_time > 0 else 0
    stats["concurrency"] = concurrency
    stats["total_requests"] = total_requests
    stats["successful"] = len(latencies)
    return stats


def bench_concurrent_triton(url: str, total_requests: int, concurrency: int) -> Dict:
    """Fire concurrent single-image requests at Triton backend (tests dynamic batching)."""
    img = create_test_images(1)[0]
    tensor = image_to_triton_tensor(img)
    batch = np.expand_dims(tensor, 0)
    payload = {
        "inputs": [{"name": "image", "shape": list(batch.shape), "datatype": "FP32",
                     "data": batch.flatten().tolist()}],
        "outputs": [{"name": "embedding"}],
    }

    # Warm-up
    requests.post(f"{url}/v2/models/openclip_vit_b32/infer", json=payload)

    latencies = []
    start_wall = time.perf_counter()

    def _call():
        t0 = time.perf_counter()
        r = requests.post(f"{url}/v2/models/openclip_vit_b32/infer", json=payload)
        return time.perf_counter() - t0, r.status_code

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_call) for _ in range(total_requests)]
        for f in as_completed(futures):
            elapsed, status = f.result()
            if status == 200:
                latencies.append(elapsed)

    wall_time = time.perf_counter() - start_wall
    stats = percentile_stats(latencies) if latencies else {}
    stats["wall_time_s"] = round(wall_time, 3)
    stats["throughput_rps"] = round(len(latencies) / wall_time, 2) if wall_time > 0 else 0
    stats["concurrency"] = concurrency
    stats["total_requests"] = total_requests
    stats["successful"] = len(latencies)
    return stats


# ---------------------------------------------------------------------------
# GPU utilization snapshot
# ---------------------------------------------------------------------------

def gpu_utilization_snapshot() -> Optional[Dict]:
    """Capture GPU utilization and specs via nvidia-smi (if available)."""
    try:
        # Get utilization and memory stats
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        parts = [x.strip() for x in out.strip().split(",")]
        
        # Get GPU model name and driver version
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            timeout=5, text=True,
        ).strip().split(",")
        
        return {
            "gpu_name": gpu_info[0].strip(),
            "gpu_memory_total_mb": gpu_info[1].strip(),
            "gpu_driver_version": gpu_info[2].strip(),
            "gpu_utilization_pct": float(parts[0]),
            "memory_utilization_pct": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "temperature_c": float(parts[4]),
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# Triton Prometheus metrics
# ---------------------------------------------------------------------------

def fetch_triton_metrics(metrics_url: str) -> Optional[Dict]:
    """Fetch Triton Prometheus metrics (queue time, compute time, request count)."""
    try:
        r = requests.get(metrics_url, timeout=5)
        if r.status_code != 200:
            return None

        text = r.text
        parsed = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            # Key Triton metrics
            if "nv_inference_request_success" in line and "openclip_vit_b32" in line:
                parsed["inference_request_success"] = _extract_value(line)
            if "nv_inference_queue_duration_us" in line and "openclip_vit_b32" in line:
                parsed["queue_duration_us"] = _extract_value(line)
            if "nv_inference_compute_infer_duration_us" in line and "openclip_vit_b32" in line:
                parsed["compute_infer_duration_us"] = _extract_value(line)
            if "nv_inference_compute_input_duration_us" in line and "openclip_vit_b32" in line:
                parsed["compute_input_duration_us"] = _extract_value(line)
            if "nv_inference_compute_output_duration_us" in line and "openclip_vit_b32" in line:
                parsed["compute_output_duration_us"] = _extract_value(line)

        return parsed if parsed else None
    except Exception:
        return None


def _extract_value(prom_line: str) -> float:
    """Extract numeric value from a Prometheus metric line."""
    try:
        return float(prom_line.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def print_latency_table(label: str, stats: Dict):
    print(f"\n  {label}:")
    print(f"    {'Mean':>8s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}  {'Min':>8s}  {'Max':>8s}  {'N':>5s}")
    print(f"    {stats['mean_ms']:>7.1f}ms {stats['p50_ms']:>7.1f}ms "
          f"{stats['p95_ms']:>7.1f}ms {stats['p99_ms']:>7.1f}ms "
          f"{stats['min_ms']:>7.1f}ms {stats['max_ms']:>7.1f}ms  {stats['count']:>5d}")


def print_comparison_summary(results: Dict):
    """Print a side-by-side comparison table."""
    print_section("COMPARISON SUMMARY")

    has_pytorch = "pytorch" in results
    has_triton = "triton" in results

    if not (has_pytorch and has_triton):
        print("  (Only one backend tested — skipping side-by-side comparison)")
        return

    rows = []

    # Single-image latency
    if "single" in results.get("pytorch", {}) and "single" in results.get("triton", {}):
        pt = results["pytorch"]["single"]
        tr = results["triton"]["single"]
        rows.append(("Single image (mean)", f"{pt['mean_ms']:.1f}ms", f"{tr['mean_ms']:.1f}ms",
                      f"{((tr['mean_ms'] - pt['mean_ms']) / pt['mean_ms'] * 100):+.1f}%"))
        rows.append(("Single image (p95)", f"{pt['p95_ms']:.1f}ms", f"{tr['p95_ms']:.1f}ms",
                      f"{((tr['p95_ms'] - pt['p95_ms']) / pt['p95_ms'] * 100):+.1f}%"))

    # Batch latency (largest batch tested)
    for bs in [32, 16, 8, 4]:
        key = f"batch_{bs}"
        if key in results.get("pytorch", {}) and key in results.get("triton", {}):
            pt = results["pytorch"][key]
            tr = results["triton"][key]
            rows.append((f"Batch {bs} (mean)", f"{pt['mean_ms']:.1f}ms", f"{tr['mean_ms']:.1f}ms",
                          f"{((tr['mean_ms'] - pt['mean_ms']) / pt['mean_ms'] * 100):+.1f}%"))
            break

    # Concurrent throughput
    if "concurrent" in results.get("pytorch", {}) and "concurrent" in results.get("triton", {}):
        pt = results["pytorch"]["concurrent"]
        tr = results["triton"]["concurrent"]
        rows.append(("Throughput (req/s)", f"{pt['throughput_rps']:.1f}", f"{tr['throughput_rps']:.1f}",
                      f"{((tr['throughput_rps'] - pt['throughput_rps']) / pt['throughput_rps'] * 100):+.1f}%"))
        rows.append(("Concurrent p95", f"{pt['p95_ms']:.1f}ms", f"{tr['p95_ms']:.1f}ms",
                      f"{((tr['p95_ms'] - pt['p95_ms']) / pt['p95_ms'] * 100):+.1f}%"))

    # Cold start
    if "cold_start" in results.get("pytorch", {}) and "cold_start" in results.get("triton", {}):
        pt_cs = results["pytorch"]["cold_start"]["first_request_ms"]
        tr_cs = results["triton"]["cold_start"]["first_request_ms"]
        rows.append(("Cold-start (1st req)", f"{pt_cs:.0f}ms", f"{tr_cs:.0f}ms",
                      f"{((tr_cs - pt_cs) / pt_cs * 100):+.1f}%"))

    # Print table
    print(f"\n  {'Metric':<25s} {'PyTorch':>12s} {'Triton':>12s} {'Δ':>10s}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10}")
    for row in rows:
        print(f"  {row[0]:<25s} {row[1]:>12s} {row[2]:>12s} {row[3]:>10s}")

    print()
    print("  Negative Δ = Triton is faster.  Positive Δ = PyTorch is faster.")
    print("  Dynamic batching benefits show most clearly under concurrency.")


def default_output_path(now: Optional[datetime] = None) -> Path:
    """Default output path for benchmark results (timestamped, non-overwriting)."""
    if now is None:
        now = datetime.now()
    out_dir = PROJECT_ROOT / "benchmark_results"
    filename = f"benchmark_results_{now.strftime('%Y%m%d_%H%M%S')}.json"
    return out_dir / filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch FastAPI vs. NVIDIA Triton Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=["pytorch", "triton", "both"], default="both",
                        help="Which backend(s) to benchmark (default: both)")
    parser.add_argument("--pytorch-url", default="http://localhost:8002",
                        help="PyTorch backend URL")
    parser.add_argument("--triton-url", default="http://localhost:8003",
                        help="Triton backend URL")
    parser.add_argument("--triton-metrics-url", default="http://localhost:8005/metrics",
                        help="Triton Prometheus metrics URL")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Iterations per latency test (default: 20)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent workers for throughput test (default: 8)")
    parser.add_argument("--concurrent-requests", type=int, default=50,
                        help="Total requests in concurrency test (default: 50)")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32",
                        help="Comma-separated batch sizes to test (default: 1,4,8,16,32)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to save results JSON. If omitted, saves to ./benchmark_results/ "
            "with a timestamped filename"
        ),
    )
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    test_pytorch = args.backend in ("pytorch", "both")
    test_triton = args.backend in ("triton", "both")

    print("=" * 64)
    print("  Inference Backend Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    print(f"  Iterations per test : {args.iterations}")
    print(f"  Concurrency         : {args.concurrency}")
    print(f"  Concurrent requests : {args.concurrent_requests}")
    print(f"  Batch sizes         : {batch_sizes}")

    now = datetime.now()
    results: Dict = {"timestamp": now.isoformat(), "config": vars(args)}

    # ---- Check server availability ----
    if test_pytorch:
        up = is_server_up(f"{args.pytorch_url}/health")
        print(f"\n  PyTorch ({args.pytorch_url}): {'✓ UP' if up else '✗ DOWN'}")
        if not up:
            print("    → Skipping PyTorch benchmarks")
            test_pytorch = False

    if test_triton:
        up = is_server_up(f"{args.triton_url}/v2/health/ready")
        print(f"  Triton  ({args.triton_url}): {'✓ UP' if up else '✗ DOWN'}")
        if not up:
            print("    → Skipping Triton benchmarks")
            test_triton = False

    if not test_pytorch and not test_triton:
        print("\n  ✗ No backends available. Start at least one server first.")
        sys.exit(1)

    # ---- GPU snapshot (before) ----
    gpu = gpu_utilization_snapshot()
    if gpu:
        print_section("GPU HARDWARE & STATUS (before benchmarks)")
        print(f"  GPU Model       : {gpu['gpu_name']}")
        print(f"  Total VRAM      : {gpu['gpu_memory_total_mb']}")
        print(f"  Driver Version  : {gpu['gpu_driver_version']}")
        print(f"  GPU Utilization : {gpu['gpu_utilization_pct']:.0f}%")
        print(f"  Memory Used     : {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB")
        print(f"  Temperature     : {gpu['temperature_c']:.0f}°C")
        results["gpu_specs"] = {
            "name": gpu['gpu_name'],
            "vram_total": gpu['gpu_memory_total_mb'],
            "driver_version": gpu['gpu_driver_version'],
        }
        results["gpu_before"] = gpu
    else:
        print("\n  ℹ nvidia-smi not available (CPU testing)")

    # ==================================================================
    # PyTorch Benchmarks
    # ==================================================================
    if test_pytorch:
        results["pytorch"] = {}

        print_section("PyTorch Backend Benchmarks")

        # Cold start
        print("\n  1. Cold-start / first-request latency...")
        cs = measure_cold_start_pytorch(args.pytorch_url)
        if cs:
            print(f"     First request: {cs['first_request_ms']:.0f}ms")
            results["pytorch"]["cold_start"] = cs

        # Single-image latency
        print(f"\n  2. Single-image latency ({args.iterations} iterations)...")
        lats = bench_single_pytorch(args.pytorch_url, args.iterations)
        if lats:
            stats = percentile_stats(lats)
            results["pytorch"]["single"] = stats
            print_latency_table("Single image", stats)

        # Batch latency
        for bs in batch_sizes:
            if bs <= 1:
                continue
            print(f"\n  3. Batch latency (batch_size={bs}, {args.iterations} iterations)...")
            lats = bench_batch_pytorch(args.pytorch_url, bs, args.iterations)
            if lats:
                stats = percentile_stats(lats)
                results["pytorch"][f"batch_{bs}"] = stats
                print_latency_table(f"Batch {bs}", stats)
                imgs_per_sec = bs / (stats["mean_ms"] / 1000)
                print(f"    Images/sec: {imgs_per_sec:.1f}")

        # Concurrent throughput
        print(f"\n  4. Concurrent throughput ({args.concurrent_requests} requests, "
              f"{args.concurrency} workers)...")
        cstats = bench_concurrent_pytorch(args.pytorch_url, args.concurrent_requests, args.concurrency)
        results["pytorch"]["concurrent"] = cstats
        print(f"     Throughput: {cstats['throughput_rps']:.1f} req/s")
        if "p95_ms" in cstats:
            print(f"     p95 latency: {cstats['p95_ms']:.1f}ms")

    # ==================================================================
    # Triton Benchmarks
    # ==================================================================
    if test_triton:
        results["triton"] = {}

        print_section("Triton Backend Benchmarks")

        # Cold start
        print("\n  1. Cold-start / first-request latency...")
        cs = measure_cold_start_triton(args.triton_url)
        if cs:
            print(f"     First request: {cs['first_request_ms']:.0f}ms")
            results["triton"]["cold_start"] = cs

        # Single-image latency
        print(f"\n  2. Single-image latency ({args.iterations} iterations)...")
        lats = bench_single_triton(args.triton_url, args.iterations)
        if lats:
            stats = percentile_stats(lats)
            results["triton"]["single"] = stats
            print_latency_table("Single image", stats)

        # Batch latency
        for bs in batch_sizes:
            if bs <= 1:
                continue
            print(f"\n  3. Batch latency (batch_size={bs}, {args.iterations} iterations)...")
            lats = bench_batch_triton(args.triton_url, bs, args.iterations)
            if lats:
                stats = percentile_stats(lats)
                results["triton"][f"batch_{bs}"] = stats
                print_latency_table(f"Batch {bs}", stats)
                imgs_per_sec = bs / (stats["mean_ms"] / 1000)
                print(f"    Images/sec: {imgs_per_sec:.1f}")

        # Concurrent throughput
        print(f"\n  4. Concurrent throughput ({args.concurrent_requests} requests, "
              f"{args.concurrency} workers)...")
        cstats = bench_concurrent_triton(args.triton_url, args.concurrent_requests, args.concurrency)
        results["triton"]["concurrent"] = cstats
        print(f"     Throughput: {cstats['throughput_rps']:.1f} req/s")
        if "p95_ms" in cstats:
            print(f"     p95 latency: {cstats['p95_ms']:.1f}ms")

        # Triton Prometheus metrics
        print(f"\n  5. Triton server metrics...")
        metrics = fetch_triton_metrics(args.triton_metrics_url)
        if metrics:
            results["triton"]["server_metrics"] = metrics
            print(f"     Successful requests : {metrics.get('inference_request_success', 'N/A')}")
            print(f"     Queue duration (μs) : {metrics.get('queue_duration_us', 'N/A')}")
            print(f"     Compute time (μs)   : {metrics.get('compute_infer_duration_us', 'N/A')}")
        else:
            print(f"     ℹ Metrics not available at {args.triton_metrics_url}")

    # ---- GPU snapshot (after) ----
    gpu_after = gpu_utilization_snapshot()
    if gpu_after:
        print_section("GPU STATUS (after benchmarks)")
        print(f"  GPU Model       : {gpu_after['gpu_name']}")
        print(f"  GPU Utilization : {gpu_after['gpu_utilization_pct']:.0f}%")
        print(f"  Memory Used     : {gpu_after['memory_used_mb']:.0f} / {gpu_after['memory_total_mb']:.0f} MB")
        print(f"  Temperature     : {gpu_after['temperature_c']:.0f}°C")
        results["gpu_after"] = gpu_after

    # ---- Comparison ----
    print_comparison_summary(results)

    # ---- Trade-offs Analysis ----
    print_section("TRADE-OFF ANALYSIS")
    print("""
  Model Loading Latency:
    • PyTorch loads model weights into GPU memory at server start (~5-15s).
    • Triton also loads at start, but supports model versioning and hot-swap.
    • Both have comparable cold-start; Triton adds overhead for ONNX graph
      optimization on first load but amortizes over the server lifetime.

  Dynamic Batching:
    • PyTorch: manual batching only — client must build batches explicitly.
    • Triton: automatic dynamic batching — concurrent single requests are
      grouped into batches transparently, improving GPU utilization.
    • Benefit: under concurrent load, Triton can serve 2-3x more requests/sec.
    • Trade-off: max_queue_delay adds latency to each request (default 5ms).

  GPU Utilization:
    • Single requests underutilize the GPU regardless of backend.
    • Triton's dynamic batching fills GPU compute units more efficiently.
    • Larger batch sizes = higher utilization but more VRAM.
    • max_batch_size=32 with ViT-B-32 FP32 uses ~2.5GB VRAM.

  Cost Efficiency:
    • GPU instances cost $0.15-0.80/hr.
    • Higher throughput = fewer GPU-hours per job = lower cost.
    • Triton's batching can cut per-image cost by 50-70% under load.
""")

    # ---- Save results (default: yes) ----
    output_path = Path(args.output) if args.output else default_output_path(now)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ Results saved to {output_path}")
        results["results_file"] = str(output_path)
    except OSError as e:
        print(f"\n  ✗ Failed to write results file to {output_path}: {e}")

    print(f"\n{'='*64}")
    print("  Benchmark complete.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
