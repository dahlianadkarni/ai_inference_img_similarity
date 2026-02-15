#!/usr/bin/env python3
"""
Step 5B: 3-Way Benchmark — PyTorch vs Triton/ONNX vs Triton/TensorRT

Extends the Step 4 benchmark to include the TensorRT-optimized model.
Both Triton models (ONNX and TRT) run on the same instance, so the
comparison is perfectly fair (same GPU, same network path).

Measures for each backend:
  1. Cold-start / first-request latency
  2. Single-image inference latency (p50, p95, p99)
  3. Batch inference latency (varying batch sizes)
  4. Concurrent-request throughput (dynamic batching impact)
  5. GPU utilization snapshot (nvidia-smi, if available)
  6. Triton Prometheus metrics (queue time, compute time)

Usage:
  # Triton-only (ONNX vs TRT on same instance)
  python scripts/benchmark_tensorrt.py \
      --triton-url http://localhost:8000 --no-pytorch

  # All three backends
  python scripts/benchmark_tensorrt.py \
      --triton-url http://localhost:8000 \
      --pytorch-url http://localhost:8002

  # Custom settings
  python scripts/benchmark_tensorrt.py \
      --triton-url http://<vast-ip>:<port> \
      --iterations 50 --concurrency 16

Prerequisites:
  - Triton running with both openclip_vit_b32 and openclip_vit_b32_trt models
  - (Optional) PyTorch backend running on port 8002
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

try:
    import tritonclient.http as httpclient
    TRITONCLIENT_AVAILABLE = True
except ImportError:
    TRITONCLIENT_AVAILABLE = False
    print("⚠ tritonclient not installed. Install: pip install tritonclient[http]")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers (shared with benchmark_backends.py)
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


def is_server_up(url: str, timeout: float = 3.0) -> bool:
    """Check if a server is reachable."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def gpu_utilization_snapshot() -> Optional[Dict]:
    """Capture GPU utilization via nvidia-smi (if available)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "gpu_name": parts[0],
            "gpu_utilization_pct": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "temperature_c": float(parts[4]),
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# PyTorch benchmarks (reused from benchmark_backends.py)
# ---------------------------------------------------------------------------

def bench_pytorch_single(url: str, iterations: int) -> List[float]:
    """Benchmark single-image inference on PyTorch backend."""
    img = create_test_images(1)[0]
    b64 = image_to_base64(img)
    latencies = []

    for _ in range(3):  # warm-up
        requests.post(f"{url}/embed/base64", json={"images": [b64]})

    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/embed/base64", json={"images": [b64]})
        if r.status_code == 200:
            latencies.append(time.perf_counter() - t0)
    return latencies


def bench_pytorch_batch(url: str, batch_size: int, iterations: int) -> List[float]:
    """Benchmark batch inference on PyTorch backend."""
    imgs = create_test_images(batch_size)
    b64s = [image_to_base64(im) for im in imgs]
    latencies = []

    requests.post(f"{url}/embed/base64", json={"images": b64s})  # warm-up

    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/embed/base64", json={"images": b64s})
        if r.status_code == 200:
            latencies.append(time.perf_counter() - t0)
    return latencies


def bench_pytorch_concurrent(url: str, total_requests: int, concurrency: int) -> Dict:
    """Fire concurrent requests at PyTorch backend."""
    img = create_test_images(1)[0]
    b64 = image_to_base64(img)

    requests.post(f"{url}/embed/base64", json={"images": [b64]})  # warm-up

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


# ---------------------------------------------------------------------------
# Triton benchmarks (generic — works for both ONNX and TRT models)
# ---------------------------------------------------------------------------

def _get_triton_client(url: str) -> "httpclient.InferenceServerClient":
    """Create a Triton HTTP client from a URL."""
    url_parts = url.replace('http://', '').replace('https://', '').split(':')
    host = url_parts[0]
    port = url_parts[1] if len(url_parts) > 1 else '8000'
    return httpclient.InferenceServerClient(url=f"{host}:{port}")


def bench_triton_single(
    url: str, model_name: str, iterations: int
) -> List[float]:
    """Benchmark single-image inference on a Triton model."""
    client = _get_triton_client(url)
    img = create_test_images(1)[0]
    tensor = image_to_triton_tensor(img)
    batch = np.expand_dims(tensor, 0)

    inputs = [httpclient.InferInput("image", batch.shape, "FP32")]
    inputs[0].set_data_from_numpy(batch)
    outputs = [httpclient.InferRequestedOutput("embedding")]

    latencies = []

    for _ in range(3):  # warm-up
        try:
            client.infer(model_name, inputs, outputs=outputs)
        except Exception:
            pass

    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            client.infer(model_name, inputs, outputs=outputs)
            latencies.append(time.perf_counter() - t0)
        except Exception:
            pass
    return latencies


def bench_triton_batch(
    url: str, model_name: str, batch_size: int, iterations: int
) -> List[float]:
    """Benchmark batch inference on a Triton model."""
    client = _get_triton_client(url)
    imgs = create_test_images(batch_size)
    tensors = np.array([image_to_triton_tensor(im) for im in imgs], dtype=np.float32)

    inputs = [httpclient.InferInput("image", tensors.shape, "FP32")]
    inputs[0].set_data_from_numpy(tensors)
    outputs = [httpclient.InferRequestedOutput("embedding")]

    latencies = []

    try:  # warm-up
        client.infer(model_name, inputs, outputs=outputs)
    except Exception:
        pass

    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            client.infer(model_name, inputs, outputs=outputs)
            latencies.append(time.perf_counter() - t0)
        except Exception:
            pass
    return latencies


def bench_triton_concurrent(
    url: str, model_name: str, total_requests: int, concurrency: int
) -> Dict:
    """Fire concurrent requests at a Triton model."""
    url_parts = url.replace('http://', '').replace('https://', '').split(':')
    host = url_parts[0]
    port = url_parts[1] if len(url_parts) > 1 else '8000'

    img = create_test_images(1)[0]
    tensor = image_to_triton_tensor(img)
    batch = np.expand_dims(tensor, 0)

    # warm-up
    try:
        c = httpclient.InferenceServerClient(url=f"{host}:{port}")
        inp = [httpclient.InferInput("image", batch.shape, "FP32")]
        inp[0].set_data_from_numpy(batch)
        out = [httpclient.InferRequestedOutput("embedding")]
        c.infer(model_name, inp, outputs=out)
    except Exception:
        pass

    latencies = []
    start_wall = time.perf_counter()

    def _call():
        t0 = time.perf_counter()
        try:
            c = httpclient.InferenceServerClient(url=f"{host}:{port}")
            inp = [httpclient.InferInput("image", batch.shape, "FP32")]
            inp[0].set_data_from_numpy(batch)
            out = [httpclient.InferRequestedOutput("embedding")]
            c.infer(model_name, inp, outputs=out)
            return time.perf_counter() - t0, 200
        except Exception:
            return time.perf_counter() - t0, 500

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
# Triton Prometheus metrics
# ---------------------------------------------------------------------------

def fetch_triton_metrics(metrics_url: str, model_name: str) -> Optional[Dict]:
    """Fetch Triton Prometheus metrics for a specific model."""
    try:
        r = requests.get(metrics_url, timeout=5)
        if r.status_code != 200:
            return None

        parsed = {}
        for line in r.text.splitlines():
            if line.startswith("#") or model_name not in line:
                continue
            if "nv_inference_request_success" in line:
                parsed["inference_request_success"] = _extract_value(line)
            if "nv_inference_queue_duration_us" in line:
                parsed["queue_duration_us"] = _extract_value(line)
            if "nv_inference_compute_infer_duration_us" in line:
                parsed["compute_infer_duration_us"] = _extract_value(line)
            if "nv_inference_compute_input_duration_us" in line:
                parsed["compute_input_duration_us"] = _extract_value(line)
            if "nv_inference_compute_output_duration_us" in line:
                parsed["compute_output_duration_us"] = _extract_value(line)

        return parsed if parsed else None
    except Exception:
        return None


def _extract_value(prom_line: str) -> float:
    try:
        return float(prom_line.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")


def print_latency_table(label: str, stats: Dict):
    print(f"\n  {label}:")
    print(f"    {'Mean':>8s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}  {'Min':>8s}  {'Max':>8s}  {'N':>5s}")
    print(f"    {stats['mean_ms']:>7.1f}ms {stats['p50_ms']:>7.1f}ms "
          f"{stats['p95_ms']:>7.1f}ms {stats['p99_ms']:>7.1f}ms "
          f"{stats['min_ms']:>7.1f}ms {stats['max_ms']:>7.1f}ms  {stats['count']:>5d}")


def print_3way_comparison(results: Dict):
    """Print a 3-way comparison table: PyTorch vs ONNX vs TensorRT."""
    print_section("3-WAY COMPARISON: PyTorch vs ONNX vs TensorRT")

    backends = []
    labels = []
    if "pytorch" in results:
        backends.append("pytorch")
        labels.append("PyTorch")
    if "triton_onnx" in results:
        backends.append("triton_onnx")
        labels.append("ONNX")
    if "triton_trt" in results:
        backends.append("triton_trt")
        labels.append("TensorRT")

    if len(backends) < 2:
        print("  (Need at least 2 backends for comparison)")
        return

    # Header
    header = f"  {'Metric':<28s}"
    for label in labels:
        header += f" {label:>12s}"
    if len(backends) >= 2:
        header += f" {'TRT vs ONNX':>12s}"
    print(header)
    print(f"  {'─'*28}" + f" {'─'*12}" * len(labels) + f" {'─'*12}")

    rows = []

    # Single image latency
    _add_metric_row(rows, results, backends, labels,
                    "Single (mean)", "single", "mean_ms", "ms")
    _add_metric_row(rows, results, backends, labels,
                    "Single (p95)", "single", "p95_ms", "ms")

    # Batch latency (largest common batch)
    for bs in [32, 16, 8, 4]:
        key = f"batch_{bs}"
        has_all = all(key in results.get(b, {}) for b in backends)
        if has_all:
            _add_metric_row(rows, results, backends, labels,
                            f"Batch-{bs} (mean)", key, "mean_ms", "ms")
            # Also compute img/s
            _add_imgs_per_sec_row(rows, results, backends, labels, bs, key)
            break

    # Concurrent throughput
    _add_metric_row(rows, results, backends, labels,
                    "Throughput (req/s)", "concurrent", "throughput_rps", "")
    _add_metric_row(rows, results, backends, labels,
                    "Concurrent (p95)", "concurrent", "p95_ms", "ms")

    # Print rows
    for row in rows:
        line = f"  {row[0]:<28s}"
        for val in row[1:len(backends)+1]:
            line += f" {val:>12s}"
        if len(row) > len(backends) + 1:
            line += f" {row[-1]:>12s}"
        print(line)

    print()
    print("  ─ Negative % = TensorRT is faster than ONNX")
    print("  ─ Both Triton models run on the same GPU instance")
    if "pytorch" in results:
        print("  ─ PyTorch numbers include network latency if remote")


def _add_metric_row(rows, results, backends, labels, name, section, metric, unit):
    """Add a comparison row for a single metric."""
    row = [name]
    vals = []
    for backend in backends:
        data = results.get(backend, {}).get(section, {})
        val = data.get(metric)
        if val is not None:
            row.append(f"{val:.1f}{unit}")
            vals.append(val)
        else:
            row.append("—")
            vals.append(None)

    # TRT vs ONNX delta
    onnx_idx = backends.index("triton_onnx") if "triton_onnx" in backends else None
    trt_idx = backends.index("triton_trt") if "triton_trt" in backends else None
    if onnx_idx is not None and trt_idx is not None and vals[onnx_idx] and vals[trt_idx]:
        if "throughput" in metric:
            # Higher is better for throughput
            delta = ((vals[trt_idx] - vals[onnx_idx]) / vals[onnx_idx]) * 100
        else:
            # Lower is better for latency
            delta = ((vals[trt_idx] - vals[onnx_idx]) / vals[onnx_idx]) * 100
        row.append(f"{delta:+.1f}%")
    else:
        row.append("—")

    rows.append(row)


def _add_imgs_per_sec_row(rows, results, backends, labels, batch_size, section):
    """Add an images/sec row."""
    row = [f"Batch-{batch_size} (img/s)"]
    vals = []
    for backend in backends:
        data = results.get(backend, {}).get(section, {})
        mean_ms = data.get("mean_ms")
        if mean_ms and mean_ms > 0:
            ips = batch_size / (mean_ms / 1000)
            row.append(f"{ips:.1f}")
            vals.append(ips)
        else:
            row.append("—")
            vals.append(None)

    onnx_idx = backends.index("triton_onnx") if "triton_onnx" in backends else None
    trt_idx = backends.index("triton_trt") if "triton_trt" in backends else None
    if onnx_idx is not None and trt_idx is not None and vals[onnx_idx] and vals[trt_idx]:
        delta = ((vals[trt_idx] - vals[onnx_idx]) / vals[onnx_idx]) * 100
        row.append(f"{delta:+.1f}%")
    else:
        row.append("—")

    rows.append(row)


# ---------------------------------------------------------------------------
# Run all benchmarks for a single backend
# ---------------------------------------------------------------------------

def run_backend_benchmarks(
    backend_key: str,
    label: str,
    bench_single_fn,
    bench_batch_fn,
    bench_concurrent_fn,
    iterations: int,
    batch_sizes: List[int],
    concurrent_requests: int,
    concurrency: int,
) -> Dict:
    """Run the full benchmark suite for one backend."""
    result = {}

    print_section(f"{label} Benchmarks")

    # Single-image latency
    print(f"\n  1. Single-image latency ({iterations} iterations)...")
    lats = bench_single_fn(iterations)
    if lats:
        stats = percentile_stats(lats)
        result["single"] = stats
        print_latency_table("Single image", stats)

    # Batch latency
    for bs in batch_sizes:
        if bs <= 1:
            continue
        print(f"\n  2. Batch latency (batch_size={bs}, {iterations} iterations)...")
        lats = bench_batch_fn(bs, iterations)
        if lats:
            stats = percentile_stats(lats)
            result[f"batch_{bs}"] = stats
            print_latency_table(f"Batch {bs}", stats)
            imgs_per_sec = bs / (stats["mean_ms"] / 1000)
            print(f"    Images/sec: {imgs_per_sec:.1f}")

    # Concurrent throughput
    print(f"\n  3. Concurrent throughput ({concurrent_requests} requests, "
          f"{concurrency} workers)...")
    cstats = bench_concurrent_fn(concurrent_requests, concurrency)
    result["concurrent"] = cstats
    print(f"     Throughput: {cstats['throughput_rps']:.1f} req/s")
    if "p95_ms" in cstats:
        print(f"     p95 latency: {cstats['p95_ms']:.1f}ms")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 5B: 3-Way Benchmark — PyTorch vs Triton/ONNX vs Triton/TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pytorch-url", default="http://localhost:8002",
                        help="PyTorch backend URL")
    parser.add_argument("--triton-url", default="http://localhost:8000",
                        help="Triton server URL (serves both ONNX and TRT models)")
    parser.add_argument("--triton-metrics-url", default="http://localhost:8002/metrics",
                        help="Triton Prometheus metrics URL")
    parser.add_argument("--no-pytorch", action="store_true",
                        help="Skip PyTorch benchmarks")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Iterations per latency test (default: 20)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent workers (default: 8)")
    parser.add_argument("--concurrent-requests", type=int, default=100,
                        help="Total requests in concurrency test (default: 100)")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32",
                        help="Comma-separated batch sizes (default: 1,4,8,16,32)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-timestamped)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    now = datetime.now()
    print("=" * 68)
    print("  Step 5B: 3-Way Benchmark")
    print(f"  PyTorch vs Triton/ONNX vs Triton/TensorRT")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print(f"  Iterations         : {args.iterations}")
    print(f"  Concurrency        : {args.concurrency}")
    print(f"  Concurrent requests: {args.concurrent_requests}")
    print(f"  Batch sizes        : {batch_sizes}")

    results: Dict = {
        "timestamp": now.isoformat(),
        "benchmark_type": "step_5b_tensorrt_comparison",
        "config": vars(args),
    }

    # ── Check availability ──
    test_pytorch = not args.no_pytorch
    test_onnx = False
    test_trt = False

    if test_pytorch:
        up = is_server_up(f"{args.pytorch_url}/health")
        print(f"\n  PyTorch   ({args.pytorch_url}): {'✓ UP' if up else '✗ DOWN'}")
        if not up:
            print("    → Skipping PyTorch benchmarks")
            test_pytorch = False

    # Check Triton health
    triton_up = is_server_up(f"{args.triton_url}/v2/health/ready")
    print(f"  Triton    ({args.triton_url}): {'✓ UP' if triton_up else '✗ DOWN'}")

    if triton_up:
        # Check ONNX model
        onnx_up = is_server_up(f"{args.triton_url}/v2/models/openclip_vit_b32/ready")
        print(f"  ├─ ONNX   (openclip_vit_b32):     {'✓ LOADED' if onnx_up else '✗ NOT LOADED'}")
        test_onnx = onnx_up

        # Check TRT model
        trt_up = is_server_up(f"{args.triton_url}/v2/models/openclip_vit_b32_trt/ready")
        print(f"  └─ TRT    (openclip_vit_b32_trt):  {'✓ LOADED' if trt_up else '✗ NOT LOADED'}")
        test_trt = trt_up

    if not test_pytorch and not test_onnx and not test_trt:
        print("\n  ✗ No backends available. Start at least one server.")
        sys.exit(1)

    # ── GPU snapshot ──
    gpu = gpu_utilization_snapshot()
    if gpu:
        print_section("GPU HARDWARE")
        print(f"  GPU:         {gpu['gpu_name']}")
        print(f"  Memory:      {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB")
        print(f"  Utilization: {gpu['gpu_utilization_pct']:.0f}%")
        print(f"  Temperature: {gpu['temperature_c']:.0f}°C")
        results["gpu"] = gpu

    # ── PyTorch benchmarks ──
    if test_pytorch:
        results["pytorch"] = run_backend_benchmarks(
            backend_key="pytorch",
            label="PyTorch",
            bench_single_fn=lambda iters: bench_pytorch_single(args.pytorch_url, iters),
            bench_batch_fn=lambda bs, iters: bench_pytorch_batch(args.pytorch_url, bs, iters),
            bench_concurrent_fn=lambda total, conc: bench_pytorch_concurrent(
                args.pytorch_url, total, conc),
            iterations=args.iterations,
            batch_sizes=batch_sizes,
            concurrent_requests=args.concurrent_requests,
            concurrency=args.concurrency,
        )

    # ── Triton ONNX benchmarks ──
    if test_onnx:
        results["triton_onnx"] = run_backend_benchmarks(
            backend_key="triton_onnx",
            label="Triton / ONNX Runtime",
            bench_single_fn=lambda iters: bench_triton_single(
                args.triton_url, "openclip_vit_b32", iters),
            bench_batch_fn=lambda bs, iters: bench_triton_batch(
                args.triton_url, "openclip_vit_b32", bs, iters),
            bench_concurrent_fn=lambda total, conc: bench_triton_concurrent(
                args.triton_url, "openclip_vit_b32", total, conc),
            iterations=args.iterations,
            batch_sizes=batch_sizes,
            concurrent_requests=args.concurrent_requests,
            concurrency=args.concurrency,
        )

        # Fetch ONNX metrics
        metrics = fetch_triton_metrics(args.triton_metrics_url, "openclip_vit_b32")
        if metrics:
            results["triton_onnx"]["server_metrics"] = metrics
            print(f"\n  ONNX Metrics:")
            print(f"    Requests:     {metrics.get('inference_request_success', 'N/A')}")
            print(f"    Compute (μs): {metrics.get('compute_infer_duration_us', 'N/A')}")

    # ── Triton TensorRT benchmarks ──
    if test_trt:
        results["triton_trt"] = run_backend_benchmarks(
            backend_key="triton_trt",
            label="Triton / TensorRT (FP16)",
            bench_single_fn=lambda iters: bench_triton_single(
                args.triton_url, "openclip_vit_b32_trt", iters),
            bench_batch_fn=lambda bs, iters: bench_triton_batch(
                args.triton_url, "openclip_vit_b32_trt", bs, iters),
            bench_concurrent_fn=lambda total, conc: bench_triton_concurrent(
                args.triton_url, "openclip_vit_b32_trt", total, conc),
            iterations=args.iterations,
            batch_sizes=batch_sizes,
            concurrent_requests=args.concurrent_requests,
            concurrency=args.concurrency,
        )

        # Fetch TRT metrics
        metrics = fetch_triton_metrics(args.triton_metrics_url, "openclip_vit_b32_trt")
        if metrics:
            results["triton_trt"]["server_metrics"] = metrics
            print(f"\n  TensorRT Metrics:")
            print(f"    Requests:     {metrics.get('inference_request_success', 'N/A')}")
            print(f"    Compute (μs): {metrics.get('compute_infer_duration_us', 'N/A')}")

    # ── GPU snapshot (after) ──
    gpu_after = gpu_utilization_snapshot()
    if gpu_after:
        print_section("GPU STATUS (after benchmarks)")
        print(f"  Memory:      {gpu_after['memory_used_mb']:.0f} / {gpu_after['memory_total_mb']:.0f} MB")
        print(f"  Utilization: {gpu_after['gpu_utilization_pct']:.0f}%")
        print(f"  Temperature: {gpu_after['temperature_c']:.0f}°C")
        results["gpu_after"] = gpu_after

    # ── 3-Way comparison ──
    print_3way_comparison(results)

    # ── Analysis ──
    print_section("ANALYSIS: TensorRT vs ONNX Runtime")
    print("""
  TensorRT Advantages:
    • Kernel fusion: combines multiple ops into single optimized kernels
    • FP16 precision: ~2x throughput with minimal accuracy impact on ViT
    • Layer-level optimization: auto-tunes for specific GPU architecture
    • Reduced memory bandwidth: smaller FP16 tensors move faster

  TensorRT Trade-offs:
    • GPU-specific: engine must be rebuilt for each GPU architecture
    • Longer startup: 2-10 min build on first start (amortized)
    • Less portable: can't move .plan files between GPU types
    • Debugging harder: optimized graph is opaque

  When TensorRT Wins Most:
    • Batch inference (FP16 shines with large tensors)
    • High-concurrency serving (more req/s per GPU-dollar)
    • Production deployment on fixed GPU type

  When ONNX/PyTorch May Be Better:
    • Development/debugging (readable graph, easy profiling)
    • Mixed GPU fleet (need portability across GPU types)
    • Very low latency single requests (TRT overhead less impactful)
""")

    # ── Save results ──
    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = PROJECT_ROOT / "benchmark_results"
        output_path = out_dir / f"tensorrt_benchmark_{now.strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to {output_path}")

    print(f"\n{'='*68}")
    print("  Step 5B benchmark complete.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
