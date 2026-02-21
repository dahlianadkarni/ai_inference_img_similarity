#!/usr/bin/env python3
"""
Step 7: 5-Way Protocol Comparison Benchmark.

Benchmarks all protocol/backend combinations on the same GPU instance
to isolate transport overhead from GPU compute time:

  1. PyTorch FastAPI        — HTTP/1.1 + base64 JPEG  (~10KB/image)
  2. Triton ONNX  HTTP      — HTTP/1.1 + binary FP32  (~602KB/image)
  3. Triton ONNX  gRPC      — HTTP/2   + binary FP32  (~602KB/image)
  4. Triton TRT   HTTP      — HTTP/1.1 + binary FP32  (~602KB/image)
  5. Triton TRT   gRPC      — HTTP/2   + binary FP32  (~602KB/image)

GPU compute is the same for ONNX-HTTP vs ONNX-gRPC (and TRT-HTTP vs TRT-gRPC),
so latency deltas within each pair are pure transport overhead.

Usage:
  # Local (run build_triton_local.sh for Triton, start PyTorch on 8002)
  python scripts/benchmark_grpc_vs_http.py

  # Remote Vast.ai — step6a docker-compose (all backends on one machine)
  python scripts/benchmark_grpc_vs_http.py \\
    --pytorch-url      http://1.2.3.4:8002 \\
    --onnx-http-url    http://1.2.3.4:8010 \\
    --onnx-grpc-url    1.2.3.4:8011 \\
    --onnx-metrics-url http://1.2.3.4:8012 \\
    --trt-http-url     http://1.2.3.4:8020 \\
    --trt-grpc-url     1.2.3.4:8021 \\
    --trt-metrics-url  http://1.2.3.4:8022 \\
    --iterations 30

  # Skip backends you don't have running:
  python scripts/benchmark_grpc_vs_http.py --skip-trt

Output:
  benchmark_results/step7_5way_<timestamp>.json
  Printed comparison tables to stdout

Port reference (step6a docker-compose):
  PyTorch HTTP  : 8002
  ONNX HTTP     : 8010   ONNX gRPC  : 8011   ONNX Metrics : 8012
  TRT  HTTP     : 8020   TRT  gRPC  : 8021   TRT  Metrics : 8022
"""

import argparse
import base64
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------

try:
    import tritonclient.http as httpclient
    TRITONCLIENT_HTTP_AVAILABLE = True
except ImportError:
    TRITONCLIENT_HTTP_AVAILABLE = False
    print("ERROR: tritonclient[http] not installed. Run: pip install 'tritonclient[http,grpc]'")
    sys.exit(1)

try:
    import tritonclient.grpc as grpcclient
    TRITONCLIENT_GRPC_AVAILABLE = True
except ImportError:
    TRITONCLIENT_GRPC_AVAILABLE = False
    print("ERROR: tritonclient[grpc] not installed. Run: pip install 'tritonclient[http,grpc]'")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Defaults (local, matching local docker-compose / build_triton_local.sh)
# ---------------------------------------------------------------------------
DEFAULT_PYTORCH_URL      = "http://localhost:8002"
DEFAULT_ONNX_HTTP_URL    = "http://localhost:8003"
DEFAULT_ONNX_GRPC_URL    = "localhost:8004"
DEFAULT_ONNX_METRICS_URL = "http://localhost:8005"
DEFAULT_TRT_HTTP_URL     = "http://localhost:8003"   # TRT uses same container port locally
DEFAULT_TRT_GRPC_URL     = "localhost:8004"
DEFAULT_TRT_METRICS_URL  = "http://localhost:8005"

ONNX_MODEL_NAME = "openclip_vit_b32"
TRT_MODEL_NAME  = "openclip_vit_b32_trt"

BATCH_SIZES        = [1, 4, 8, 16, 32]
CONCURRENCY_LEVELS = [1, 8, 16]
WARMUP_ITERS       = 5
REQUEST_TIMEOUT    = 120  # seconds
WARMUP_TIMEOUT     = 300  # TRT engine compilation can take minutes


# ---------------------------------------------------------------------------
# Image / tensor helpers
# ---------------------------------------------------------------------------

def make_jpeg_b64(width: int = 224, height: int = 224) -> str:
    """Random image as base64-encoded JPEG (~10KB — what PyTorch backend accepts)."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def make_tensor_batch(batch_size: int) -> np.ndarray:
    """Random float32 batch [B, 3, 224, 224] in [0,1] (~602KB/image — what Triton accepts)."""
    return np.random.rand(batch_size, 3, 224, 224).astype(np.float32)


# ---------------------------------------------------------------------------
# Triton metrics parsing
# ---------------------------------------------------------------------------

def parse_triton_metrics(metrics_url: str, model_name: str) -> Dict:
    """Fetch Triton Prometheus metrics and extract per-model counters."""
    try:
        text = requests.get(f"{metrics_url}/metrics", timeout=5).text
    except Exception:
        return {}
    metrics = {}
    for key, pattern in [
        ("success", rf'nv_inference_request_success{{model="{model_name}".*?}}\s+([\d.]+)'),
        ("compute", rf'nv_inference_compute_infer_duration_us{{model="{model_name}".*?}}\s+([\d.]+)'),
        ("request", rf'nv_inference_request_duration_us{{model="{model_name}".*?}}\s+([\d.]+)'),
    ]:
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def compute_server_metrics(m0: Dict, m1: Dict) -> Optional[Dict]:
    """Derive per-request GPU compute and total request ms from metric deltas."""
    if not (m0 and m1 and "success" in m0 and "success" in m1):
        return None
    reqs = m1["success"] - m0["success"]
    if reqs <= 0:
        return None
    return {
        "gpu_compute_ms":    (m1.get("compute", 0) - m0.get("compute", 0)) / reqs / 1000,
        "server_request_ms": (m1.get("request", 0) - m0.get("request", 0)) / reqs / 1000,
        "request_count":     reqs,
    }


# ---------------------------------------------------------------------------
# Per-protocol inference callables
# ---------------------------------------------------------------------------

def _pytorch_infer(pytorch_url: str, batch_b64: List[str]) -> np.ndarray:
    """Single call to PyTorch FastAPI /embed/base64."""
    resp = requests.post(
        f"{pytorch_url}/embed/base64",
        json={"images": batch_b64},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float32)


def _triton_http_infer(host_port: str, model_name: str, tensor: np.ndarray) -> np.ndarray:
    """Single call to Triton via HTTP binary protocol (tritonclient.http)."""
    client  = httpclient.InferenceServerClient(url=host_port, verbose=False)
    inputs  = [httpclient.InferInput("image", tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(tensor)
    outputs = [httpclient.InferRequestedOutput("embedding")]
    result  = client.infer(model_name, inputs, outputs=outputs)
    return result.as_numpy("embedding")


def _triton_grpc_infer(grpc_url: str, model_name: str, tensor: np.ndarray) -> np.ndarray:
    """Single call to Triton via gRPC (tritonclient.grpc)."""
    client  = grpcclient.InferenceServerClient(url=grpc_url, verbose=False)
    inputs  = [grpcclient.InferInput("image", tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(tensor)
    outputs = [grpcclient.InferRequestedOutput("embedding")]
    result  = client.infer(model_name, inputs, outputs=outputs)
    return result.as_numpy("embedding")


# ---------------------------------------------------------------------------
# Generic benchmark helpers
# ---------------------------------------------------------------------------

def _timed(fn: Callable) -> Optional[float]:
    try:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"    [warn] {e}")
        return None


def bench_serial(fn: Callable, iterations: int, warmup: int = WARMUP_ITERS) -> Dict:
    """Serial latency benchmark. Returns p50/p95/p99/mean."""
    for _ in range(warmup):
        fn()
    lats = [lat for _ in range(iterations) if (lat := _timed(fn)) is not None]
    if not lats:
        return {"error": "all requests failed"}
    arr = np.array(lats)
    return {
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "min_ms":  float(arr.min()),
        "max_ms":  float(arr.max()),
        "n":       len(lats),
    }


def bench_concurrent(fn: Callable, concurrency: int, total: int,
                     batch_size: int, warmup: int = WARMUP_ITERS) -> Dict:
    """Concurrent throughput benchmark."""
    for _ in range(warmup):
        fn()
    lats = []
    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_timed, fn) for _ in range(total)]
        for f in as_completed(futures):
            v = f.result()
            if v is not None:
                lats.append(v)
    wall_s = time.perf_counter() - wall_t0
    arr = np.array(lats) if lats else np.array([0.0])
    return {
        "concurrency":  concurrency,
        "p50_ms":       float(np.percentile(arr, 50)),
        "p95_ms":       float(np.percentile(arr, 95)),
        "mean_ms":      float(arr.mean()),
        "imgs_per_sec": float(len(lats) * batch_size / wall_s),
        "n":            len(lats),
    }


# ---------------------------------------------------------------------------
# Per-backend benchmark runners
# ---------------------------------------------------------------------------

def run_pytorch(pytorch_url: str, iterations: int) -> Dict:
    print(f"\n{'─'*60}")
    print("  1. PyTorch FastAPI  (HTTP/1.1 + base64 JPEG ~10KB/img)")
    print(f"{'─'*60}")
    results = {"serial": {}, "concurrent": {}}
    for bs in BATCH_SIZES:
        print(f"  batch={bs} serial ...", end=" ", flush=True)
        imgs = [make_jpeg_b64() for _ in range(bs)]
        fn   = lambda i=imgs: _pytorch_infer(pytorch_url, i)
        r    = bench_serial(fn, iterations)
        r["imgs_per_sec"]         = bs / (r.get("p50_ms", 1) / 1000)
        r["payload_approx_kb"]    = bs * 10
        results["serial"][f"batch_{bs}"] = r
        print(f"p50={r.get('p50_ms', 0):.1f}ms  {r.get('imgs_per_sec', 0):.1f}img/s")

    print("  concurrent (batch=1) ...")
    imgs1 = [make_jpeg_b64()]
    for conc in CONCURRENCY_LEVELS:
        fn = lambda i=imgs1: _pytorch_infer(pytorch_url, i)
        results["concurrent"][f"conc_{conc}"] = bench_concurrent(fn, conc, 60, 1)
    return results


def run_triton_http(http_url: str, metrics_url: Optional[str],
                    model_name: str, label: str, iterations: int) -> Dict:
    print(f"\n{'─'*60}")
    print(f"  {label}  (HTTP/1.1 + binary FP32 ~602KB/img)")
    print(f"{'─'*60}")
    host_port = http_url.replace("http://", "").replace("https://", "").rstrip("/")
    results = {"serial": {}, "concurrent": {}}
    for bs in BATCH_SIZES:
        print(f"  batch={bs} serial ...", end=" ", flush=True)
        t  = make_tensor_batch(bs)
        m0 = parse_triton_metrics(metrics_url, model_name) if metrics_url else {}
        fn = lambda tensor=t: _triton_http_infer(host_port, model_name, tensor)
        r  = bench_serial(fn, iterations)
        m1 = parse_triton_metrics(metrics_url, model_name) if metrics_url else {}
        r["imgs_per_sec"]   = bs / (r.get("p50_ms", 1) / 1000)
        r["payload_kb"]     = (bs * 3 * 224 * 224 * 4) / 1024
        r["server_metrics"] = compute_server_metrics(m0, m1) or {}
        results["serial"][f"batch_{bs}"] = r
        gpu = r["server_metrics"].get("gpu_compute_ms")
        print(f"p50={r.get('p50_ms', 0):.1f}ms  {r.get('imgs_per_sec', 0):.1f}img/s"
              + (f"  GPU={gpu:.1f}ms" if gpu else ""))

    print("  concurrent (batch=1) ...")
    t1 = make_tensor_batch(1)
    for conc in CONCURRENCY_LEVELS:
        fn = lambda tensor=t1: _triton_http_infer(host_port, model_name, tensor)
        results["concurrent"][f"conc_{conc}"] = bench_concurrent(fn, conc, 60, 1)
    return results


def run_triton_grpc(grpc_url: str, metrics_url: Optional[str],
                    model_name: str, label: str, iterations: int) -> Dict:
    print(f"\n{'─'*60}")
    print(f"  {label}  (gRPC HTTP/2 + binary FP32 ~602KB/img)")
    print(f"{'─'*60}")
    results = {"serial": {}, "concurrent": {}}
    for bs in BATCH_SIZES:
        print(f"  batch={bs} serial ...", end=" ", flush=True)
        t  = make_tensor_batch(bs)
        m0 = parse_triton_metrics(metrics_url, model_name) if metrics_url else {}
        fn = lambda tensor=t: _triton_grpc_infer(grpc_url, model_name, tensor)
        r  = bench_serial(fn, iterations)
        m1 = parse_triton_metrics(metrics_url, model_name) if metrics_url else {}
        r["imgs_per_sec"]   = bs / (r.get("p50_ms", 1) / 1000)
        r["payload_kb"]     = (bs * 3 * 224 * 224 * 4) / 1024
        r["server_metrics"] = compute_server_metrics(m0, m1) or {}
        results["serial"][f"batch_{bs}"] = r
        gpu = r["server_metrics"].get("gpu_compute_ms")
        print(f"p50={r.get('p50_ms', 0):.1f}ms  {r.get('imgs_per_sec', 0):.1f}img/s"
              + (f"  GPU={gpu:.1f}ms" if gpu else ""))

    print("  concurrent (batch=1) ...")
    t1 = make_tensor_batch(1)
    for conc in CONCURRENCY_LEVELS:
        fn = lambda tensor=t1: _triton_grpc_infer(grpc_url, model_name, tensor)
        results["concurrent"][f"conc_{conc}"] = bench_concurrent(fn, conc, 60, 1)
    return results


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

BACKEND_ORDER = [
    ("pytorch",   "PyTorch HTTP      "),
    ("onnx_http", "Triton ONNX HTTP  "),
    ("onnx_grpc", "Triton ONNX gRPC  "),
    ("trt_http",  "Triton TRT  HTTP  "),
    ("trt_grpc",  "Triton TRT  gRPC  "),
]


def print_summary(results: Dict):
    backends = results.get("backends", {})

    print("\n" + "=" * 80)
    print("5-WAY COMPARISON — Serial p50 latency (ms)")
    print("=" * 80)
    hdr = f"{'Backend':<24}" + "".join(f"  b={bs:>2} p50" for bs in BATCH_SIZES)
    print(hdr)
    print("-" * (24 + len(BATCH_SIZES) * 12))
    for key, label in BACKEND_ORDER:
        if key not in backends:
            continue
        row = f"{label:<24}"
        for bs in BATCH_SIZES:
            p50 = backends[key]["serial"].get(f"batch_{bs}", {}).get("p50_ms")
            row += f"  {p50:>7.1f}ms" if p50 else f"  {'—':>9}"
        print(row)

    print()
    print("5-WAY COMPARISON — Serial throughput (img/s)")
    print("-" * (24 + len(BATCH_SIZES) * 12))
    for key, label in BACKEND_ORDER:
        if key not in backends:
            continue
        row = f"{label:<24}"
        for bs in BATCH_SIZES:
            ips = backends[key]["serial"].get(f"batch_{bs}", {}).get("imgs_per_sec")
            row += f"  {ips:>7.1f}  " if ips else f"  {'—':>9}"
        print(row)

    # gRPC vs HTTP speedup within each backend model
    print()
    print("gRPC speedup vs HTTP same backend (p50; >1.0× means gRPC is faster)")
    print("-" * 60)
    for pair_http, pair_grpc, pair_label in [
        ("onnx_http", "onnx_grpc", "ONNX gRPC/HTTP"),
        ("trt_http",  "trt_grpc",  "TRT  gRPC/HTTP"),
    ]:
        if pair_http not in backends or pair_grpc not in backends:
            continue
        row = f"  {pair_label:<16}"
        for bs in BATCH_SIZES:
            h = backends[pair_http]["serial"].get(f"batch_{bs}", {}).get("p50_ms")
            g = backends[pair_grpc]["serial"].get(f"batch_{bs}", {}).get("p50_ms")
            row += f"  {h/g:>5.2f}×" if (h and g and g > 0) else f"  {'—':>5} "
        print(row)

    # Server-side GPU compute (should be equal for HTTP and gRPC of same backend)
    print()
    print("Server-side GPU compute (Triton metrics) — should be equal for HTTP/gRPC pairs:")
    for key, label in BACKEND_ORDER:
        if key == "pytorch" or key not in backends:
            continue
        for bs in BATCH_SIZES:
            gpu = backends[key]["serial"].get(f"batch_{bs}", {}).get("server_metrics", {}).get("gpu_compute_ms")
            if gpu:
                print(f"  {label.strip():<22} batch={bs}  GPU={gpu:.2f}ms")
                break

    # Concurrent throughput summary
    print()
    print("Concurrent throughput (batch=1, img/s):")
    print(f"  {'Backend':<24}" + "".join(f"  conc={c:>2}" for c in CONCURRENCY_LEVELS))
    print("  " + "-" * (22 + len(CONCURRENCY_LEVELS) * 10))
    for key, label in BACKEND_ORDER:
        if key not in backends:
            continue
        row = f"  {label:<24}"
        for conc in CONCURRENCY_LEVELS:
            ips = backends[key].get("concurrent", {}).get(f"conc_{conc}", {}).get("imgs_per_sec")
            row += f"  {ips:>7.1f}" if ips else f"  {'—':>7}"
        print(row)


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_triton_http(url: str, label: str) -> bool:
    try:
        r  = requests.get(f"{url}/v2/health/ready", timeout=10)
        ok = r.status_code == 200
        print(f"  {'✓' if ok else '✗'} {label} HTTP: {url}  ({r.status_code})")
        return ok
    except Exception as e:
        print(f"  ✗ {label} HTTP unreachable: {e}")
        return False


def check_triton_grpc(url: str, label: str) -> bool:
    try:
        c  = grpcclient.InferenceServerClient(url=url, verbose=False)
        ok = c.is_server_ready()
        print(f"  {'✓' if ok else '✗'} {label} gRPC: {url}")
        return ok
    except Exception as e:
        print(f"  ✗ {label} gRPC unreachable: {e}")
        return False


def check_pytorch(url: str) -> bool:
    try:
        r  = requests.get(f"{url}/health", timeout=10)
        ok = r.status_code == 200
        print(f"  {'✓' if ok else '✗'} PyTorch HTTP: {url}  ({r.status_code})")
        return ok
    except Exception as e:
        print(f"  ✗ PyTorch unreachable: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 7: 5-way protocol comparison benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pytorch-url",      default=DEFAULT_PYTORCH_URL,
                        help="PyTorch FastAPI HTTP URL")
    parser.add_argument("--onnx-http-url",    default=DEFAULT_ONNX_HTTP_URL,
                        help="Triton ONNX HTTP URL")
    parser.add_argument("--onnx-grpc-url",    default=DEFAULT_ONNX_GRPC_URL,
                        help="Triton ONNX gRPC endpoint (host:port, no scheme)")
    parser.add_argument("--onnx-metrics-url", default=DEFAULT_ONNX_METRICS_URL,
                        help="Triton ONNX Prometheus metrics URL")
    parser.add_argument("--trt-http-url",     default=DEFAULT_TRT_HTTP_URL,
                        help="Triton TRT HTTP URL")
    parser.add_argument("--trt-grpc-url",     default=DEFAULT_TRT_GRPC_URL,
                        help="Triton TRT gRPC endpoint (host:port, no scheme)")
    parser.add_argument("--trt-metrics-url",  default=DEFAULT_TRT_METRICS_URL,
                        help="Triton TRT Prometheus metrics URL")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Serial iterations per batch size per backend")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch backend")
    parser.add_argument("--skip-onnx",    action="store_true", help="Skip Triton ONNX backends")
    parser.add_argument("--skip-trt",     action="store_true", help="Skip Triton TRT backends")
    args = parser.parse_args()

    print("=" * 70)
    print("Step 7: 5-Way Protocol Comparison")
    print("=" * 70)
    print(f"  PyTorch      : {args.pytorch_url}")
    print(f"  ONNX HTTP    : {args.onnx_http_url}   gRPC: {args.onnx_grpc_url}")
    print(f"  TRT  HTTP    : {args.trt_http_url}   gRPC: {args.trt_grpc_url}")
    print(f"  Iterations   : {args.iterations} per batch size per backend")
    print(f"  Batch sizes  : {BATCH_SIZES}")
    print("=" * 70)

    # Health checks
    print("\nHealth checks:")
    if not args.skip_pytorch and not check_pytorch(args.pytorch_url):
        print("  WARNING: PyTorch unhealthy — skipping.")
        args.skip_pytorch = True
    if not args.skip_onnx:
        if not check_triton_http(args.onnx_http_url, "ONNX") and \
           not check_triton_grpc(args.onnx_grpc_url, "ONNX"):
            print("  WARNING: ONNX Triton unreachable — skipping.")
            args.skip_onnx = True
    if not args.skip_trt:
        if not check_triton_http(args.trt_http_url, "TRT") and \
           not check_triton_grpc(args.trt_grpc_url, "TRT"):
            print("  WARNING: TRT Triton unreachable — skipping.")
            args.skip_trt = True

    results = {
        "timestamp": datetime.now().isoformat(),
        "config":    vars(args),
        "backends":  {},
    }

    if not args.skip_pytorch:
        results["backends"]["pytorch"] = run_pytorch(args.pytorch_url, args.iterations)

    if not args.skip_onnx:
        results["backends"]["onnx_http"] = run_triton_http(
            args.onnx_http_url, args.onnx_metrics_url,
            ONNX_MODEL_NAME, "2. Triton ONNX HTTP", args.iterations,
        )
        results["backends"]["onnx_grpc"] = run_triton_grpc(
            args.onnx_grpc_url, args.onnx_metrics_url,
            ONNX_MODEL_NAME, "3. Triton ONNX gRPC", args.iterations,
        )

    if not args.skip_trt:
        results["backends"]["trt_http"] = run_triton_http(
            args.trt_http_url, args.trt_metrics_url,
            TRT_MODEL_NAME, "4. Triton TRT  HTTP", args.iterations,
        )
        results["backends"]["trt_grpc"] = run_triton_grpc(
            args.trt_grpc_url, args.trt_metrics_url,
            TRT_MODEL_NAME, "5. Triton TRT  gRPC", args.iterations,
        )

    print_summary(results)

    out_dir = PROJECT_ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"step7_5way_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
