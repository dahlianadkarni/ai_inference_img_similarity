#!/usr/bin/env python3
"""
macOS Local Inference Benchmark — in-process, no HTTP/gRPC overhead.

Benchmarks all inference backends available on macOS Apple Silicon directly
in-process (no HTTP, no serialization). This isolates pure compute time and
lets you compare against the server-side GPU compute metrics from remote runs.

Backends tested:
  1. PyTorch CPU         — open_clip on CPU
  2. PyTorch MPS         — open_clip on Apple Silicon GPU (Metal/MPS)
  3. ONNX Runtime CPU    — ORT CPUExecutionProvider
  4. ONNX Runtime CoreML — ORT CoreMLExecutionProvider (ANE/GPU via CoreML)

Note: Triton Inference Server is Linux-only; not available on macOS.

What this measures vs remote benchmarks
---------------------------------------
Remote benchmarks (step6a, step7) measure CLIENT-SIDE latency, which includes:
  - Network round-trip + payload transfer (10KB JPEG or 602KB tensor)
  - Server decode/preprocess
  - GPU compute
  - Response serialization

This script measures IN-PROCESS time only:
  - Preprocess:   image decode + model preprocessing (resize, normalize, tensor)
  - Inference:    model forward pass only
  - Total:        preprocess + inference

The "inference only" metric is the closest equivalent to the server-side
GPU compute metrics reported in step6a (4.4ms ONNX, 2.0ms TRT on A100).

GPU comparison context (A100 SXM4, server-side compute):
  PyTorch       ~10–15ms  (estimated; client accepts JPEG so no direct measure)
  Triton ONNX    4.4ms    GPU compute
  Triton TRT     2.0ms    GPU compute (best; consumer RTX 4080 = 2.0ms too)

Usage:
  # Quick run (10 iterations, smaller batches)
  python scripts/benchmark_macos_local.py --iterations 10

  # Full run
  python scripts/benchmark_macos_local.py --iterations 50

  # Skip slow backends
  python scripts/benchmark_macos_local.py --skip-onnx-coreml

  # Custom batch sizes
  python scripts/benchmark_macos_local.py --batch-sizes 1 8 32

Output:
  benchmark_results_macOS_local/macos_local_<timestamp>.json
  Printed comparison tables to stdout

Requirements (already in requirements.txt):
  torch, open_clip_torch, onnxruntime, pillow, numpy
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = REPO_ROOT / "model_repository" / "openclip_vit_b32" / "1" / "model.onnx"
RESULTS_DIR = REPO_ROOT / "benchmark_results_macOS_local"

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32]
DEFAULT_ITERATIONS = 30
WARMUP_ITERATIONS = 5
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
IMAGE_SIZE = 224


# ─── Utility ──────────────────────────────────────────────────────────────────

def make_random_pil_images(n: int, size: int = IMAGE_SIZE) -> List[Image.Image]:
    """Create n random RGB PIL images of the given size."""
    return [
        Image.fromarray(
            np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        )
        for _ in range(n)
    ]


def stats(times_ms: List[float]) -> Dict[str, float]:
    """Compute latency statistics from a list of per-call milliseconds."""
    arr = np.array(times_ms)
    return {
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "min_ms":  float(arr.min()),
        "max_ms":  float(arr.max()),
    }


# ─── PyTorch backends ─────────────────────────────────────────────────────────

def run_pytorch_benchmark(
    device_str: str,
    batch_sizes: List[int],
    iterations: int,
) -> Optional[Dict]:
    """Benchmark open_clip on the given torch device ('cpu' or 'mps')."""
    try:
        import torch
        import open_clip
    except ImportError as e:
        print(f"  [SKIP] PyTorch {device_str}: {e}")
        return None

    if device_str == "mps" and not torch.backends.mps.is_available():
        print(f"  [SKIP] MPS not available on this machine")
        return None

    device = torch.device(device_str)
    backend_id = f"pytorch_{device_str}"

    print(f"\n{'─'*60}")
    print(f"  Backend: {backend_id.upper()}")
    print(f"  Loading model '{MODEL_NAME}' ({PRETRAINED}) on {device_str}...")

    t0 = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f}ms")

    results_by_batch: Dict[int, Dict] = {}

    for batch_size in batch_sizes:
        pil_images = make_random_pil_images(batch_size)

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            with torch.no_grad():
                model.encode_image(tensors)
        if device_str == "mps":
            torch.mps.synchronize()

        preprocess_times: List[float] = []
        inference_times: List[float] = []

        for _ in range(iterations):
            # Preprocess timing
            t_pre = time.perf_counter()
            tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            pre_ms = (time.perf_counter() - t_pre) * 1000
            preprocess_times.append(pre_ms)

            # Inference timing
            t_inf = time.perf_counter()
            with torch.no_grad():
                emb = model.encode_image(tensors)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            if device_str == "mps":
                torch.mps.synchronize()
            inf_ms = (time.perf_counter() - t_inf) * 1000
            inference_times.append(inf_ms)

        total_times = [p + i for p, i in zip(preprocess_times, inference_times)]
        imgs_per_sec = batch_size / (np.mean(total_times) / 1000)
        inf_imgs_per_sec = batch_size / (np.mean(inference_times) / 1000)

        results_by_batch[batch_size] = {
            "preprocess": stats(preprocess_times),
            "inference":  stats(inference_times),
            "total":      stats(total_times),
            "imgs_per_sec_total":     round(imgs_per_sec, 1),
            "imgs_per_sec_inference": round(inf_imgs_per_sec, 1),
        }

        print(
            f"  batch={batch_size:2d}  "
            f"preprocess={np.mean(preprocess_times):.1f}ms  "
            f"inference={np.mean(inference_times):.1f}ms  "
            f"total={np.mean(total_times):.1f}ms  "
            f"({imgs_per_sec:.1f} img/s)"
        )

    # Free memory
    del model
    if device_str == "mps":
        import torch
        torch.mps.empty_cache()

    return {
        "backend": backend_id,
        "device": device_str,
        "framework": "pytorch",
        "model": f"{MODEL_NAME}/{PRETRAINED}",
        "model_load_ms": round(load_ms, 1),
        "iterations": iterations,
        "warmup_iterations": WARMUP_ITERATIONS,
        "batch_results": {str(bs): v for bs, v in results_by_batch.items()},
    }


# ─── ONNX Runtime backends ────────────────────────────────────────────────────

def run_onnx_benchmark(
    provider: str,
    batch_sizes: List[int],
    iterations: int,
) -> Optional[Dict]:
    """
    Benchmark ONNX Runtime with the given execution provider.

    provider: 'CPUExecutionProvider' or 'CoreMLExecutionProvider'

    The ONNX model expects a pre-processed float32 tensor [B, 3, 224, 224],
    the same format Triton receives from the benchmark client. This directly
    mirrors the server-side GPU compute measured in remote step6a/step7 runs.
    """
    try:
        import onnxruntime as ort
        import open_clip
        import torch
    except ImportError as e:
        print(f"  [SKIP] ONNX {provider}: {e}")
        return None

    if not ONNX_MODEL_PATH.exists():
        print(f"  [SKIP] ONNX model not found at: {ONNX_MODEL_PATH}")
        return None

    available = ort.get_available_providers()
    if provider not in available:
        print(f"  [SKIP] {provider} not available. Available: {available}")
        return None

    short_name = "coreml" if "CoreML" in provider else "cpu"
    backend_id = f"onnx_{short_name}"

    print(f"\n{'─'*60}")
    print(f"  Backend: {backend_id.upper()}")
    print(f"  Providers: [{provider}]")
    if "CoreML" in provider:
        print(f"  Note: CoreML EP may only cover ~50% of ViT-B-32 graph nodes.")
        print(f"        Unsupported ops (e.g. attention slices with dim=0) fall back to CPU,")
        print(f"        which can make CoreML slower than pure CPU. For best CoreML performance,")
        print(f"        export a native .mlpackage via coremltools directly.")
    print(f"  Loading ONNX model from {ONNX_MODEL_PATH.name}...")

    t0 = time.perf_counter()
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=sess_options,
        providers=[provider],
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f}ms")

    input_name = session.get_inputs()[0].name   # "image"
    output_name = session.get_outputs()[0].name  # "embedding"

    # Use open_clip's preprocess pipeline so pixel normalization matches PyTorch
    _, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )

    results_by_batch: Dict[int, Dict] = {}

    for batch_size in batch_sizes:
        pil_images = make_random_pil_images(batch_size)

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            tensors = torch.stack([preprocess(img) for img in pil_images]).numpy()
            session.run([output_name], {input_name: tensors})

        preprocess_times: List[float] = []
        inference_times: List[float] = []

        for _ in range(iterations):
            # Preprocess timing (PIL → float32 numpy, same format as Triton client)
            t_pre = time.perf_counter()
            tensors = torch.stack([preprocess(img) for img in pil_images]).numpy()
            pre_ms = (time.perf_counter() - t_pre) * 1000
            preprocess_times.append(pre_ms)

            # Inference timing (pure ORT session.run, no HTTP)
            t_inf = time.perf_counter()
            session.run([output_name], {input_name: tensors})
            inf_ms = (time.perf_counter() - t_inf) * 1000
            inference_times.append(inf_ms)

        total_times = [p + i for p, i in zip(preprocess_times, inference_times)]
        imgs_per_sec = batch_size / (np.mean(total_times) / 1000)
        inf_imgs_per_sec = batch_size / (np.mean(inference_times) / 1000)

        results_by_batch[batch_size] = {
            "preprocess": stats(preprocess_times),
            "inference":  stats(inference_times),
            "total":      stats(total_times),
            "imgs_per_sec_total":     round(imgs_per_sec, 1),
            "imgs_per_sec_inference": round(inf_imgs_per_sec, 1),
        }

        print(
            f"  batch={batch_size:2d}  "
            f"preprocess={np.mean(preprocess_times):.1f}ms  "
            f"inference={np.mean(inference_times):.1f}ms  "
            f"total={np.mean(total_times):.1f}ms  "
            f"({imgs_per_sec:.1f} img/s)"
        )

    del session

    return {
        "backend": backend_id,
        "device": short_name,
        "framework": "onnxruntime",
        "provider": provider,
        "model": str(ONNX_MODEL_PATH.relative_to(REPO_ROOT)),
        "model_load_ms": round(load_ms, 1),
        "iterations": iterations,
        "warmup_iterations": WARMUP_ITERATIONS,
        "batch_results": {str(bs): v for bs, v in results_by_batch.items()},
    }


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(all_results: List[Dict], batch_sizes: List[int]) -> None:
    """Print a comparison table across all backends for each batch size."""

    print(f"\n{'═'*92}")
    print("  INFERENCE-ONLY LATENCY  (ms, p50)  — closest to server-side GPU compute in remote benchmarks")
    print(f"{'═'*92}")
    header = f"  {'Backend':<22}" + "".join(f"  batch={bs:<5}" for bs in batch_sizes)
    print(header)
    print(f"  {'─'*22}" + "".join(f"  {'─'*9}" for _ in batch_sizes))
    for r in all_results:
        row = f"  {r['backend']:<22}"
        for bs in batch_sizes:
            val = r["batch_results"].get(str(bs), {}).get("inference", {}).get("p50_ms")
            row += f"  {f'{val:.1f}ms':<9}" if val else f"  {'N/A':<9}"
        print(row)

    print(f"\n{'═'*92}")
    print("  TOTAL THROUGHPUT  (img/s, inference only)  — preprocess excluded")
    print(f"{'═'*92}")
    print(header)
    print(f"  {'─'*22}" + "".join(f"  {'─'*9}" for _ in batch_sizes))
    for r in all_results:
        row = f"  {r['backend']:<22}"
        for bs in batch_sizes:
            val = r["batch_results"].get(str(bs), {}).get("imgs_per_sec_inference")
            row += f"  {f'{val:.1f}':<9}" if val else f"  {'N/A':<9}"
        print(row)

    print(f"\n{'═'*92}")
    print("  TOTAL PIPELINE  (img/s, preprocess + inference)")
    print(f"{'═'*92}")
    print(header)
    print(f"  {'─'*22}" + "".join(f"  {'─'*9}" for _ in batch_sizes))
    for r in all_results:
        row = f"  {r['backend']:<22}"
        for bs in batch_sizes:
            val = r["batch_results"].get(str(bs), {}).get("imgs_per_sec_total")
            row += f"  {f'{val:.1f}':<9}" if val else f"  {'N/A':<9}"
        print(row)

    # Compare to remote GPU context
    print(f"\n{'═'*92}")
    print("  REMOTE GPU CONTEXT (server-side compute only, from step6a A100 SXM4 benchmark):")
    print("    PyTorch FastAPI   ~10–15ms  (estimated GPU compute; client accepts JPEG)")
    print("    Triton ONNX CUDA   4.4ms   (measured server-side GPU compute)")
    print("    Triton TRT EP      2.0ms   (measured; consumer RTX 4080 also 2.0ms)")
    print(f"{'═'*92}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark OpenCLIP inference backends locally on macOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                   help=f"Number of benchmark iterations per batch size (default: {DEFAULT_ITERATIONS})")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
                   metavar="N", help=f"Batch sizes to test (default: {DEFAULT_BATCH_SIZES})")
    p.add_argument("--skip-pytorch-cpu",  action="store_true", help="Skip PyTorch CPU backend")
    p.add_argument("--skip-pytorch-mps",  action="store_true", help="Skip PyTorch MPS backend")
    p.add_argument("--skip-onnx-cpu",     action="store_true", help="Skip ONNX Runtime CPU backend")
    p.add_argument("--skip-onnx-coreml",  action="store_true", help="Skip ONNX Runtime CoreML backend")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  macOS Local Inference Benchmark")
    print(f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Platform:   {platform.platform()}")
    print(f"  Machine:    {platform.machine()}")
    print(f"  Model:      {MODEL_NAME} / {PRETRAINED}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Iterations: {args.iterations} (+{WARMUP_ITERATIONS} warmup)")
    print(f"  ONNX model: {ONNX_MODEL_PATH.relative_to(REPO_ROOT)}")
    print("=" * 60)

    all_results: List[Dict] = []

    if not args.skip_pytorch_cpu:
        r = run_pytorch_benchmark("cpu", args.batch_sizes, args.iterations)
        if r:
            all_results.append(r)

    if not args.skip_pytorch_mps:
        r = run_pytorch_benchmark("mps", args.batch_sizes, args.iterations)
        if r:
            all_results.append(r)

    if not args.skip_onnx_cpu:
        r = run_onnx_benchmark("CPUExecutionProvider", args.batch_sizes, args.iterations)
        if r:
            all_results.append(r)

    if not args.skip_onnx_coreml:
        r = run_onnx_benchmark("CoreMLExecutionProvider", args.batch_sizes, args.iterations)
        if r:
            all_results.append(r)

    if not all_results:
        print("No backends ran — check flags or dependencies.")
        sys.exit(1)

    # Summary table
    print_summary_table(all_results, args.batch_sizes)

    # Save JSON
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"macos_local_{timestamp}.json"

    output = {
        "metadata": {
            "script": "scripts/benchmark_macos_local.py",
            "timestamp": timestamp,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "onnx_model_path": str(ONNX_MODEL_PATH.relative_to(REPO_ROOT)),
            "iterations": args.iterations,
            "warmup_iterations": WARMUP_ITERATIONS,
            "batch_sizes": args.batch_sizes,
            "note": (
                "In-process benchmark, no HTTP/serialization overhead. "
                "'inference' timing = model forward pass only, directly comparable "
                "to server-side GPU compute metrics from remote step6a/step7 runs. "
                "CoreML EP caveat: the ViT-B-32 ONNX model has ~52% node coverage "
                "(636/1222 nodes) — unsupported ops fall back to CPU, making "
                "onnx_coreml slower than onnx_cpu here. A native .mlpackage via "
                "coremltools would give true CoreML/ANE performance."
            ),
            "remote_gpu_context": {
                "source": "step6a A100 SXM4 80GB (benchmark_results/step6a_a100_remote.json)",
                "pytorch_fastapi_gpu_compute_ms": "~10-15 (estimated)",
                "triton_onnx_cuda_ep_ms": 4.4,
                "triton_trt_ep_ms": 2.0,
            },
        },
        "results": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
