#!/usr/bin/env python3
"""
Multi-GPU Scaling Benchmark for Triton Inference Server (Step 6B)

This script tests Triton's multi-GPU scaling by running benchmarks at various
concurrency levels to find the saturation point and measure throughput scaling.

Key Measurements:
  1. Throughput (images/sec) vs concurrency
  2. Latency (p50, p95, p99) vs concurrency
  3. Server-side GPU compute time (from Triton metrics)
  4. Server-side queue time (dynamic batching impact)
  5. Per-GPU utilization and memory usage

Usage:
  # Single test run (2x GPU instance)
  python scripts/benchmark_multigpu.py \
      --triton-url http://207.180.148.74:8000 \
      --config-name "2-gpu-nvlink" \
      --concurrency 1,2,4,8,16,32,64 \
      --iterations 100 \
      --output benchmark_results/step6b_2gpu.json

  # Full sweep (run this 4 times for 1x, 2x, 4x, 8x)
  for gpus in 1 2 4 8; do
    python scripts/benchmark_multigpu.py \
      --triton-url http://INSTANCE_IP:8000 \
      --config-name "${gpus}-gpu-config" \
      --concurrency 1,2,4,8,16,32,64,128 \
      --iterations 100 \
      --output benchmark_results/step6b_${gpus}gpu.json
  done

Requirements:
  - Triton server running with multi-GPU config
  - tritonclient[all] installed (pip install tritonclient[all])
  - requests, numpy, pillow
"""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
import tritonclient.http as httpclient

# ---------------------------------------------------------------------------
# Test Image Generation
# ---------------------------------------------------------------------------

def create_test_images(count: int, size: int = 224) -> List[np.ndarray]:
    """Create synthetic test images (deterministic)."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(count)]


def image_to_triton_tensor(img_array: np.ndarray) -> np.ndarray:
    """Preprocess image for Triton ONNX backend (CHW float32, [0,1])."""
    t = img_array.astype(np.float32) / 255.0
    return np.transpose(t, (2, 0, 1))  # HWC -> CHW


# ---------------------------------------------------------------------------
# Statistics Helpers
# ---------------------------------------------------------------------------

def percentile_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute latency statistics in milliseconds."""
    if not latencies:
        return {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "min_ms": 0, "max_ms": 0, "count": 0}
    
    a = np.array(latencies)
    return {
        "mean_ms": round(float(np.mean(a) * 1000), 3),
        "p50_ms": round(float(np.percentile(a, 50) * 1000), 3),
        "p95_ms": round(float(np.percentile(a, 95) * 1000), 3),
        "p99_ms": round(float(np.percentile(a, 99) * 1000), 3),
        "min_ms": round(float(np.min(a) * 1000), 3),
        "max_ms": round(float(np.max(a) * 1000), 3),
        "count": len(latencies),
    }


# ---------------------------------------------------------------------------
# Triton Inference
# ---------------------------------------------------------------------------

def triton_infer_single(host: str, port: str, img_tensor: np.ndarray, timeout: int = 30) -> Tuple[float, bool]:
    """
    Send single-image inference request to Triton.
    
    Returns:
        (latency_seconds, success)
    """
    try:
        t0 = time.perf_counter()
        
        client = httpclient.InferenceServerClient(url=f"{host}:{port}", connection_timeout=timeout)
        batch = np.expand_dims(img_tensor, 0)  # Add batch dimension
        
        inputs = [httpclient.InferInput("image", batch.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch)
        
        outputs = [httpclient.InferRequestedOutput("embedding")]
        
        result = client.infer("openclip_vit_b32", inputs, outputs=outputs)
        
        elapsed = time.perf_counter() - t0
        
        # Verify we got valid embedding
        embedding = result.as_numpy("embedding")
        if embedding.shape != (1, 512):
            return elapsed, False
            
        return elapsed, True
        
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"Inference error: {e}")
        return elapsed, False


# ---------------------------------------------------------------------------
# Concurrent Benchmark
# ---------------------------------------------------------------------------

def benchmark_concurrent(
    host: str,
    port: str,
    total_requests: int,
    concurrency: int,
    img_tensor: np.ndarray,
    warmup: int = 3
) -> Dict:
    """
    Run concurrent inference benchmark.
    
    Args:
        host: Triton server host
        port: Triton server port
        total_requests: Total number of requests to send
        concurrency: Number of concurrent workers
        img_tensor: Preprocessed image tensor (CHW format)
        warmup: Number of warmup requests
        
    Returns:
        Dictionary with latency stats and throughput
    """
    print(f"  Running {total_requests} requests with concurrency {concurrency}...")
    
    # Warmup
    for _ in range(warmup):
        triton_infer_single(host, port, img_tensor, timeout=60)
    
    latencies = []
    errors = 0
    start_wall = time.perf_counter()
    
    def _call():
        return triton_infer_single(host, port, img_tensor, timeout=60)
    
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_call) for _ in range(total_requests)]
        for f in as_completed(futures):
            elapsed, success = f.result()
            if success:
                latencies.append(elapsed)
            else:
                errors += 1
    
    wall_time = time.perf_counter() - start_wall
    
    stats = percentile_stats(latencies)
    stats["wall_time_s"] = round(wall_time, 3)
    stats["throughput_imgs_per_sec"] = round(len(latencies) / wall_time, 2) if wall_time > 0 else 0
    stats["concurrency"] = concurrency
    stats["total_requests"] = total_requests
    stats["successful"] = len(latencies)
    stats["errors"] = errors
    
    print(f"    → Throughput: {stats['throughput_imgs_per_sec']:.1f} img/s, "
          f"Latency p50: {stats['p50_ms']:.1f}ms, p95: {stats['p95_ms']:.1f}ms")
    
    return stats


# ---------------------------------------------------------------------------
# Triton Metrics Collection
# ---------------------------------------------------------------------------

def fetch_triton_metrics(metrics_url: str) -> Optional[Dict]:
    """
    Fetch and parse Triton Prometheus metrics.
    
    Returns dict with:
      - inference_count: Total successful inferences
      - queue_time_us: Average queue time in microseconds
      - compute_time_us: Average compute time in microseconds
      - per_gpu_utilization: List of utilization per GPU (if available)
    """
    try:
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        
        metrics = {}
        lines = response.text.split('\n')
        
        # Parse key metrics
        for line in lines:
            if line.startswith('#'):
                continue
                
            if 'nv_inference_request_success{model="openclip_vit_b32"' in line:
                metrics['inference_count'] = int(line.split()[-1])
            
            elif 'nv_inference_queue_duration_us{model="openclip_vit_b32"' in line:
                metrics['queue_time_us'] = float(line.split()[-1])
            
            elif 'nv_inference_compute_infer_duration_us{model="openclip_vit_b32"' in line:
                metrics['compute_time_us'] = float(line.split()[-1])
            
            elif 'nv_gpu_utilization{gpu=' in line:
                if 'per_gpu_utilization' not in metrics:
                    metrics['per_gpu_utilization'] = []
                gpu_util = float(line.split()[-1])
                metrics['per_gpu_utilization'].append(gpu_util)
        
        return metrics if metrics else None
        
    except Exception as e:
        print(f"Warning: Could not fetch Triton metrics: {e}")
        return None


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Triton Scaling Benchmark")
    parser.add_argument("--triton-url", default="http://localhost:8000",
                        help="Triton inference URL (e.g., http://207.180.148.74:8000)")
    parser.add_argument("--metrics-url", default=None,
                        help="Triton metrics URL (default: infer from triton-url)")
    parser.add_argument("--config-name", required=True,
                        help="Config name for results (e.g., '4-gpu-nvlink')")
    parser.add_argument("--gpu-name", default=None,
                        help="GPU name for results file (e.g., 'a100', 'rtx4080'). If provided, will be included in output filename.")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32",
                        help="Comma-separated concurrency levels to test")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations per concurrency level")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (default: benchmark_results/step6b_{gpu}_{config}_{timestamp}.json)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup requests per concurrency level")
    
    args = parser.parse_args()
    
    # Parse triton URL
    triton_url = args.triton_url.replace('http://', '').replace('https://', '')
    if ':' in triton_url:
        host, port = triton_url.split(':')
    else:
        host = triton_url
        port = '8000'
    
    # Determine metrics URL
    if args.metrics_url:
        metrics_url = args.metrics_url
    else:
        # Assume metrics on port 8002
        metrics_url = f"http://{host}:8002/metrics"
    
    # Parse concurrency levels
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(',')]
    
    # Generate output filename if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        gpu_part = f"{args.gpu_name.lower()}_" if args.gpu_name else ""
        output_path = output_dir / f"step6b_{gpu_part}{args.config_name}_{timestamp}.json"
    
    print("=" * 80)
    print(f"Multi-GPU Triton Benchmark - {args.config_name}")
    print("=" * 80)
    print(f"Triton URL:       {host}:{port}")
    print(f"Metrics URL:      {metrics_url}")
    print(f"Concurrency:      {concurrency_levels}")
    print(f"Iterations:       {args.iterations}")
    print(f"Output:           {output_path}")
    print()
    
    # Health check
    print("Checking Triton health...")
    try:
        health_response = requests.get(f"http://{host}:{port}/v2/health/ready", timeout=10)
        if health_response.status_code == 200:
            print("  ✓ Triton is ready")
        else:
            print(f"  ✗ Triton health check failed: {health_response.status_code}")
            return 1
    except Exception as e:
        print(f"  ✗ Could not connect to Triton: {e}")
        return 1
    
    # Check model readiness
    try:
        model_response = requests.get(f"http://{host}:{port}/v2/models/openclip_vit_b32/ready", timeout=10)
        if model_response.status_code == 200:
            print("  ✓ Model openclip_vit_b32 is ready")
        else:
            print(f"  ✗ Model not ready: {model_response.status_code}")
            return 1
    except Exception as e:
        print(f"  ✗ Could not check model: {e}")
        return 1
    
    print()
    
    # Prepare test image
    print("Preparing test image...")
    test_images = create_test_images(1)
    img_tensor = image_to_triton_tensor(test_images[0])
    print(f"  Image shape: {img_tensor.shape}")
    print()
    
    # Run benchmarks
    results = {
        "config_name": args.config_name,
        "timestamp": datetime.now().isoformat(),
        "triton_url": f"http://{host}:{port}",
        "metrics_url": metrics_url,
        "iterations_per_concurrency": args.iterations,
        "benchmarks": []
    }
    
    for concurrency in concurrency_levels:
        print(f"Testing concurrency {concurrency}...")
        
        # Collect metrics before
        metrics_before = fetch_triton_metrics(metrics_url)
        
        # Run benchmark
        bench_result = benchmark_concurrent(
            host, port, args.iterations, concurrency, img_tensor, args.warmup
        )
        
        # Collect metrics after
        time.sleep(1)  # Give metrics time to update
        metrics_after = fetch_triton_metrics(metrics_url)
        
        # Calculate delta metrics
        if metrics_before and metrics_after:
            bench_result["triton_metrics"] = {
                "inference_count_delta": metrics_after.get('inference_count', 0) - metrics_before.get('inference_count', 0),
                "queue_time_us": metrics_after.get('queue_time_us'),
                "compute_time_us": metrics_after.get('compute_time_us'),
                "per_gpu_utilization": metrics_after.get('per_gpu_utilization'),
            }
        
        results["benchmarks"].append(bench_result)
        print()
    
    # Save results
    print(f"Saving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Concurrency':<15} {'Throughput':<20} {'Latency p50':<15} {'Latency p95':<15}")
    print("-" * 80)
    for bench in results["benchmarks"]:
        print(f"{bench['concurrency']:<15} "
              f"{bench['throughput_imgs_per_sec']:>7.1f} img/s      "
              f"{bench['p50_ms']:>7.1f} ms      "
              f"{bench['p95_ms']:>7.1f} ms")
    print("=" * 80)
    print()
    print(f"✓ Benchmark complete! Results saved to {output_path}")
    print()
    print("Next steps:")
    print("  1. Review results with: python scripts/analyze_multigpu_results.py")
    print("  2. Collect metrics from: curl", metrics_url)
    print("  3. Check GPU utilization: nvidia-smi")
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
