#!/usr/bin/env python3
"""
Step 6A: Benchmark all 3 backends locally (after SSH into instance).

This script runs INSIDE the Vast.ai instance where all 3 backends are deployed.
It connects via localhost, eliminating network latency for more accurate results.

Backends tested:
  1. PyTorch FastAPI (localhost:8002)
  2. Triton ONNX CUDA EP (localhost:8010)
  3. Triton TensorRT EP (localhost:8020)

Usage (after SSH into Vast.ai instance):
  # Install dependencies first
  pip3 install numpy pillow tritonclient[http] requests

  # Run benchmark
  python3 benchmark_all_three_local.py --iterations 50

  # Copy results back to your Mac
  cat benchmark_results/step6a_local_comparison.json

Note: This eliminates network latency, showing true backend performance.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
from PIL import Image

try:
    import tritonclient.http as httpclient
    TRITONCLIENT_AVAILABLE = True
except ImportError:
    TRITONCLIENT_AVAILABLE = False
    print("ERROR: tritonclient not available")
    print("Install with: pip3 install tritonclient[http]")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZES = [1, 4, 8, 16, 32]
CONCURRENT_REQUESTS = 16

# Localhost ports (from docker-compose-step6a.yml)
PYTORCH_PORT = 8002
TRITON_ONNX_HTTP = 8010
TRITON_ONNX_METRICS = 8012
TRITON_TRT_HTTP = 8020
TRITON_TRT_METRICS = 8022


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_image(size=(224, 224)) -> np.ndarray:
    """Create a random test image as numpy array."""
    return np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)


def numpy_to_base64(img: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    import base64
    from io import BytesIO
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def parse_triton_metrics(metrics_text: str, model_name: str) -> Dict:
    """Parse Triton Prometheus metrics for a specific model."""
    metrics = {}
    
    pattern_success = rf'nv_inference_request_success{{model="{model_name}".*?}}\s+([\d.]+)'
    match = re.search(pattern_success, metrics_text)
    if match:
        metrics['nv_inference_request_success'] = float(match.group(1))
    
    pattern_compute = rf'nv_inference_compute_infer_duration_us{{model="{model_name}".*?}}\s+([\d.]+)'
    match = re.search(pattern_compute, metrics_text)
    if match:
        metrics['nv_inference_compute_infer_duration_us'] = float(match.group(1))
    
    pattern_request = rf'nv_inference_request_duration_us{{model="{model_name}".*?}}\s+([\d.]+)'
    match = re.search(pattern_request, metrics_text)
    if match:
        metrics['nv_inference_request_duration_us'] = float(match.group(1))
    
    return metrics


# =============================================================================
# PyTorch Backend Benchmarks
# =============================================================================

def benchmark_pytorch_single(iterations: int) -> Dict:
    """Benchmark PyTorch backend with single images."""
    print(f"\n{'='*60}")
    print("PyTorch Backend - Single Image")
    print(f"{'='*60}")
    
    url = f"http://localhost:{PYTORCH_PORT}/embed/base64"
    
    # Warmup
    print("Warming up...")
    img = create_test_image()
    img_b64 = numpy_to_base64(img)
    for _ in range(5):
        requests.post(url, json={"images": [img_b64]}, timeout=10)
    
    # Benchmark
    print(f"Running {iterations} iterations...")
    latencies = []
    
    for i in range(iterations):
        img = create_test_image()
        img_b64 = numpy_to_base64(img)
        
        start = time.time()
        response = requests.post(url, json={"images": [img_b64]}, timeout=10)
        elapsed_ms = (time.time() - start) * 1000
        
        if response.status_code != 200:
            print(f"  Error on iteration {i+1}: {response.status_code}")
            continue
        
        latencies.append(elapsed_ms)
    
    if not latencies:
        return {"error": "All requests failed"}
    
    results = {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "std_ms": np.std(latencies),
        "iterations": len(latencies),
    }
    
    print(f"\nResults:")
    print(f"  Mean:   {results['mean_ms']:.1f}ms")
    print(f"  Median: {results['median_ms']:.1f}ms")
    print(f"  P95:    {results['p95_ms']:.1f}ms")
    print(f"  P99:    {results['p99_ms']:.1f}ms")
    print(f"  Min:    {results['min_ms']:.1f}ms")
    print(f"  Max:    {results['max_ms']:.1f}ms")
    
    return results


def benchmark_pytorch_batch(batch_sizes: List[int]) -> Dict:
    """Benchmark PyTorch backend with different batch sizes."""
    print(f"\n{'='*60}")
    print("PyTorch Backend - Batch Processing")
    print(f"{'='*60}")
    
    url = f"http://localhost:{PYTORCH_PORT}/embed/base64"
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create batch
        images = [numpy_to_base64(create_test_image()) for _ in range(batch_size)]
        
        # Warmup
        requests.post(url, json={"images": images}, timeout=30)
        
        # Benchmark (5 runs)
        latencies = []
        for _ in range(5):
            start = time.time()
            response = requests.post(url, json={"images": images}, timeout=30)
            elapsed_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                latencies.append(elapsed_ms)
        
        if latencies:
            mean_ms = np.mean(latencies)
            throughput = (batch_size / mean_ms) * 1000  # images/sec
            
            results[f"batch_{batch_size}"] = {
                "mean_ms": mean_ms,
                "min_ms": np.min(latencies),
                "max_ms": np.max(latencies),
                "throughput_imgs_per_sec": throughput,
            }
            
            print(f"  Mean latency: {mean_ms:.1f}ms ({np.min(latencies):.1f}-{np.max(latencies):.1f}ms)")
            print(f"  Throughput:   {throughput:.1f} img/s")
    
    return results


# =============================================================================
# Triton Backend Benchmarks
# =============================================================================

def benchmark_triton_single(http_port: int, metrics_port: int, 
                            model_name: str, label: str, iterations: int) -> Dict:
    """Benchmark Triton backend with single images and collect server metrics."""
    print(f"\n{'='*60}")
    print(f"{label} - Single Image")
    print(f"{'='*60}")
    
    url = f"localhost:{http_port}"
    metrics_url = f"http://localhost:{metrics_port}/metrics"
    
    try:
        client = httpclient.InferenceServerClient(url=url)
    except Exception as e:
        return {"error": f"Failed to connect: {e}"}
    
    # Warmup
    print("Warming up...")
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    inp = [httpclient.InferInput("image", img.shape, "FP32")]
    inp[0].set_data_from_numpy(img)
    out = [httpclient.InferRequestedOutput("embedding")]
    
    for _ in range(5):
        client.infer(model_name, inp, outputs=out)
    
    # Get initial metrics
    try:
        m0_text = requests.get(metrics_url, timeout=5).text
        m0 = parse_triton_metrics(m0_text, model_name)
    except:
        m0 = {}
        print("  Warning: Could not fetch initial metrics")
    
    # Benchmark
    print(f"Running {iterations} iterations...")
    latencies = []
    
    for i in range(iterations):
        img = np.random.rand(1, 3, 224, 224).astype(np.float32)
        inp = [httpclient.InferInput("image", img.shape, "FP32")]
        inp[0].set_data_from_numpy(img)
        out = [httpclient.InferRequestedOutput("embedding")]
        
        start = time.time()
        try:
            result = client.infer(model_name, inp, outputs=out)
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)
        except Exception as e:
            print(f"  Error on iteration {i+1}: {e}")
            continue
    
    # Get final metrics
    try:
        m1_text = requests.get(metrics_url, timeout=5).text
        m1 = parse_triton_metrics(m1_text, model_name)
    except:
        m1 = {}
        print("  Warning: Could not fetch final metrics")
    
    if not latencies:
        return {"error": "All requests failed"}
    
    results = {
        "client_mean_ms": np.mean(latencies),
        "client_median_ms": np.median(latencies),
        "client_p95_ms": np.percentile(latencies, 95),
        "client_p99_ms": np.percentile(latencies, 99),
        "client_min_ms": np.min(latencies),
        "client_max_ms": np.max(latencies),
        "client_std_ms": np.std(latencies),
        "iterations": len(latencies),
    }
    
    # Calculate server-side metrics
    if m0 and m1 and 'nv_inference_request_success' in m0 and 'nv_inference_request_success' in m1:
        reqs = m1['nv_inference_request_success'] - m0['nv_inference_request_success']
        if reqs > 0:
            compute_delta = m1.get('nv_inference_compute_infer_duration_us', 0) - m0.get('nv_inference_compute_infer_duration_us', 0)
            request_delta = m1.get('nv_inference_request_duration_us', 0) - m0.get('nv_inference_request_duration_us', 0)
            
            results['server_compute_ms'] = (compute_delta / reqs) / 1000
            results['server_request_ms'] = (request_delta / reqs) / 1000
    
    print(f"\nClient-side results (localhost, no network):")
    print(f"  Mean:   {results['client_mean_ms']:.2f}ms")
    print(f"  Median: {results['client_median_ms']:.2f}ms")
    print(f"  P95:    {results['client_p95_ms']:.2f}ms")
    print(f"  P99:    {results['client_p99_ms']:.2f}ms")
    print(f"  Min:    {results['client_min_ms']:.2f}ms")
    print(f"  Max:    {results['client_max_ms']:.2f}ms")
    
    if 'server_compute_ms' in results:
        print(f"\nServer-side results (from Triton metrics):")
        print(f"  GPU compute:   {results['server_compute_ms']:.2f}ms")
        print(f"  Total request: {results['server_request_ms']:.2f}ms")
        
        # Calculate overhead
        overhead_ms = results['client_mean_ms'] - results['server_request_ms']
        print(f"  Client overhead: {overhead_ms:.2f}ms (serialization + localhost)")
    
    return results


def benchmark_triton_batch(http_port: int, model_name: str, 
                           label: str, batch_sizes: List[int]) -> Dict:
    """Benchmark Triton backend with different batch sizes."""
    print(f"\n{'='*60}")
    print(f"{label} - Batch Processing")
    print(f"{'='*60}")
    
    url = f"localhost:{http_port}"
    
    try:
        client = httpclient.InferenceServerClient(url=url)
    except Exception as e:
        return {"error": f"Failed to connect: {e}"}
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create batch
        img = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
        inp = [httpclient.InferInput("image", img.shape, "FP32")]
        inp[0].set_data_from_numpy(img)
        out = [httpclient.InferRequestedOutput("embedding")]
        
        # Warmup
        client.infer(model_name, inp, outputs=out)
        
        # Benchmark (5 runs)
        latencies = []
        for _ in range(5):
            start = time.time()
            try:
                result = client.infer(model_name, inp, outputs=out)
                elapsed_ms = (time.time() - start) * 1000
                latencies.append(elapsed_ms)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if latencies:
            mean_ms = np.mean(latencies)
            min_ms = np.min(latencies)
            max_ms = np.max(latencies)
            throughput = (batch_size / mean_ms) * 1000  # images/sec
            
            results[f"batch_{batch_size}"] = {
                "mean_ms": mean_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "throughput_imgs_per_sec": throughput,
            }
            
            print(f"  Mean latency: {mean_ms:.1f}ms ({min_ms:.1f}-{max_ms:.1f}ms)")
            print(f"  Throughput:   {throughput:.1f} img/s")
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark all 3 backends locally (after SSH)")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations for single-image test")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch benchmarks")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip Triton ONNX benchmarks")
    parser.add_argument("--skip-trt", action="store_true", help="Skip Triton TRT benchmarks")
    parser.add_argument("--output", type=str, default="benchmark_results/step6a_local_comparison.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    print("="*70)
    print("STEP 6A: 3-WAY BACKEND COMPARISON (LOCAL)")
    print("="*70)
    print("Running from inside Vast.ai instance (localhost)")
    print(f"Iterations: {args.iterations}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print()
    
    # Check connectivity
    print("Checking services...")
    services = [
        ("PyTorch", f"http://localhost:{PYTORCH_PORT}/health"),
        ("Triton ONNX", f"http://localhost:{TRITON_ONNX_HTTP}/v2/health/ready"),
        ("Triton TRT", f"http://localhost:{TRITON_TRT_HTTP}/v2/health/ready"),
    ]
    
    for name, url in services:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"  ✓ {name} is ready")
            else:
                print(f"  ✗ {name} returned {resp.status_code}")
        except Exception as e:
            print(f"  ✗ {name} not accessible: {e}")
    
    print()
    
    results = {
        "config": {
            "location": "localhost (inside Vast.ai instance)",
            "iterations": args.iterations,
            "batch_sizes": BATCH_SIZES,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "backends": {}
    }
    
    # Benchmark PyTorch
    if not args.skip_pytorch:
        print("\n" + "="*70)
        print("BACKEND 1: PyTorch FastAPI")
        print("="*70)
        
        pytorch_results = {
            "single_image": benchmark_pytorch_single(args.iterations),
            "batch": benchmark_pytorch_batch(BATCH_SIZES),
        }
        results["backends"]["pytorch"] = pytorch_results
    
    # Benchmark Triton ONNX
    if not args.skip_onnx:
        print("\n" + "="*70)
        print("BACKEND 2: Triton ONNX CUDA EP")
        print("="*70)
        
        triton_onnx_results = {
            "single_image": benchmark_triton_single(
                TRITON_ONNX_HTTP, TRITON_ONNX_METRICS,
                "openclip_vit_b32", "Triton ONNX CUDA EP", args.iterations
            ),
            "batch": benchmark_triton_batch(
                TRITON_ONNX_HTTP, "openclip_vit_b32",
                "Triton ONNX CUDA EP", BATCH_SIZES
            ),
        }
        results["backends"]["triton_onnx"] = triton_onnx_results
    
    # Benchmark Triton TRT
    if not args.skip_trt:
        print("\n" + "="*70)
        print("BACKEND 3: Triton TensorRT EP")
        print("="*70)
        
        triton_trt_results = {
            "single_image": benchmark_triton_single(
                TRITON_TRT_HTTP, TRITON_TRT_METRICS,
                "openclip_vit_b32_trt", "Triton TensorRT EP", args.iterations
            ),
            "batch": benchmark_triton_batch(
                TRITON_TRT_HTTP, "openclip_vit_b32_trt",
                "Triton TensorRT EP", BATCH_SIZES
            ),
        }
        results["backends"]["triton_trt"] = triton_trt_results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n1. Client-Side Latency (localhost, minimal network overhead):")
    print(f"{'Backend':<25} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-" * 61)
    
    for backend_name, backend_data in results["backends"].items():
        if "single_image" in backend_data and "client_mean_ms" in backend_data["single_image"]:
            single = backend_data["single_image"]
            print(f"{backend_name:<25} {single['client_mean_ms']:>10.2f}  "
                  f"{single.get('client_p95_ms', 0):>10.2f}  "
                  f"{single.get('client_p99_ms', 0):>10.2f}")
    
    print("\n2. Server-Side GPU Compute Time (Triton only):")
    print(f"{'Backend':<25} {'GPU Compute (ms)':<20} {'Speedup vs ONNX':<20}")
    print("-" * 65)
    
    onnx_compute = None
    for backend_name, backend_data in results["backends"].items():
        if "single_image" in backend_data and "server_compute_ms" in backend_data["single_image"]:
            compute = backend_data["single_image"]["server_compute_ms"]
            if backend_name == "triton_onnx":
                onnx_compute = compute
            
            speedup = ""
            if onnx_compute and backend_name != "triton_onnx":
                speedup = f"{onnx_compute / compute:.2f}x"
            
            print(f"{backend_name:<25} {compute:>18.2f}  {speedup:>18}")
    
    print("\n3. Batch Throughput (batch-32):")
    print(f"{'Backend':<25} {'Latency (ms)':<15} {'Throughput (img/s)':<20}")
    print("-" * 60)
    
    for backend_name, backend_data in results["backends"].items():
        if "batch" in backend_data and "batch_32" in backend_data["batch"]:
            batch32 = backend_data["batch"]["batch_32"]
            print(f"{backend_name:<25} {batch32['mean_ms']:>13.1f}  "
                  f"{batch32['throughput_imgs_per_sec']:>18.1f}")
    
    print(f"\nFull results saved to: {output_path}")
    print("\nTo copy results to your Mac:")
    print(f"  cat {output_path}")


if __name__ == "__main__":
    main()
