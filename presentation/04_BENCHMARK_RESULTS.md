# Benchmark Results: Complete Data Reference

> All benchmark data collected across Steps 4–6B, organized for easy comparison.

---

## Table of Contents

1. [Hardware Tested](#1-hardware-tested)
2. [Step 4: Initial Triton vs PyTorch](#2-step-4-initial-triton-vs-pytorch)
3. [Step 5A: After Binary Protocol Fix](#3-step-5a-after-binary-protocol-fix)
4. [Step 5B: TensorRT EP vs ONNX CUDA EP](#4-step-5b-tensorrt-ep-vs-onnx-cuda-ep)
5. [Step 6A: 3-Way Comparison (A100)](#5-step-6a-3-way-comparison-a100)
6. [Step 6A: 3-Way Comparison (RTX 4080)](#6-step-6a-3-way-comparison-rtx-4080)
7. [Step 6B: Multi-GPU Scaling (4× RTX 4080)](#7-step-6b-multi-gpu-scaling-4-rtx-4080)
8. [Cross-Step Comparison Tables](#8-cross-step-comparison-tables)
9. [ONNX Model Profile](#9-onnx-model-profile)

---

## 1. Hardware Tested

| GPU | Type | VRAM | Provider | Steps |
|-----|------|:----:|----------|:-----:|
| RTX A4000 | Workstation | 16 GB | Vast.ai | 4, 5A, 5B |
| A100 SXM4 | Datacenter | 80 GB | Vast.ai | 6A |
| RTX 4080 | Consumer | 16 GB | Vast.ai | 6A, 6B |
| 4× RTX 4080 | Consumer (multi) | 4×16 GB | Vast.ai | 6B |

**Client:** MacBook Pro (macOS), benchmarking over internet to Vast.ai datacenters.

---

## 2. Step 4: Initial Triton vs PyTorch

**GPU:** RTX A4000 | **Protocol:** JSON `.tolist()` | **Date:** Feb 14, 2026

| Metric | PyTorch FastAPI | Triton (ONNX) | PyTorch Advantage |
|--------|:-:|:-:|:-:|
| Cold-start | 541ms | 614ms | 1.1× |
| Single p50 | 168ms | 564ms | 3.4× |
| Single mean | 189ms | 724ms | 3.8× |
| Batch-4 | 359ms | 1,863ms | 5.2× |
| Batch-8 | 447ms | 3,623ms | 8.1× |
| Batch-16 | 625ms | 6,690ms | 10.7× |
| **Batch-32** | **1,070ms** | **9,649ms** | **9.0×** |
| Concurrent req/s | 36.8 | 8.7 | 4.2× |

**Diagnosis:** Triton's poor showing was caused by JSON serialization, not inference.

---

## 3. Step 5A: After Binary Protocol Fix

**GPU:** RTX A4000 | **Protocol:** Binary tensor | **Date:** Feb 14, 2026

### 3.1 Serialization Overhead (The Root Cause)

| Batch Size | JSON `.tolist()` | Binary Encoding | Payload (JSON) | Payload (Binary) | JSON Slower By |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 52.5ms | 0.01ms | 2,980 KB | 588 KB | 4,799× |
| 4 | 211.4ms | 0.15ms | 11,918 KB | 2,352 KB | 1,403× |
| 8 | 428.4ms | 0.26ms | 23,836 KB | 4,704 KB | 1,642× |
| 16 | 887.9ms | 0.65ms | 47,672 KB | 9,408 KB | 1,363× |
| 32 | 1,842.2ms | 1.74ms | 95,344 KB | 18,816 KB | 1,058× |

### 3.2 PyTorch vs Triton (Post-Fix, Same GPU)

| Metric | PyTorch | Triton (Binary) | Before (JSON) | Improvement |
|--------|:-:|:-:|:-:|:-:|
| Single p50 | 168ms | 229ms | 564ms | 2.5× faster |
| Batch-32 | 1,070ms | 5,614ms | 9,649ms | 1.7× faster |
| Concurrent | 36.8 req/s | 35.8 req/s | 8.7 req/s | 4.1× faster |

**Finding:** Binary fix eliminated serialization bottleneck. Remaining gap (5.2× for batch-32) is inherent ONNX Runtime vs PyTorch GPU performance.

---

## 4. Step 5B: TensorRT EP vs ONNX CUDA EP

**GPU:** RTX A4000 | **Metrics:** Server-side (Triton Prometheus) | **Date:** Feb 15, 2026

### 4.1 Server-Side Metrics (No Network Overhead)

| Metric | ONNX CUDA EP | TensorRT EP (FP16) | Speedup |
|--------|:-:|:-:|:-:|
| **GPU compute** | 31.1ms | **4.7ms** | **6.6×** |
| Server request | 41.9ms | 15.9ms | 2.6× |
| Input processing | 1.2ms | 0.5ms | 2.4× |

### 4.2 Client-Side Latency (Includes Network)

| Metric | ONNX CUDA EP | TensorRT EP |
|--------|:-:|:-:|
| Mean | 359ms | 454ms |
| p50 | 339ms | 454ms |
| Min | 210ms | 241ms |

**Note:** Client latency is network-dominated (~200–400ms RTT). Server-side metrics reveal the true GPU performance.

### 4.3 Comparison Across All Backends (Server-Side GPU Compute)

| Backend | GPU Compute | vs ONNX Baseline |
|---------|:-:|:-:|
| ONNX Runtime (CUDA EP) | 31.1ms | baseline |
| **TensorRT EP (FP16)** | **4.7ms** | **6.6× faster** |
| PyTorch (native) | ~10–15ms (est.) | ~2–3× faster |

---

## 5. Step 6A: 3-Way Comparison (A100)

**GPU:** A100 SXM4 80GB | **All 3 backends on same instance** | **Date:** Feb 15, 2026

### 5.1 Single-Image Latency (30 iterations)

| Metric | PyTorch | Triton ONNX | Triton TRT |
|--------|:-:|:-:|:-:|
| **Mean** | **56.9ms** ⚡ | 182.9ms | 212.5ms |
| Median | **55.8ms** | 180.2ms | 200.3ms |
| P95 | **68.9ms** | 199.4ms | 324.9ms |
| P99 | **72.3ms** | 210.5ms | 350.2ms |
| Server GPU Compute | ~10–15ms (est.) | **4.4ms** ⚡ | 29.1ms |

### 5.2 Batch Throughput (img/s)

| Batch Size | PyTorch | Triton ONNX | Triton TRT |
|:---:|:-:|:-:|:-:|
| 1 | **17.5** | 4.8 | 5.0 |
| 4 | **29.8** | 18.1 | 10.2 |
| 8 | **39.7** | 27.3 | timeout |
| 16 | **52.0** | 30.9 | timeout |
| 32 | **64.3** | 32.3 | timeout |

### 5.3 Batch Latency (ms)

| Batch Size | PyTorch | Triton ONNX | Triton TRT |
|:---:|:-:|:-:|:-:|
| 1 | **57.1** | 206.2 | 199.8 |
| 4 | **134.2** | 220.6 | 392.4 |
| 8 | **201.7** | 292.7 | — |
| 16 | **308.0** | 517.3 | — |
| 32 | **497.6** | 990.7 | — |

### 5.4 Latency Breakdown (Triton ONNX, Single Image)

| Component | Time | % of Total |
|-----------|:----:|:----------:|
| GPU compute | 4.4ms | 2.4% |
| Server overhead | 10.7ms | 5.8% |
| **Network + transfer** | **167.8ms** | **91.7%** |
| **Total (client)** | **182.9ms** | 100% |

### 5.5 VRAM Usage (All 3 Backends Loaded)

| Resource | Value |
|----------|:-----:|
| Total VRAM used | 3.3 GB |
| GPU utilization | 4.1% |
| GPU power (idle) | 67W / 400W TDP |

---

## 6. Step 6A: 3-Way Comparison (RTX 4080)

**GPU:** RTX 4080 16GB | **All 3 backends on same instance** | **Date:** Feb 15, 2026

### 6.1 Single-Image Latency (30 iterations)

| Metric | PyTorch | Triton ONNX | Triton TRT |
|--------|:-:|:-:|:-:|
| Mean | 219.2ms† | 337.4ms | 336.5ms |
| Median | 198.5ms† | 274.4ms | 272.5ms |
| Server GPU Compute | N/A | 5.7ms | **2.0ms** ⚡ |

†PyTorch showed anomalous performance on this instance.

### 6.2 Key Finding: GPU Architecture Changes the Winner

| GPU | Best for GPU Compute | Compute Time | Runner-Up | Compute Time |
|-----|:---:|:---:|:---:|:---:|
| **A100 SXM4** | ONNX CUDA EP | **4.4ms** | TRT EP | 29.1ms |
| **RTX 4080** | TensorRT EP | **2.0ms** | ONNX CUDA EP | 5.7ms |

### 6.3 VRAM Usage

| Resource | RTX 4080 | A100 |
|----------|:--------:|:----:|
| Total VRAM used | 2.8 GB | 3.3 GB |
| GPU utilization | 17.4% | 4.1% |

---

## 7. Step 6B: Multi-GPU Scaling (4× RTX 4080)

**GPU:** 4× RTX 4080 | **Interconnect:** PCIe Gen4 | **Date:** Feb 15, 2026

### 7.1 Throughput vs Concurrency

| Concurrency | Throughput (img/s) | Latency p50 | Latency p95 |
|:-----------:|:------------------:|:-----------:|:-----------:|
| 1 | 3.4 | 267.5ms | 457.0ms |
| 4 | 11.0 | 270.2ms | 839.9ms |
| 8 | 20.2 | 255.0ms | 836.4ms |
| 16 | 31.9 | 302.8ms | 999.5ms |
| **32** | **43.2** ⭐ | 415.9ms | 1,300.5ms |
| 64 | 37.4 | 999.6ms | 2,111.4ms |
| 128 | 37.1 | 1,637.8ms | 2,250.0ms |

### 7.2 Scaling Efficiency

| Config | Throughput | Scaling Factor | Efficiency | Cost/hr | Cost per 1K imgs |
|--------|:---------:|:--------------:|:----------:|:-------:|:-----------------:|
| 1× RTX 4080 | 24.3 img/s | 1.0× | baseline | $2.00 | $0.023 |
| **4× RTX 4080** | **43.2 img/s** | **1.8×** | **45%** | $8.00 | $0.051 |
| 1× A100 (ref) | 64.3 img/s | 2.6× | — | $1.50 | $0.006 |

### 7.3 Server-Side Timing (Triton Metrics)

| Metric | Per-Request Average |
|--------|:-------------------:|
| Total request time | 17.9ms |
| Queue time | 8.8ms (49%) |
| GPU compute | 8.4ms (47%) |
| Input processing | 0.7ms (4%) |

### 7.4 Dynamic Batching Efficiency

| Metric | Value |
|--------|:-----:|
| Total requests | 721 |
| Model executions | 560 |
| Batch efficiency | 22% |
| Avg batch size | ~1.3 |

### 7.5 GPU Memory Usage (Per GPU)

| GPU | VRAM Used | VRAM Total | Power |
|:---:|:---------:|:----------:|:-----:|
| GPU 0 | 2.47 GB | 16 GB | 14.9W |
| GPU 1 | 2.45 GB | 16 GB | 6.7W |
| GPU 2 | 2.46 GB | 16 GB | 21.8W |
| GPU 3 | 2.46 GB | 16 GB | 11.1W |
| **Total** | **9.84 GB** | **64 GB** | **54.5W** |

---

## 8. Cross-Step Comparison Tables

### 8.1 Single-Image Client Latency Evolution

| Step | Backend | GPU | Protocol | Latency (p50) |
|:----:|---------|-----|----------|:-------------:|
| 4 | Triton ONNX | A4000 | JSON | 564ms |
| 5A | Triton ONNX | A4000 | Binary | 229ms |
| 5B | Triton TRT EP | A4000 | Binary | 454ms* |
| 6A | PyTorch | A100 | base64 JPEG | **55.8ms** ⚡ |
| 6A | Triton ONNX | A100 | Binary | 180.2ms |
| 6A | Triton TRT | A100 | Binary | 200.3ms |
| 6A | Triton ONNX | RTX 4080 | Binary | 274.4ms |
| 6A | Triton TRT | RTX 4080 | Binary | 272.5ms |

*High variance due to network, not GPU performance.

### 8.2 Server-Side GPU Compute Evolution

| Step | Backend | GPU | Compute Time |
|:----:|---------|-----|:------------:|
| 5A | ONNX CUDA EP | A4000 | 31.1ms |
| 5B | TensorRT EP | A4000 | **4.7ms** |
| 6A | ONNX CUDA EP | A100 | **4.4ms** |
| 6A | TensorRT EP | A100 | 29.1ms |
| 6A | ONNX CUDA EP | RTX 4080 | 5.7ms |
| 6A | TensorRT EP | RTX 4080 | **2.0ms** ⚡ |
| 6B | ONNX CUDA EP | 4× RTX 4080 | 8.4ms |

### 8.3 Throughput Comparison (Best Results Per Step)

| Step | Config | Best Throughput | Notes |
|:----:|--------|:---------------:|-------|
| 4 | PyTorch, A4000 | 29.9 img/s | Batch-32 |
| 5A | PyTorch, A4000 | 36.8 req/s | Concurrent |
| 6A | PyTorch, A100 | **64.3 img/s** ⚡ | Batch-32 |
| 6A | ONNX, RTX 4080 | 24.3 img/s | Batch-32 |
| 6B | ONNX, 4× RTX 4080 | 43.2 img/s | Concurrency-32 |

### 8.4 Protocol Impact Summary

| Format | Payload per Image | Network Time | Used By |
|--------|:-----------------:|:------------:|---------|
| JSON `.tolist()` | ~2,980 KB | ~1,800ms (batch-32) | Step 4 (initial) |
| Binary tensor | ~602 KB | ~150–170ms | Triton (Steps 5–6) |
| Base64 JPEG | ~10 KB | ~25–35ms | PyTorch FastAPI |

**Takeaway:** 300× payload difference between JSON and base64 JPEG. Protocol design dominates performance.

---

## 9. ONNX Model Profile

**Model:** OpenCLIP ViT-B/32 | **File:** 335.5 MB | **Opset:** 14 | **Nodes:** 2,272

### 9.1 Operator Breakdown

| Operator | Total Time | % of Compute | Invocations |
|----------|:----------:|:------------:|:-----------:|
| **MatMul** | 28,153ms | 75.7% | 30,195 |
| Gemm | 2,644ms | 7.1% | 5,940 |
| BiasGelu | 1,467ms | 3.9% | 5,940 |
| Conv | 1,365ms | 3.7% | 495 |
| Transpose | 1,227ms | 3.3% | 48,015 |
| LayerNormalization | 733ms | 2.0% | 12,870 |
| Add | 437ms | 1.2% | 24,255 |
| Gather | 349ms | 0.9% | 54,945 |
| Mul | 173ms | 0.5% | 23,760 |
| Softmax | 149ms | 0.4% | 5,940 |

### 9.2 Batch Scaling (CPU)

| Batch Size | Latency | Images/sec | Scaling |
|:---:|:---:|:---:|:---:|
| 1 | 29.9ms | 33.4 | 1.0× |
| 4 | 81.7ms | 49.0 | 2.7× |
| 8 | 152.9ms | 52.3 | 5.1× |
| 16 | 290.3ms | 55.1 | 9.7× |
| 32 | 585.0ms | 54.7 | 19.5× |

### 9.3 Session Creation (Cold Start)

| Metric | Value |
|--------|:-----:|
| Mean | 278.9ms |
| Min | 261.3ms |
| Max | 293.3ms |

---

## Raw Data Files

| File | Contents |
|------|----------|
| `benchmark_results/step6a_a100_remote.json` | Step 6A A100 raw data |
| `benchmark_results/step6a_rtx4080_remote.json` | Step 6A RTX 4080 raw data |
| `benchmark_results/step6b_rtx4080_4gpu-rtx4080_20260215_191823.json` | Step 6B multi-GPU raw data |
| `benchmark_results/tensorrt_ep_results.json` | Step 5B TRT EP results |
| `benchmark_results/triton_rtx3070_20260214_083702.json` | Step 4 Triton baseline |
| `benchmark_results/pytorch_rtx3070_20260214_090742.json` | Step 4 PyTorch baseline |
| `benchmark_results/onnx_profile_20260214_114632.json` | ONNX profiling data |

---

*All benchmarks conducted remotely from macOS client to Vast.ai GPU instances, February 14–15, 2026.*
