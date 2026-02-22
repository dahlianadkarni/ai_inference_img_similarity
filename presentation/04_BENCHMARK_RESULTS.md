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
8. [Step 7: 5-Way Protocol Comparison (A100)](#8-step-7-5-way-protocol-comparison-a100)
9. [Step 7B: 5-Way Protocol Comparison (RTX 4090)](#9-step-7b-5-way-protocol-comparison-rtx-4090)
10. [Cross-Step Comparison Tables](#10-cross-step-comparison-tables)
11. [ONNX Model Profile](#11-onnx-model-profile)
12. [Step 8: Local Kubernetes (kind) — CPU Baseline](#12-step-8-local-kubernetes-kind--cpu-baseline)

---

## 1. Hardware Tested

| GPU | Type | VRAM | Provider | Steps |
|-----|------|:----:|----------|:-----:|
| RTX A4000 | Workstation | 16 GB | Vast.ai | 4, 5A, 5B |
| A100 SXM4 | Datacenter | 80 GB | Vast.ai | 6A, 7 |
| RTX 4080 | Consumer | 16 GB | Vast.ai | 6A, 6B |
| 4× RTX 4080 | Consumer (multi) | 4×16 GB | Vast.ai | 6B |
| RTX 4090 | Consumer | 24 GB | Vast.ai | 7B |
| Apple M-series (CPU) | Local (kind) | — | Local machine | 8 |

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
| Batch-32 throughput (img/s) | 22.7 | 24.3 | 18.6 |
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

| Config | Throughput | Scaling Factor | Efficiency | Cost/hr (actual) | Cost per 1K imgs |
|--------|:---------:|:--------------:|:----------:|:-------:|:-----------------:|
| 1× RTX 4080 | 24.3 img/s | 1.0× | baseline | $0.092 | **$0.001** ⚡ |
| **4× RTX 4080** | **43.2 img/s** | **1.8×** | **45%** | $0.30–2.00 | $0.002–0.013 |
| 1× A100 (ref) | 64.3 img/s | 2.6× | — | $0.85 | $0.004 |

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

## 8. Step 7: 5-Way Protocol Comparison (A100)

**GPU:** A100 SXM4 80GB | **Location:** Massachusetts | **IP:** 207.180.148.74 | **Date:** Feb 20, 2026  
**5 backends on same instance (step6a docker-compose) | 30 iterations**

### 8.1 Serial p50 Latency (ms)

| Backend | b=1 | b=4 | b=8 | b=16 | b=32 |
|---------|----:|----:|----:|-----:|-----:|
| PyTorch HTTP (JPEG ~10KB) | **64.2** | **150.9** | **214.9** | **344.2** | **659.4** |
| Triton ONNX HTTP | 208.7 | 427.3 | 869.2 | 1974.2 | 5266.9 |
| Triton ONNX gRPC | 217.5 | 303.5 | 617.8 | 1292.1 | 3126.8 |
| Triton TRT  HTTP | 171.4 | 295.1 | 632.8 | 1079.7 | 2640.2 |
| Triton TRT  gRPC | 200.2 | 317.9 | 409.6 | 747.5 | 2224.1 |

### 8.2 Serial Throughput (img/s)

| Backend | b=1 | b=4 | b=8 | b=16 | b=32 |
|---------|----:|----:|----:|-----:|-----:|
| PyTorch HTTP | **15.6** | **26.5** | **37.2** | **46.5** | **48.5** |
| Triton ONNX HTTP | 4.8 | 9.4 | 9.2 | 8.1 | 6.1 |
| Triton ONNX gRPC | 4.6 | 13.2 | 12.9 | 12.4 | 10.2 |
| Triton TRT  HTTP | 5.8 | 13.6 | 12.6 | 14.8 | 12.1 |
| Triton TRT  gRPC | 5.0 | 12.6 | 19.5 | 21.4 | 14.4 |

### 8.3 gRPC vs HTTP Speedup (p50; >1.0× = gRPC faster)

| Pair | b=1 | b=4 | b=8 | b=16 | b=32 |
|------|----:|----:|----:|-----:|-----:|
| ONNX gRPC/HTTP | 0.96× | 1.41× | 1.41× | 1.53× | 1.68× |
| TRT  gRPC/HTTP | 0.86× | 0.93× | 1.54× | 1.44× | 1.19× |

### 8.4 Concurrent Throughput (batch=1, img/s)

| Backend | conc=1 | conc=8 | conc=16 |
|---------|-------:|-------:|--------:|
| PyTorch HTTP | 14.1 | 24.0 | 30.5 |
| Triton ONNX HTTP | 5.1 | **28.8** | **43.6** |
| Triton ONNX gRPC | 4.7 | 16.6 | 7.5 |
| Triton TRT  HTTP | 5.5 | 32.4 | 22.4 |
| Triton TRT  gRPC | 4.7 | 14.3 | 15.4 |

### 8.5 Server-Side GPU Compute (batch=1)

| Backend | GPU compute |
|---------|------------:|
| Triton ONNX HTTP | 8.67ms |
| Triton ONNX gRPC | 4.04ms |
| Triton TRT HTTP | ⚠️ 1657.75ms (engine compilation) |
| Triton TRT gRPC | **3.63ms** (post-cache, true latency) |

---

## 9. Step 7B: 5-Way Protocol Comparison (RTX 4090)

**GPU:** RTX 4090 24GB | **Location:** Pennsylvania | **IP:** 173.185.79.174 | **Cost:** $0.391/hr | **Date:** Feb 20, 2026  
**Same 5 backends | 30 iterations**

### 9.1 Serial p50 Latency (ms)

| Backend | b=1 | b=4 | b=8 | b=16 | b=32 |
|---------|----:|----:|----:|-----:|-----:|
| PyTorch HTTP (JPEG ~10KB) | **137.2** | **303.0** | **355.0** | **497.4** | **812.7** |
| Triton ONNX HTTP | 272.1 | 386.3 | 1466.9 | 3307.5 | 6731.8 |
| Triton ONNX gRPC | 318.5 | 432.2 | 724.9 | 2066.1 | 6610.7 |
| Triton TRT  HTTP | 269.9 | 424.1 | 783.9 | 4127.1 | 9130.2 |
| Triton TRT  gRPC | 312.5 | 501.4 | 956.0 | 2966.0 | 8434.8 |

### 9.2 gRPC vs HTTP Speedup (p50)

| Pair | b=1 | b=4 | b=8 | b=16 | b=32 |
|------|----:|----:|----:|-----:|-----:|
| ONNX gRPC/HTTP | 0.85× | 0.89× | 2.02× | 1.60× | 1.02× |
| TRT  gRPC/HTTP | 0.86× | 0.85× | 0.82× | 1.39× | 1.08× |

### 9.3 Concurrent Throughput (batch=1, img/s)

| Backend | conc=1 | conc=8 | conc=16 |
|---------|-------:|-------:|--------:|
| PyTorch HTTP | 6.7 | **47.3** | **49.1** |
| Triton ONNX HTTP | 3.2 | 17.5 | 32.2 |
| Triton ONNX gRPC | 2.6 | 4.1 | 4.0 |
| Triton TRT  HTTP | 2.8 | 20.1 | 21.8 |
| Triton TRT  gRPC | 2.9 | 4.2 | 6.1 |

### 9.4 Cross-GPU Comparison — batch=1 p50 (ms)

| Backend | A100 SXM4 (MA) | RTX 4090 (PA) | RTX/A100 ratio |
|---------|---------------:|--------------:|---------------:|
| PyTorch HTTP | 64.2 | 137.2 | 2.14× slower |
| Triton ONNX HTTP | 208.7 | 272.1 | 1.30× slower |
| Triton ONNX gRPC | 217.5 | 318.5 | 1.46× slower |
| Triton TRT  HTTP | 171.4 | 269.9 | 1.57× slower |
| Triton TRT  gRPC | 200.2 | 312.5 | 1.56× slower |

**Key insight:** PyTorch shows the largest relative slowdown (2.1×) because it scales with GPU compute speed. Triton ONNX/TRT baselines are only 1.3–1.6× slower — they remain transport-bound at 602KB regardless of GPU tier.

---

## 10. Cross-Step Comparison Tables

### 10.1 Single-Image Client Latency Evolution

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
| 7 | PyTorch HTTP | A100 | base64 JPEG | **64.2ms** ⚡ |
| 7 | Triton ONNX HTTP | A100 | Binary | 208.7ms |
| 7 | Triton ONNX gRPC | A100 | Binary (gRPC) | 217.5ms |
| 7 | PyTorch HTTP | RTX 4090 | base64 JPEG | **137.2ms** |
| 7 | Triton ONNX HTTP | RTX 4090 | Binary | 272.1ms |
| 7 | Triton ONNX gRPC | RTX 4090 | Binary (gRPC) | 318.5ms |

*High variance due to network, not GPU performance.

### 10.2 Server-Side GPU Compute Evolution

| Step | Backend | GPU | Compute Time |
|:----:|---------|-----|:------------:|
| 5A | ONNX CUDA EP | A4000 | 31.1ms |
| 5B | TensorRT EP | A4000 | **4.7ms** |
| 6A | ONNX CUDA EP | A100 | **4.4ms** |
| 6A | TensorRT EP | A100 | 29.1ms |
| 6A | ONNX CUDA EP | RTX 4080 | 5.7ms |
| 6A | TensorRT EP | RTX 4080 | **2.0ms** ⚡ |
| 6B | ONNX CUDA EP | 4× RTX 4080 | 8.4ms |
| 7 | ONNX CUDA EP (HTTP) | A100 | 8.67ms |
| 7 | ONNX CUDA EP (gRPC) | A100 | 4.04ms |
| 7 | TRT EP (gRPC, cached) | A100 | **3.63ms** |
| 7 | ONNX CUDA EP (gRPC) | RTX 4090 | 4.41ms |
| 7 | TRT EP (gRPC, cached) | RTX 4090 | **2.68ms** |

### 10.3 Throughput Comparison (Best Results Per Step)

| Step | Config | Best Throughput | Notes |
|:----:|--------|:---------------:|-------|
| 4 | PyTorch, A4000 | 29.9 img/s | Batch-32 |
| 5A | PyTorch, A4000 | 36.8 req/s | Concurrent |
| 6A | PyTorch, A100 | **64.3 img/s** ⚡ | Batch-32 |
| 6A | ONNX, RTX 4080 | 24.3 img/s | Batch-32 |
| 6B | ONNX, 4× RTX 4080 | 43.2 img/s | Concurrency-32 |
| 7 | ONNX HTTP, A100 | 43.6 img/s | Concurrent, conc=16 |
| 7 | PyTorch, RTX 4090 | 49.1 img/s | Concurrent, conc=16 |

### 10.4 Protocol Impact Summary

| Format | Payload per Image | Network Time | Used By |
|--------|:-----------------:|:------------:|---------|
| JSON `.tolist()` | ~2,980 KB | ~1,800ms (batch-32) | Step 4 (initial) |
| Binary tensor (HTTP) | ~602 KB | ~150–170ms | Triton HTTP (Steps 5–7) |
| Binary tensor (gRPC) | ~602 KB | ~155–175ms at batch=1; ~120ms at batch=8+ | Triton gRPC (Step 7) |
| Base64 JPEG | ~10 KB | ~25–35ms | PyTorch FastAPI |

**Takeaway:** gRPC and HTTP/1.1 binary are functionally equivalent at batch=1 — the bottleneck is 602KB payload transfer regardless of framing. Protocol choice only matters for batch≥4 (gRPC ~1.4–2.0× faster) or under high concurrency (HTTP wins — gRPC client has channel contention).

---

## 11. ONNX Model Profile

**Model:** OpenCLIP ViT-B/32 | **File:** 335.5 MB | **Opset:** 14 | **Nodes:** 2,272

### 11.1 Operator Breakdown

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

### 11.2 Batch Scaling (CPU)

| Batch Size | Latency | Images/sec | Scaling |
|:---:|:---:|:---:|:---:|
| 1 | 29.9ms | 33.4 | 1.0× |
| 4 | 81.7ms | 49.0 | 2.7× |
| 8 | 152.9ms | 52.3 | 5.1× |
| 16 | 290.3ms | 55.1 | 9.7× |
| 32 | 585.0ms | 54.7 | 19.5× |

### 11.3 Session Creation (Cold Start)

| Metric | Value |
|--------|:-----:|
| Mean | 278.9ms |
| Min | 261.3ms |
| Max | 293.3ms |

---

---

## 12. Step 8: Local Kubernetes (kind) — CPU Baseline

**Infrastructure:** kind v0.31.0, Kubernetes v1.35.0, Apple Silicon ARM64, macOS 15.7.3  
**Image:** `photo-duplicate-inference:k8s-cpu` (388 MB, CPU-only, same `Dockerfile` as GPU image)  
**Port:** `localhost:8092` (NodePort — fully isolated from docker-compose ports 8002/8003/8004)  
**Date:** 2026-02-21

### 12.1 Single-Image Inference Latency (CPU, concurrency=5)

| Metric | CPU (local kind) | GPU A100 (remote) | CPU/GPU ratio |
|--------|:-:|:-:|:-:|
| **p50** | **1,020ms** | 64ms | **16× slower** |
| p75 | 1,200ms | — | — |
| p90 | 3,450ms | — | — |
| p95 | 4,150ms | — | — |
| Fastest | 400ms | — | — |
| Throughput | 3.5 req/s | ~30 req/s | **8× lower** |

*High p90/p95 are expected: CPU can process ~1 image/s; at concurrency=5 the queue builds up.*

### 12.2 HPA Scale-Up Under Load (600 requests, concurrency=30)

| Event | CPU % of request | Replicas | Time from load start |
|-------|:-:|:-:|:-:|
| Idle (baseline) | 1% | 2 | — |
| Load start | **210%** | 2 | ~15s |
| First scale decision | **397% (1988m/2000m)** | 2→4 | ~30s |
| Second scale decision | 397% | 4→6 (max) | ~60s |
| Post-load (tapering) | 145% | 6 | ~90s |
| Idle (after stabilization) | 1% | 6→2 | +3 min |

### 12.3 HPA Events (`kubectl describe hpa`)

```
Normal  SuccessfulRescale  New size: 4; reason: cpu resource utilization above target
Normal  SuccessfulRescale  New size: 6; reason: cpu resource utilization above target
```

### 12.4 K8s vs docker-compose Coexistence

| Setup | Port | Manager | Running simultaneously? |
|-------|:----:|---------|:---:|
| docker-compose PyTorch | 8002 | docker-compose | ✅ |
| docker-compose Triton ONNX | 8003/8004 | docker-compose | ✅ |
| kind K8s PyTorch | **8092** | kubectl | ✅ |

*Verified: both setups run side-by-side. Zero interference.*

### 12.5 Resource Utilization at Peak Load

```
NAME                                CPU(cores)   MEMORY
inference-deploy-...-8zh2d          1988m        942Mi   ← saturated at 2000m limit
inference-deploy-...-dgtcg          1989m        1134Mi  ← saturated at 2000m limit
inference-deploy-...-6ckwm          311m         353Mi   ← new pod starting up
inference-deploy-...-x2tsj          309m         484Mi   ← new pod starting up
```

---

## Raw Data Files

| File | Contents |
|------|----------|
| `benchmark_results/step6a_a100_remote.json` | Step 6A A100 raw data |
| `benchmark_results/step6a_rtx4080_remote.json` | Step 6A RTX 4080 raw data |
| `benchmark_results/step7_5way_20260220_221313.json` | Step 7 A100 5-way gRPC/HTTP raw data |
| `benchmark_results/step7_5way_20260220_232454.json` | Step 7 RTX 4090 5-way gRPC/HTTP raw data |
| `benchmark_results/step6b_rtx4080_4gpu-rtx4080_20260215_191823.json` | Step 6B multi-GPU raw data |
| `benchmark_results/tensorrt_ep_results.json` | Step 5B TRT EP results |
| `benchmark_results/triton_rtx3070_20260214_083702.json` | Step 4 Triton baseline |
| `benchmark_results/pytorch_rtx3070_20260214_090742.json` | Step 4 PyTorch baseline |
| `benchmark_results/onnx_profile_20260214_114632.json` | ONNX profiling data |

---

*GPU benchmarks: macOS client → Vast.ai GPU instances, February 14–20, 2026.*  
*Step 8 K8s benchmarks: local kind cluster on Apple Silicon, February 21, 2026.*
