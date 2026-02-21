# ML Inference Infrastructure: Executive Summary

> **Project:** Photo Near-Duplicate Detection — Inference Infrastructure Deep-Dive  
> **Duration:** Steps 1–7, culminating February 20, 2026  
> **Model:** OpenCLIP ViT-B/32 (335MB, 512-dim embeddings)  
> **Goal:** Hands-on exploration of production ML inference patterns

---

## The App in 30 Seconds

A macOS app that finds near-duplicate photos using AI embeddings:
1. **Scan** photos from Apple Photos library
2. **Generate embeddings** via OpenCLIP (the ML-heavy step)
3. **Group** similar images using cosine similarity
4. **Review & delete** duplicates through a local web UI

The embedding step is the bottleneck — and the entire infrastructure project focuses on **how to serve that model efficiently**.

---

## What I Built & Explored

| Step | What | Key Outcome |
|:----:|------|-------------|
| **1** | Split app into client + inference service | Clean separation of concerns; stateless API boundary |
| **2** | Containerize inference service (Docker) | Reproducible deployment; `/embed` + `/healthz` endpoints |
| **3** | Deploy to cloud GPU (Vast.ai) | NVIDIA container runtime, CUDA-enabled base images |
| **4** | Serve via NVIDIA Triton Inference Server | ONNX export, dynamic batching, gRPC + HTTP |
| **5A** | Optimize Triton/ONNX Runtime | Found & fixed 1000× serialization bottleneck; profiled model |
| **5B** | TensorRT Execution Provider (FP16) | 6.6× faster GPU compute vs ONNX CUDA EP |
| **6A** | 3-way backend comparison (A100 + RTX 4080) | PyTorch wins client-side; Triton wins server-side |
| **6B** | Multi-GPU scaling study (4× RTX 4080) | Only 1.8× scaling (45% efficiency) — network-bound |
| **7** | 5-way gRPC vs HTTP comparison (A100 + RTX 4090) | gRPC slower at batch=1; HTTP wins under concurrency; PyTorch still leads |

---

## Top 5 Findings

### 1. Serialization Overhead Can Be the Real Bottleneck
JSON `.tolist()` serialization of image tensors was **1,000–4,800× slower** than binary encoding, adding 1.8 seconds per batch-32 request. The "slow Triton" was actually slow JSON.

### 2. Backend Performance Depends on the GPU Architecture
- **Consumer GPUs** (RTX 4080): TensorRT EP wins — **2.0ms** GPU compute
- **Datacenter GPUs** (A100): ONNX CUDA EP wins — **4.4ms** GPU compute
- Same model, same code, opposite winners depending on hardware.

### 3. Network Overhead Dominates Remote Inference
Triton achieves 4.4ms GPU compute but 183ms client-side latency. The 602KB tensor payload transfer takes 40× longer than the actual inference. PyTorch wins remotely by accepting 10KB JPEG images instead.

### 4. Multi-GPU Scaling Is Not Free
4× GPUs delivered only 1.8× throughput (45% efficiency) at 2.2× the cost per image. The network is the bottleneck, not the GPU — adding more GPUs doesn't help.

### 5. Protocol Design Matters More Than Compute Optimization
PyTorch (56.9ms) beats Triton (182.9ms) for remote clients despite Triton being 12.8× faster at GPU compute — because PyTorch accepts efficient JPEG images while Triton requires raw float tensors.

---

## Architecture at a Glance

```
┌─────────────────────────┐           ┌──────────────────────────┐
│   Client / UI App       │           │   Inference Service      │
│   (macOS, Port 8000)    │──HTTP───▶ │   (GPU, Port 8002)       │
│                         │           │                          │
│ • Scan Apple Photos     │           │ Backends explored:       │
│ • Request embeddings    │           │ ├─ PyTorch + FastAPI     │
│ • Store & group results │           │ ├─ Triton (ONNX CUDA EP) │
│ • Web UI for review     │           │ └─ Triton (TensorRT EP)  │
└─────────────────────────┘           └──────────────────────────┘
```

---

## Key Numbers

| Metric | PyTorch | Triton ONNX | TensorRT EP |
|--------|:-------:|:-----------:|:-----------:|
| **Server GPU compute** | ~10–15ms (est.) | **4.4ms** ⚡ | 2.0–29.1ms* |
| **Client latency (remote)** | **56.9ms** ⚡ | 182.9ms | 212.5ms |
| **Batch-32 throughput** | **64.3 img/s** | 32.3 img/s | N/A (timeout) |
| **VRAM usage** | ~1 GB | ~0.9 GB | ~1 GB |
| **Model size** | 335MB (.pt) | 335MB (.onnx) | 335MB (.onnx + FP16) |

*TRT EP performance varies by GPU: 2.0ms on RTX 4080, 29.1ms on A100.*

### Step 7 — gRPC vs HTTP, batch=1 p50 (A100, Massachusetts)

| Backend | Latency | vs PyTorch |
|---------|--------:|-----------:|
| PyTorch HTTP (JPEG) | **64.2ms** | — |
| Triton ONNX HTTP | 208.7ms | 3.2× |
| Triton ONNX gRPC | 217.5ms | 3.4× |
| Triton TRT HTTP | 171.4ms | 2.7× |
| Triton TRT gRPC | 200.2ms | 3.1× |

*gRPC was slower than HTTP at batch=1 on both A100 and RTX 4090. PyTorch still wins by 2.7–3.4×.*

---

## Skills & Technologies Demonstrated

- **ML Serving:** PyTorch, ONNX Runtime, TensorRT, NVIDIA Triton Inference Server
- **Containerization:** Docker multi-stage builds, NVIDIA Container Toolkit, Docker Compose
- **Cloud GPU:** Vast.ai deployment, CUDA runtime configuration, multi-GPU orchestration
- **Benchmarking:** Client vs server-side metrics, Prometheus, controlled A/B comparisons
- **Performance Analysis:** Profiling, bottleneck identification, serialization optimization
- **API Design:** FastAPI, gRPC (tritonclient), binary protocols, stateless service architecture

---

*For the full technical walkthrough, see [02_TECHNICAL_DEEP_DIVE.md](02_TECHNICAL_DEEP_DIVE.md).*  
*For architecture diagrams, see [03_ARCHITECTURE.md](03_ARCHITECTURE.md).*  
*For all benchmark data, see [04_BENCHMARK_RESULTS.md](04_BENCHMARK_RESULTS.md).*
