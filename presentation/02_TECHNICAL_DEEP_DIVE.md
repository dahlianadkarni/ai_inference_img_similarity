# ML Inference Infrastructure: Technical Deep-Dive

> A hands-on exploration of production ML inference patterns, from a monolithic Python app
> to containerized multi-backend GPU inference with NVIDIA Triton.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Step 1: Client–Service Separation](#2-step-1-clientservice-separation)
3. [Step 2: Containerization](#3-step-2-containerization)
4. [Step 3: Cloud GPU Deployment](#4-step-3-cloud-gpu-deployment)
5. [Step 4: NVIDIA Triton Inference Server](#5-step-4-nvidia-triton-inference-server)
6. [Step 5A: ONNX Runtime Optimization](#6-step-5a-onnx-runtime-optimization)
7. [Step 5B: TensorRT Optimization](#7-step-5b-tensorrt-optimization)
8. [Step 6A: 3-Way Backend Comparison](#8-step-6a-3-way-backend-comparison)
9. [Step 6B: Multi-GPU Scaling Study](#9-step-6b-multi-gpu-scaling-study)
10. [Step 7: gRPC vs HTTP Protocol Comparison](#10-step-7-grpc-vs-http-protocol-comparison)
11. [Step 8: Local Kubernetes (kind)](#11-step-8-local-kubernetes-kind)
12. [Key Learnings & Takeaways](#12-key-learnings--takeaways)

---

## 1. Project Context

### The Application

A macOS desktop app that detects near-duplicate photos in Apple Photos using AI:

| Pipeline Stage | What It Does | Bottleneck |
|:---:|---|:---:|
| **Scan** | Export photos via AppleScript, compute MD5 + perceptual hashes | I/O (1–5 min) |
| **Embed** | Generate OpenCLIP ViT-B/32 embeddings for each image | **GPU** (10–30s) |
| **Group** | Cosine similarity clustering with configurable threshold | CPU (<1s) |
| **Review** | Web UI for side-by-side comparison, feedback, and deletion | UI (instant) |

The embedding stage is the compute bottleneck — and the focus of this entire infrastructure exploration.

### The Model

- **OpenCLIP ViT-B/32**: Vision transformer trained on LAION-2B
- **Input**: 224×224×3 RGB image (preprocessed)
- **Output**: 512-dimensional float32 embedding vector
- **Model size**: ~335 MB (ONNX), ~600 MB (PyTorch with dependencies)
- **Key ops**: 75.7% MatMul, 7.1% Gemm, 3.9% BiasGelu (2,272 total ONNX nodes)

### Why This Model?

CLIP/OpenCLIP is widely used in production for:
- Image search and retrieval
- Content moderation
- Recommendation systems
- Multimodal applications

It's small enough to iterate quickly, but complex enough (a full ViT transformer) to exercise real inference infrastructure.

---

## 2. Step 1: Client–Service Separation

### The Problem

The original app loaded the model inline — embedding generation was tightly coupled to the UI:

```
┌──────────────────────────────────┐
│        Monolithic App            │
│                                  │
│  Scan → Load Model → Embed →    │
│  Group → Display UI             │
│                                  │
│  (Everything in one process)     │
└──────────────────────────────────┘
```

### The Solution

Split into two independent services communicating over HTTP:

```
┌──────────────────────┐         ┌──────────────────────┐
│   Client / UI        │         │  Inference Service    │
│   (Port 8000)        │──HTTP──▶│  (Port 8002)         │
│                      │         │                      │
│ • Scan photos        │         │ • Load model once    │
│ • Request embeddings │         │ • POST /embed        │
│ • Store results      │         │ • GET /healthz       │
│ • Group & display    │         │ • Stateless          │
└──────────────────────┘         └──────────────────────┘
```

### Key Design Decisions

1. **Stateless inference service**: Each `/embed` request is independent. No session state, no accumulated context. This enables horizontal scaling and independent deployment.

2. **Three embedding modes** for flexibility:
   - `local` — model loads in-process (original behavior, for development)
   - `remote` — calls inference service over HTTP (production pattern)
   - `auto` — tries remote, falls back to local (graceful degradation)

3. **Clean API boundary**: The inference service accepts images and returns embeddings. It doesn't know about photos, duplicates, or the UI. This separation mirrors production patterns (Triton, TorchServe, vLLM all follow this pattern).

### What I Learned

- The split itself is straightforward, but designing the API interface required thinking about what crosses the boundary (raw images? preprocessed tensors? base64 encoded data?). This decision has massive performance implications — as I discovered later in Steps 5–6.
- The "auto" mode pattern is valuable in practice — local development doesn't require running a separate service.

---

## 3. Step 2: Containerization

### What I Did

Built a Docker image for the inference service:

```dockerfile
# Multi-stage build approach
FROM python:3.11-slim

# Install dependencies
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy application code
COPY src/inference_service/ ./src/inference_service/

# Health check built in
HEALTHCHECK CMD curl -f http://localhost:8002/healthz || exit 1

EXPOSE 8002
CMD ["python", "-m", "src.inference_service"]
```

### Key Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Base image | `python:3.11-slim` | Small, well-maintained, sufficient for CPU testing |
| Dependencies | Separate `requirements-ml.txt` | Only ML dependencies in container, not UI/scanner |
| Health check | Built-in `HEALTHCHECK` | Docker monitors service health automatically |
| User | Non-root | Security best practice |
| Model loading | On startup (not per-request) | Avoid 674ms cold-start on each request |

### What I Learned

- Separating `requirements-ml.txt` from `requirements.txt` keeps the container focused. The inference service doesn't need AppleScript bindings or UI dependencies.
- Docker health checks are important — without them, orchestrators (Docker Compose, Kubernetes) can't detect a failed model load.

---

## 4. Step 3: Cloud GPU Deployment

### What I Did

Deployed the same container to cloud GPUs using NVIDIA Container Toolkit:

```dockerfile
# GPU-enabled image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# NVIDIA runtime handles GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Deployment Targets

Tested on multiple GPU providers and hardware:

| Provider | GPU | VRAM | Location | Cost/hr | Outcome |
|----------|----------|:----:|----------|:-------:|-------------------------------|
| Vast.ai  | RTX A4000      | 16 GB  | QC, Canada   | $0.092      | ✅ Primary test platform (Steps 4–5B) |
| Vast.ai  | A100 SXM4      | 80 GB  | MA, US       | $0.85       | ✅ Step 6A comparison |
| Vast.ai  | RTX 4080       | 16 GB  | PA, US       | $0.092      | ✅ Step 6A comparison |
| Vast.ai  | 4× RTX 4080    | 4×16 GB| PA, US       | $0.30–2.00  | ✅ Step 6B multi-GPU |

### Deployment Workflow

```bash
# Build for linux/amd64 (required for cloud GPUs)
docker buildx build --platform linux/amd64 -f Dockerfile.gpu \
  -t dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64 .

# Push to Docker Hub
docker push dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64

# On Vast.ai: just specify the image, NVIDIA runtime is automatic
```

### What I Learned

- `docker buildx` with `--platform linux/amd64` is essential when building on Apple Silicon (M1/M2) for Linux GPU deployment.
- Vast.ai provides the simplest GPU access for experimentation — specify a Docker image, get a GPU instance in minutes.
- The NVIDIA Container Toolkit is transparent: the same `torch.cuda.is_available()` code works locally (CPU) and on GPU without changes.

---

## 5. Step 4: NVIDIA Triton Inference Server

### Why Triton?

Triton is NVIDIA's production inference server, used by major ML platforms. It provides features that a simple FastAPI server doesn't:

| Feature | FastAPI (Step 3) | Triton (Step 4) |
|---------|:---:|:---:|
| Dynamic batching | ❌ (manual) | ✅ (automatic) |
| Multi-model serving | ❌ | ✅ |
| Model versioning | ❌ | ✅ |
| gRPC support | ❌ | ✅ |
| Prometheus metrics | ❌ | ✅ |
| Multiple backend support | ❌ | ✅ (ONNX, TensorRT, PyTorch, TensorFlow) |

### What I Did

1. **Exported model to ONNX** using `scripts/export_to_onnx.py`
2. **Created Triton model repository** structure:
   ```
   model_repository/
   └── openclip_vit_b32/
       ├── config.pbtxt       # Model config (inputs, outputs, batching)
       └── 1/
           └── model.onnx     # Exported ONNX model (335 MB)
   ```
3. **Configured dynamic batching** with queue delay:
   ```
   dynamic_batching {
     max_queue_delay_microseconds: 10000
     preferred_batch_size: [4, 8, 16, 32]
   }
   ```
4. **Built Docker image** based on `nvcr.io/nvidia/tritonserver:24.01-py3`
5. **Updated client** to support both PyTorch and Triton backends

### Dynamic Batching Deep-Dive

Dynamic batching is Triton's killer feature. Instead of processing one request at a time:

```
Without batching:        With dynamic batching:
Request 1 → GPU → Done  Request 1 ─┐
Request 2 → GPU → Done  Request 2 ─┼─→ GPU (batch of 3) → Done
Request 3 → GPU → Done  Request 3 ─┘
Total: 3× GPU calls     Total: 1× GPU call (much faster)
```

The `max_queue_delay_microseconds` parameter controls how long Triton waits to collect more requests before dispatching a batch. Higher = better throughput, worse latency for individual requests.

### Initial Benchmark Results (Step 4)

The first Triton benchmarks were **disappointing**:

| Metric | PyTorch | Triton | |
|--------|:-------:|:------:|---|
| Batch-32 latency | 1.7s | **9.6s** | 5.6× slower! |
| Single-image p50 | 168ms | **564ms** | 3.4× slower! |

This led to the investigation in Step 5A...

### What I Learned

- ONNX export is mostly straightforward for standard architectures (ViT), but custom ops or dynamic shapes can cause issues.
- Triton's model repository structure is rigid but well-designed — it enforces versioning and clean configuration.
- The initial "Triton is slow" result was misleading — the bottleneck was in how we sent data, not in Triton itself.

---

## 6. Step 5A: ONNX Runtime Optimization

### The Investigation

Step 4's poor results triggered a deep investigation. I profiled the entire pipeline and found a **critical serialization bottleneck**.

### Root Cause: JSON Serialization

The benchmark was sending image tensors to Triton as JSON using `.tolist()`, which converts every float32 to a decimal string:

| Batch Size | JSON `.tolist()` Time | Binary Encoding Time | JSON Is Slower By |
|:---:|:---:|:---:|:---:|
| 1 | 52.5ms | 0.01ms | **4,799×** |
| 32 | 1,842ms | 1.74ms | **1,058×** |

For a batch of 32 images (3×224×224), this serializes **4.8 million floats** as text — producing a 95MB JSON payload vs 18MB binary.

### The Fix

Switched to Triton's native binary tensor protocol via `tritonclient[http]`:

```python
# Before (JSON — 1,842ms for batch-32):
data = {"inputs": [{"data": tensor.tolist()}]}
requests.post(url, json=data)

# After (binary — 1.74ms for batch-32):
input_tensor = tritonclient.http.InferInput("input", shape, "FP32")
input_tensor.set_data_from_numpy(tensor)
client.infer(model_name, inputs=[input_tensor])
```

### ONNX Model Profiling

I also profiled the ONNX model to understand its internal performance:

| Operator | % of Compute | Notes |
|----------|:---:|---|
| MatMul | 75.7% | Dominates — expected for transformers |
| Gemm | 7.1% | Dense layers |
| BiasGelu | 3.9% | Activation function |
| Conv | 3.7% | Patch embedding (first layer) |
| Everything else | 9.6% | Transpose, LayerNorm, Add, etc. |

**No CPU fallbacks detected** — all 2,272 ops run natively in ONNX Runtime on GPU.

### Results After Fix

| Metric | Before (JSON) | After (Binary) | Improvement |
|--------|:---:|:---:|:---:|
| Batch-32 latency | 9,649ms | 5,614ms | **1.7× faster** |
| Concurrent throughput | 8.7 req/s | 35.8 req/s | **4.1× faster** |
| Single p50 | 564ms | 229ms | **2.5× faster** |

### Remaining Gap

Even after the fix, ONNX Runtime was **5.2× slower** than native PyTorch for batch-32. This isn't a configuration issue — it's inherent to ONNX Runtime's GPU execution vs PyTorch's optimized cuBLAS/cuDNN kernels.

### What I Learned

- **Always measure before optimizing.** The "Triton is slow" narrative was wrong — the real bottleneck was client-side JSON serialization.
- Binary protocols matter enormously when transferring tensor data. A 1,000× difference in serialization time dominates everything else.
- ONNX Runtime's GPU performance is decent but not competitive with native PyTorch for MatMul-heavy workloads. The 5.2× gap is significant.

---

## 7. Step 5B: TensorRT Optimization

### Goal

Close the 5.2× gap between ONNX Runtime and native PyTorch using TensorRT's kernel fusion and FP16 optimization.

### Architecture Pivot

The original plan was standalone `.plan` file conversion via `trtexec`. This failed because:
1. `trtexec` isn't included in the Triton container image
2. `pip install tensorrt` (v10.15) was incompatible with Triton 24.01's CUDA 12.3

**Solution:** Use ONNX Runtime's **TensorRT Execution Provider**, which leverages TRT 8.6 already bundled in the Triton container. Same ONNX model file, different execution provider.

```
model_repository/
├── openclip_vit_b32/           # ONNX CUDA EP (Step 4)
│   ├── config.pbtxt
│   └── 1/model.onnx
│
└── openclip_vit_b32_trt/       # TensorRT EP (Step 5B)
    ├── config.pbtxt            # TRT EP config with FP16
    └── 1/model.onnx → symlink  # Same model, different EP
```

### Why TensorRT Is Faster

1. **Kernel Fusion**: Combines sequences of ops (LayerNorm + MatMul + GELU → single kernel), eliminating intermediate memory reads/writes
2. **FP16 Precision**: Halves memory bandwidth requirements (safe for embeddings — cosine similarity >0.999 vs FP32)
3. **Architecture-Specific Tuning**: Auto-tunes kernel implementations for the specific GPU architecture
4. **Memory Layout Optimization**: Reorders tensor memory for optimal GPU cache utilization

### Results (RTX A4000)

| Metric | ONNX CUDA EP | TensorRT EP | Speedup |
|--------|:---:|:---:|:---:|
| **GPU compute** | 31.1ms | **4.7ms** | **6.6× faster** |
| Server request | 41.9ms | 15.9ms | 2.6× faster |
| Input processing | 1.2ms | 0.5ms | 2.4× faster |

**TRT EP's 4.7ms compute likely matches or exceeds native PyTorch** (~10–15ms estimated), closing the 5.2× gap from Step 5A.

### Trade-offs

| Pro | Con |
|-----|-----|
| 6.6× faster GPU compute | First startup: 2–5 min (engine compilation) |
| FP16 precision (safe for embeddings) | Each new batch shape triggers recompilation |
| No separate conversion step | TRT EP behavior varies by GPU architecture |
| Same ONNX model file | Less portable than pure ONNX |

### What I Learned

- TensorRT's kernel fusion provides dramatic speedups for transformer models. The 6.6× improvement is real and comes from eliminating memory round-trips between ops.
- FP16 is safe for embedding models — the precision loss is negligible for cosine similarity.
- The TRT EP approach (via ONNX Runtime) is more practical than standalone `.plan` conversion — it avoids version compatibility headaches.
- First-load latency (2–5 min) is acceptable for long-running servers but problematic for serverless/autoscaling.

---

## 8. Step 6A: 3-Way Backend Comparison

### Setup

Deployed all three backends on the same GPU for fair comparison:

| Backend | How | Port |
|---------|-----|:---:|
| PyTorch + FastAPI | Direct model loading | 8002 |
| Triton ONNX CUDA EP | ONNX Runtime | 8003 |
| Triton TensorRT EP | ONNX RT + TRT EP | 8003 |

**All running simultaneously on a single Docker container** (`Dockerfile.step6a-all`), tested on both A100 SXM4 80GB (Boston, MA) and RTX 4080 16GB (Lititz, PA). Client ran on macOS in MA — network RTT varied by instance location (~20–30ms).

### Results: A100 SXM4 80GB

| Metric | PyTorch | Triton ONNX | Triton TRT |
|--------|:-------:|:-----------:|:----------:|
| **Client latency (single)** | **56.9ms** ⚡ | 182.9ms | 212.5ms |
| **Server GPU compute** | ~10–15ms (est.) | **4.4ms** ⚡ | 29.1ms |
| **Batch-32 throughput** | **64.3 img/s** | 32.3 img/s | N/A |

### Results: RTX 4080 16GB

| Metric | PyTorch | Triton ONNX | Triton TRT |
|--------|:-------:|:-----------:|:----------:|
| **Client latency (single)** | 219ms | 337ms | 337ms |
| **Server GPU compute** | ~10–15ms (est.) | 5.7ms | **2.0ms** ⚡ |
| **Batch-32 throughput** | 22.7 img/s | **24.3 img/s** ⚡ | 18.6 img/s |

### The Key Insight: Where Does the Time Go?

For Triton ONNX on A100 (182.9ms client-side latency):
- GPU compute: **4.4ms** (2.4% of total!)
- Server overhead: 10.7ms (5.8%)
- **Network + tensor transfer: 167.8ms (91.7%)**

The 602KB raw float tensor payload takes ~150–170ms to transfer over the internet. PyTorch sends ~10KB base64 JPEG images instead, which is why it wins at the client level.

### GPU Architecture Matters

A fascinating finding: the **best backend depends on the GPU**:

| GPU | Best Backend | GPU Compute |
|-----|:---:|:---:|
| A100 SXM4 (datacenter) | ONNX CUDA EP | 4.4ms |
| RTX 4080 (consumer) | TensorRT EP | 2.0ms |

The A100's Tensor Cores are highly optimized for generic matrix operations (ONNX benefits), while TensorRT's kernel fusion provides larger gains on consumer GPU architectures.

### VRAM Usage

All three backends combined used only **3.3 GB** on the A100 (4.1% of 80GB) and **2.8 GB** on the RTX 4080 (17.4% of 16GB). This model is tiny by modern standards.

### What I Learned

- **Protocol design matters more than compute optimization.** PyTorch wins remotely by accepting efficient images, despite Triton being 12.8× faster at actual inference.
- **Server-side metrics are essential** for understanding where time is spent. Client-side latency alone is misleading.
- **Hardware-aware deployment** is real: the best backend choice depends on the GPU you're running on.

---

## 9. Step 6B: Multi-GPU Scaling Study

### Setup

- **Instance**: 4× NVIDIA RTX 4080 (16GB each) on Vast.ai (PA — same machine as 1× RTX 4080 benchmark)
- **Interconnect**: PCIe Gen4
- **Config**: Triton with `instance_group` distributing across all 4 GPUs
- **Test**: Varied concurrency from 1 to 128 concurrent requests

### Throughput vs Concurrency

| Concurrency | Throughput | Latency (p50) | Notes |
|:-----------:|:----------:|:-------------:|-------|
| 1 | 3.4 img/s | 268ms | Single request |
| 4 | 11.0 img/s | 270ms | Linear scaling |
| 8 | 20.2 img/s | 255ms | Still scaling |
| 16 | 31.9 img/s | 303ms | Good |
| **32** | **43.2 img/s** | 416ms | **Peak** ⭐ |
| 64 | 37.4 img/s | 1,000ms | Saturated |
| 128 | 37.1 img/s | 1,638ms | Over-saturated |

### Scaling Efficiency

| Config | Throughput | Scaling | Efficiency |
|--------|:---------:|:-------:|:----------:|
| 1× RTX 4080 | 24.3 img/s | 1.0× | baseline |
| 4× RTX 4080 | 43.2 img/s | 1.8× | **45%** |
| Expected (ideal) | ~97 img/s | 4.0× | 100% |

### Why Only 45% Efficiency?

**Server-side timing breakdown** (from Triton Prometheus metrics):

```
Avg request duration:    17.9ms
├── Queue time:          8.8ms   (49% — waiting for batch formation)
├── GPU compute:         8.4ms   (47% — actual inference)
└── Input processing:    0.7ms   (4% — tensor copy)
```

But the **client-side** latency is 416ms at peak — meaning **95% of time is network overhead**, not server-side processing.

### Cost Analysis

| Config | Cost/hr (actual) | Throughput | Cost per 1K images |
|--------|:-------:|:----------:|:------------------:|
| 1× RTX 4080 | $0.092 | 24.3 img/s | **$0.001** ⚡ |
| 1× A100 | $0.85 | 64.3 img/s | $0.004 |
| 4× RTX 4080 | $0.30–2.00 | 43.2 img/s | $0.002–0.013 |

1× RTX 4080 at $0.092/hr is **3.5× more cost-efficient** than the A100 ($0.85/hr). Even the 4× RTX 4080 at best spot pricing ($0.30/hr) is **2× more expensive per image** than a single GPU. Cheapest hardware with adequate throughput wins on cost.

### What I Learned

- **Multi-GPU doesn't solve network-bound workloads.** If 97% of time is in network transfer, adding GPUs only affects the other 3%.
- **Horizontal scaling** (multiple single-GPU instances behind a load balancer) would be more effective than vertical scaling (multiple GPUs on one machine).
- **Dynamic batching efficiency** drops at high concurrency: only 22% of requests were actually batched (560 executions for 721 requests).
- **The saturation point** is clear at concurrency 32 — beyond that, throughput drops and latency climbs.

---

## 10. Step 7: gRPC vs HTTP Protocol Comparison

### Motivation

Steps 6A/6B established that 97% of client-side latency is transport overhead (602KB float tensor transfer). The natural next question: does gRPC (HTTP/2 + protobuf framing + persistent connections) cut into that 178ms overhead?

### Setup

5-way comparison on the same GPU instance — all backends tested back-to-back:

| # | Backend | Protocol | Payload/image |
|---|---------|----------|---------------|
| 1 | PyTorch FastAPI | HTTP/1.1 | ~10 KB (JPEG) |
| 2 | Triton ONNX EP | HTTP/1.1 binary | ~602 KB (FP32) |
| 3 | Triton ONNX EP | gRPC (HTTP/2) | ~602 KB (FP32) |
| 4 | Triton TRT EP | HTTP/1.1 binary | ~602 KB (FP32) |
| 5 | Triton TRT EP | gRPC (HTTP/2) | ~602 KB (FP32) |

Run on two instances: **A100 SXM4 80GB** (Massachusetts, $0.894/hr) and **RTX 4090** (Pennsylvania, $0.391/hr).

> **Two ways to measure throughput (used throughout this section):**
>
> - **Batch-32 throughput** — *sequential* requests, one at a time, each sending 32 images in a single payload. Tests raw GPU utilization: can batching amortize the 602KB-per-image transfer cost? Higher = better bulk/offline processing.
> - **conc=16 img/s** — 16 *simultaneous* single-image requests in-flight at once. Simulates multiple concurrent clients. Tests the server's ability to pipeline and queue requests. Higher = better real-time multi-user serving.
>
> They answer different questions and can point in opposite directions (as gRPC demonstrates below).

### Results: A100 SXM4 (Massachusetts)

**Latency, GPU compute, and batch throughput:**

| Metric | PyTorch | ONNX HTTP | ONNX gRPC | TRT HTTP | TRT gRPC |
|--------|:-------:|:---------:|:---------:|:--------:|:--------:|
| **Client latency (single)** | **64.2ms** ⚡ | 208.7ms | 217.5ms | 171.4ms | 200.2ms |
| **Server GPU compute** | ~10–15ms (est.) | 8.67ms | **4.04ms** ⚡ | 1657ms† | **3.63ms** ⚡ |
| **Batch-32 throughput** | **48.5 img/s** | 6.08 img/s | 10.23 img/s | 12.12 img/s | 14.39 img/s |

†TRT HTTP GPU compute includes TRT engine compilation in this run; post-compile true latency ≈ 3.6ms (see TRT gRPC server metric).

**Concurrent throughput (batch=1 per request):**

| Metric | PyTorch | ONNX HTTP | ONNX gRPC | TRT HTTP | TRT gRPC |
|--------|:-------:|:---------:|:---------:|:--------:|:--------:|
| **conc=8 img/s** | 24.0 | 28.8 | 16.6 | **32.4** ⚡ | 14.3 |
| **conc=16 img/s** | 30.5 | **43.6** ⚡ | 7.5 | 22.4 | 15.4 |

### Three Surprising Findings

**1. gRPC is slower at batch=1 — not faster.** Both ONNX and TRT gRPC were 4–14% slower than their HTTP counterparts at single-image latency. Persistent connections don’t help when 602KB bandwidth is the bottleneck, not TCP handshake cost.

**2. gRPC wins for large batches (batch≥4).** At batch=32, ONNX gRPC was 1.68× faster than ONNX HTTP. HTTP/2 header compression and framing efficiency pay off when payloads are multi-megabyte.

**3. HTTP scales better under concurrency.** ONNX HTTP hit 43.6 img/s at conc=16, while ONNX gRPC ‘degraded’ to 7.5 img/s — a 5.8× gap. The Python `tritonclient.grpc` client introduces channel contention under concurrent load. This is a client library limitation, not a fundamental HTTP/2 weakness.

### Results: RTX 4090 (Pennsylvania)

**Latency, GPU compute, and batch throughput:**

| Metric | PyTorch | ONNX HTTP | ONNX gRPC | TRT HTTP | TRT gRPC |
|--------|:-------:|:---------:|:---------:|:--------:|:--------:|
| **Client latency (single)** | **137.2ms** ⚡ | 272.1ms | 318.5ms | 269.9ms | 312.5ms |
| **Server GPU compute** | ~10–15ms (est.) | 10.63ms | **4.41ms** ⚡ | 1170ms† | **2.68ms** ⚡ |
| **Batch-32 throughput** | **39.4 img/s** | 4.75 img/s | 4.84 img/s | 3.50 img/s | 3.79 img/s |

†Same TRT engine compilation artifact as A100. True cached GPU compute ≈ 2.68ms (TRT gRPC server metric).

**Concurrent throughput (batch=1 per request):**

| Metric | PyTorch | ONNX HTTP | ONNX gRPC | TRT HTTP | TRT gRPC |
|--------|:-------:|:---------:|:---------:|:--------:|:--------:|
| **conc=8 img/s** | **47.3** ⚡ | 17.5 | 4.1 | 20.1 | 4.2 |
| **conc=16 img/s** | **49.1** ⚡ | 32.2 | 4.0 | 21.8 | 6.1 |

**Cross-GPU comparison (b=1 serial p50):**

| GPU | PyTorch | ONNX HTTP | ONNX gRPC | TRT HTTP | TRT gRPC |
|-----|:-------:|:---------:|:---------:|:--------:|:--------:|
| A100 SXM4 (MA) | **64.2ms** ⚡ | 208.7ms | 217.5ms | 171.4ms | 200.2ms |
| RTX 4090 (PA) | **137.2ms** ⚡ | 272.1ms | 318.5ms | 269.9ms | 312.5ms |
| RTX 4090 vs A100 | 2.1× | 1.3× | 1.5× | 1.6× | 1.6× |

PyTorch latency scales with GPU speed (2.1× gap). Triton HTTP baselines are only 1.3–1.6× worse — confirming they remain **transport-bound** (602KB bandwidth) regardless of GPU tier. Cost-efficiency: RTX 4090 at $0.391/hr is 2.3× cheaper with 1.3–2.1× worse serial latency, but actually *faster* under concurrency (49.1 vs 30.5 img/s at conc=16).



### What I Learned

- **Bandwidth, not connection overhead, is the bottleneck.** Even with persistent HTTP/2 channels, gRPC cannot escape the physics of transferring 602KB per image. The only fix is changing the input format.
- **Protocol matters more for batch workloads.** gRPC is a good choice for batch≥4 serial inference; HTTP is better for concurrent single-image requests with the current tritonclient.
- **Consumer vs datacenter GPU trade-off is consistent.** Both A100 and RTX 4090 show the same ordering: PyTorch wins, then TRT HTTP, then ONNX HTTP. The ratios scale predictably with GPU tier.

---

---

## 11. Step 8: Local Kubernetes (kind)

**Goal:** Understand Kubernetes primitives (HPA, PDB, ResourceQuota) using a local kind cluster — no cloud cost, CPU-only, learning-focused.

### Architecture

```
macOS client
     │
     ▼  localhost:8092
kind cluster (single node — Docker container)
     │
     ▼  NodePort 30092
K8s Service
     │
     ├── Pod 1: photo-duplicate-inference:k8s-cpu  (500m–2000m CPU)
     ├── Pod 2: photo-duplicate-inference:k8s-cpu  (500m–2000m CPU)
     └── ... HPA scales to Pod 3–6 under load
```

### Stack

| Component | Version | Notes |
|-----------|---------|-------|
| kind | v0.31.0 | Kubernetes-in-Docker |
| Kubernetes | v1.35.0 | cluster version |
| kubectl | v1.34.1 | + bundled kustomize |
| metrics-server | latest | patched `--kubelet-insecure-tls` for kind |
| Image | `photo-duplicate-inference:k8s-cpu` (388 MB) | CPU-only, ARM64 |

### Manifests (`k8s/`)

| File | Purpose |
|------|---------|
| `namespace.yaml` | namespace: inference |
| `configmap.yaml` | MODEL_NAME, HOST, PORT env vars |
| `deployment.yaml` | 2 replicas, Never pull, probes, 500m–2000m CPU |
| `service.yaml` | NodePort 30092 |
| `hpa.yaml` | min=2, max=6, target=60% CPU, scaleDown stabilization=180s |
| `pdb.yaml` | minAvailable=1 |
| `resourcequota.yaml` | pods:10, requests.cpu:4, limits.cpu:8 |
| `kustomization.yaml` | `kubectl apply -k k8s/` |

### Key Observations

**HPA triggered scale-up in two steps under 600-request load:**
```
SuccessfulRescale: New size: 4; reason: cpu resource utilization above target
SuccessfulRescale: New size: 6; reason: cpu resource utilization above target
```

**CPU is expected to be slow** — ViT-B/32 inference takes ~400–1000ms per image on Apple Silicon without GPU acceleration:

| Metric | CPU (kind) | GPU A100 (Step 6A) | Ratio |
|--------|:-:|:-:|:-:|
| p50 latency | 1,020ms | 64ms | **16× slower** |
| Throughput | 3.5 req/s | ~30 req/s | **8× lower** |
| Cost | $0/hr | $0.85/hr | ♾️ cheaper |

**Port isolation prevents any conflict with existing docker-compose:**

| Service | Port | Coexists? |
|---------|:----:|:---------:|
| docker-compose PyTorch | 8002 | ✅ |
| docker-compose Triton ONNX | 8003/8004 | ✅ |
| kind K8s PyTorch | 8092 | ✅ |

### What I Learned

- **HPA responds to actual CPU pressure within ~30 seconds** on a single-node kind cluster — the scale-up from 2→4→6 replicas happened in two consecutive events 30s apart, exactly as configured.
- **Readiness probes under CPU saturation are protective.** When pods saturate their CPU limit (1988m/2000m), the readiness probe fails and the pod is removed from Service endpoints — preventing new requests from hitting an overloaded pod.
- **The 3-minute scaleDown stabilization window is essential.** Without it, HPA would oscillate; with it, the cluster holds 6 replicas through the post-load cooldown before returning to 2.
- **Kubernetes primitives are just policies.** Deployment, HPA, PDB, and ResourceQuota are all declarative YAML; the complexity is in understanding the interactions (e.g., PDB + HPA max can prevent full scaleDown).
- **kind is an excellent learning environment.** Zero cloud cost, real Kubernetes API surface, fast iteration — the only trade-off is no GPU passthrough.

---

## 12. Key Learnings & Takeaways

### Infrastructure Lessons

1. **Start with separation of concerns.** Splitting the app into client and service (Step 1) made everything else possible — containerization, GPU deployment, backend swapping.

2. **Measure, then optimize.** The Step 4 "Triton is slow" narrative was wrong. Profiling revealed the real bottleneck (JSON serialization), which was a 1,000× improvement opportunity.

3. **Protocol design is an architecture decision.** The choice of data format (JPEG vs raw tensors) determines performance more than the inference backend. Step 7 confirmed this further: switching from HTTP to gRPC doesn’t help if the payload stays at 602KB.

4. **gRPC is not a free performance upgrade.** At batch=1, gRPC is marginally *slower* than HTTP due to framing overhead with the current Python client. It helps for large batch serial workloads but hurts under high concurrency due to channel contention.

4. **The same model can perform very differently** depending on the serving framework, execution provider, and GPU architecture.

5. **Multi-GPU scaling is not automatic.** Network overhead, interconnect bandwidth, and dynamic batching efficiency all create sub-linear scaling.

### Technical Skills Developed

| Area | Specific Skills |
|------|----------------|
| **ML Serving** | ONNX export, Triton config, TensorRT EP, dynamic batching tuning |
| **Docker** | Multi-stage builds, cross-platform (ARM→x86), NVIDIA runtime, health checks |
| **GPU Deployment** | Vast.ai provisioning, CUDA configuration, multi-GPU orchestration |
| **Kubernetes** | kind cluster, Deployments, HPA, PDB, ResourceQuota, readiness/liveness probes |
| **Benchmarking** | Controlled experiments, server-side vs client-side metrics, Prometheus |
| **Performance Engineering** | Profiling, bottleneck identification, serialization optimization |
| **API Design** | Stateless services, binary protocols, graceful degradation (auto mode) |

### The Journey in Numbers

| Metric | Start (Step 1) | End (Step 8) |
|--------|:-:|:-:|
| Backends supported | 1 (local Python) | 6 (local, PyTorch API, Triton ONNX HTTP/gRPC, Triton TRT HTTP/gRPC, K8s) |
| Deployment targets | macOS only | macOS + any cloud GPU + local K8s |
| Fastest GPU compute | N/A | **2.7ms** (TRT EP gRPC on RTX 4090) |
| Largest deployment | 1 process | 4× GPU, 5 simultaneous backends + K8s (6 pods) |
| Documentation pages | 1 (README) | 16+ detailed technical docs |
| Benchmark data points | 0 | 1700+ across 8 steps |
| K8s objects managed | 0 | 7 (Deployment, Service, HPA, PDB, ResourceQuota, ConfigMap, Namespace) |

---

*For architecture diagrams, see [03_ARCHITECTURE.md](03_ARCHITECTURE.md).*  
*For all benchmark data, see [04_BENCHMARK_RESULTS.md](04_BENCHMARK_RESULTS.md).*  
*For the slide deck, open [slides.html](slides.html) in a browser.*
