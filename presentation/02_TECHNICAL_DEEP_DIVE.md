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
10. [Key Learnings & Takeaways](#10-key-learnings--takeaways)

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

| Provider | GPU | VRAM | Outcome |
|----------|-----|:----:|---------|
| Vast.ai | RTX A4000 | 16 GB | ✅ Primary test platform |
| Vast.ai | A100 SXM4 | 80 GB | ✅ Step 6A comparison |
| Vast.ai | 4× RTX 4080 | 4×16 GB | ✅ Step 6B multi-GPU |

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

**All running simultaneously on a single Docker container** (`Dockerfile.step6a-all`), tested on both A100 SXM4 80GB and RTX 4080 16GB.

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

- **Instance**: 4× NVIDIA RTX 4080 (16GB each) on Vast.ai
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

| Config | Cost/hr | Throughput | Cost per 1K images |
|--------|:-------:|:----------:|:------------------:|
| 1× RTX 4080 | $2.00 | 24.3 img/s | $0.023 |
| 4× RTX 4080 | $8.00 | 43.2 img/s | $0.051 |
| 1× A100 | $1.50 | 64.3 img/s | $0.006 |

4× RTX 4080 is **2.2× more expensive per image** than 1× RTX 4080. The A100 is the best value at **3.5× more cost-efficient**.

### What I Learned

- **Multi-GPU doesn't solve network-bound workloads.** If 97% of time is in network transfer, adding GPUs only affects the other 3%.
- **Horizontal scaling** (multiple single-GPU instances behind a load balancer) would be more effective than vertical scaling (multiple GPUs on one machine).
- **Dynamic batching efficiency** drops at high concurrency: only 22% of requests were actually batched (560 executions for 721 requests).
- **The saturation point** is clear at concurrency 32 — beyond that, throughput drops and latency climbs.

---

## 10. Key Learnings & Takeaways

### Infrastructure Lessons

1. **Start with separation of concerns.** Splitting the app into client and service (Step 1) made everything else possible — containerization, GPU deployment, backend swapping.

2. **Measure, then optimize.** The Step 4 "Triton is slow" narrative was wrong. Profiling revealed the real bottleneck (JSON serialization), which was a 1,000× improvement opportunity.

3. **Protocol design is an architecture decision.** The choice of data format (JPEG vs raw tensors) determines performance more than the inference backend.

4. **The same model can perform very differently** depending on the serving framework, execution provider, and GPU architecture.

5. **Multi-GPU scaling is not automatic.** Network overhead, interconnect bandwidth, and dynamic batching efficiency all create sub-linear scaling.

### Technical Skills Developed

| Area | Specific Skills |
|------|----------------|
| **ML Serving** | ONNX export, Triton config, TensorRT EP, dynamic batching tuning |
| **Docker** | Multi-stage builds, cross-platform (ARM→x86), NVIDIA runtime, health checks |
| **GPU Deployment** | Vast.ai provisioning, CUDA configuration, multi-GPU orchestration |
| **Benchmarking** | Controlled experiments, server-side vs client-side metrics, Prometheus |
| **Performance Engineering** | Profiling, bottleneck identification, serialization optimization |
| **API Design** | Stateless services, binary protocols, graceful degradation (auto mode) |

### The Journey in Numbers

| Metric | Start (Step 1) | End (Step 6) |
|--------|:-:|:-:|
| Backends supported | 1 (local Python) | 4 (local, PyTorch API, Triton ONNX, Triton TRT) |
| Deployment targets | macOS only | macOS + any cloud GPU |
| Fastest GPU compute | N/A | **2.0ms** (TRT EP on RTX 4080) |
| Largest deployment | 1 process | 4× GPU, 3 simultaneous backends |
| Documentation pages | 1 (README) | 12+ detailed technical docs |
| Benchmark data points | 0 | 1000+ across 6 steps |

---

*For architecture diagrams, see [03_ARCHITECTURE.md](03_ARCHITECTURE.md).*  
*For all benchmark data, see [04_BENCHMARK_RESULTS.md](04_BENCHMARK_RESULTS.md).*  
*For the slide deck, open [slides.html](slides.html) in a browser.*
