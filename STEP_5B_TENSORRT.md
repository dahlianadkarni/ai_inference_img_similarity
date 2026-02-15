# Step 5B: TensorRT Optimization

> **Status**: ✅ COMPLETE — TRT EP achieves 6.6x faster GPU compute vs ONNX CUDA EP  
> **Model**: OpenCLIP ViT-B-32 (FP16 via TensorRT Execution Provider)  
> **Previous**: Step 5A found PyTorch 5.2x faster than ONNX Runtime on Triton  
> **Result**: TRT EP (4.7ms compute) likely matches or exceeds native PyTorch (~10-15ms)

---

## Table of Contents

1. [Goal](#goal)
2. [Why TensorRT?](#why-tensorrt)
3. [Architecture](#architecture)
4. [Files Created](#files-created)
5. [How It Works](#how-it-works)
6. [Deployment Guide](#deployment-guide)
7. [Benchmarking Guide](#benchmarking-guide)
8. [Expected Results](#expected-results)
9. [Troubleshooting](#troubleshooting)

---

## Goal

Convert the ONNX model to use **TensorRT Execution Provider** with FP16 precision and serve it
via Triton alongside the existing ONNX model. This enables a fair A/B comparison
on the same GPU instance:

| Backend | Model Format | Precision | Serving | Compute Time |
|---------|-------------|-----------|---------|-------------|
| PyTorch | Native .pt | FP32 | FastAPI | ~10-15ms (est.) |
| Triton/ONNX | .onnx | FP32 | Triton + ONNX Runtime CUDA EP | 31.1ms |
| **Triton/TRT EP** | **.onnx** | **FP16** | **Triton + ONNX Runtime TRT EP** | **4.7ms** |

> **Note**: We pivoted from standalone `.plan` conversion to TRT Execution Provider.
> See [Architecture Change](#architecture-change) below.

---

## Why TensorRT?

Step 5A showed ONNX Runtime on Triton was **5.2x slower** than native PyTorch.
TensorRT addresses this with:

1. **Kernel fusion**: Combines multiple ops (matmul + bias + activation) into single
   optimized CUDA kernels, eliminating memory round-trips between ops.

2. **FP16 precision**: Halves memory bandwidth requirements. ViT models tolerate FP16
   well — cosine similarity between FP32 and FP16 embeddings is typically >0.999.

3. **Architecture-specific tuning**: Auto-tunes kernel implementations for the specific
   GPU (Ampere, Ada Lovelace, etc.), unlike ONNX Runtime which uses generic kernels.

4. **Memory optimization**: Reuses tensor memory between layers, reducing VRAM footprint.

**Potential improvement**: 2-4x over ONNX Runtime, potentially matching or exceeding
native PyTorch for batch inference.

---

## Architecture

### Architecture Change

The original plan was standalone TRT `.plan` conversion via `trtexec`. This failed because:
1. `trtexec` is not included in the Triton container image
2. `pip install tensorrt` (v10.15) was incompatible with Triton 24.01's CUDA 12.3

**Solution**: Use ONNX Runtime's **TensorRT Execution Provider**, which uses the TRT 8.6 libraries already bundled in the Triton container. No separate conversion needed.

```
┌─────────────────────────────────────────────────────┐
│              Vast.ai GPU Instance                    │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │          Triton Inference Server 24.01          │ │
│  │                                                  │ │
│  │   ┌──────────────────┐ ┌──────────────────────┐ │ │
│  │   │ openclip_vit_b32 │ │ openclip_vit_b32_trt │ │ │
│  │   │  ONNX Runtime    │ │  ONNX Runtime        │ │ │
│  │   │  CUDA EP (FP32)  │ │  TRT EP (FP16)       │ │ │
│  │   │  model.onnx      │ │  model.onnx (link)   │ │ │
│  │   │  Compute: 31ms   │ │  Compute: 4.7ms      │ │ │
│  │   └──────────────────┘ └──────────────────────┘ │ │
│  │                                                  │ │
│  │   Same ONNX file, different Execution Providers   │
│  │   Ports: 8000 (HTTP), 8001 (gRPC), 8002 (metrics)│
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  First startup: TRT engine compilation (~2-5 min)     │
│  Subsequent:    Instant (engine cached in /tmp)       │
└─────────────────────────────────────────────────────┘
```

---

## Files Created

| File | Purpose |
|------|---------|
| `model_repository/openclip_vit_b32_trt/config.pbtxt` | Triton config with TRT EP accelerator (FP16) |
| `scripts/triton_trt_entrypoint.sh` | Container entrypoint: symlinks ONNX model + starts Triton |
| `Dockerfile.tensorrt` | Docker image (lightweight, no pip installs) |
| `scripts/trt_quick_test.py` | Quick A/B benchmark with server-side metrics |
| `scripts/benchmark_tensorrt.py` | Full 3-way benchmark: PyTorch vs ONNX vs TensorRT |
| `deploy_tensorrt_gpu.sh` | Build + push deployment image |
| `run_tensorrt_benchmark.sh` | Benchmark runner for remote GPU instance |
| `scripts/convert_to_tensorrt.py` | ONNX → TRT conversion (reference, not used in deployment) |

---

## How It Works

### TensorRT Execution Provider

The TRT EP approach uses ONNX Runtime's built-in TensorRT integration:

1. **Same ONNX model** — The TRT model uses a symlink to the same `.onnx` file
2. **TRT EP config** — `config.pbtxt` specifies `gpu_execution_accelerator` with TRT parameters
3. **Automatic compilation** — On first inference, ONNX Runtime delegates supported ops to TensorRT
4. **Engine caching** — Compiled TRT engines cached in `/tmp/trt_cache`
5. **FP16 precision** — Configured via `precision_mode: FP16` parameter

```
# Key config in config.pbtxt:
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [{
      name : "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }
      parameters { key: "max_workspace_size_bytes" value: "4294967296" }
      parameters { key: "trt_engine_cache_enable" value: "1" }
    }]
  }
}
```

---

## Deployment Guide

### 1. Build and Push Docker Image

```bash
# Make deploy script executable
chmod +x deploy_tensorrt_gpu.sh

# Build + push to Docker Hub
./deploy_tensorrt_gpu.sh
```

Image: `dahlianadkarni/photo-duplicate-triton:tensorrt-gpu`

### 2. Deploy on Vast.ai

**Instance Configuration:**
- **GPU**: RTX 3090, RTX 4090, A40, A10 (16GB+ VRAM recommended)
- **Docker Image**: `dahlianadkarni/photo-duplicate-triton:tensorrt-gpu`
- **Expose Ports**: 8000, 8001, 8002
- **Disk Space**: 2GB+ (for model files + TRT engine)

**Environment Variables** (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `TRT_FP16` | `1` | Enable FP16 (set 0 for FP32) |
| `TRT_MAX_BATCH` | `32` | Max batch size |
| `TRT_OPT_BATCH` | `16` | Optimal batch size for autotuning |
| `SKIP_TRT_BUILD` | `0` | Set 1 to skip TRT, use ONNX only |

### 3. Wait for Engine Build

First startup takes **2-10 minutes** extra for TRT conversion:

```bash
# Check if Triton is ready (will return 200 when both models loaded)
curl http://<instance-ip>:<port>/v2/health/ready

# Check which models are loaded
curl http://<instance-ip>:<port>/v2/models/openclip_vit_b32/ready      # ONNX
curl http://<instance-ip>:<port>/v2/models/openclip_vit_b32_trt/ready  # TRT
```

### 4. Verify Both Models

```bash
# ONNX model metadata
curl -s http://<ip>:<port>/v2/models/openclip_vit_b32 | python3 -m json.tool

# TensorRT model metadata
curl -s http://<ip>:<port>/v2/models/openclip_vit_b32_trt | python3 -m json.tool
```

---

## Benchmarking Guide

### Quick Benchmark (ONNX vs TRT only)

```bash
python scripts/benchmark_tensorrt.py \
  --triton-url http://<instance-ip>:<port> \
  --no-pytorch \
  --iterations 30
```

### Full 3-Way Benchmark (PyTorch + ONNX + TRT)

Requires a separate PyTorch instance running:

```bash
python scripts/benchmark_tensorrt.py \
  --triton-url http://<triton-ip>:<port> \
  --pytorch-url http://<pytorch-ip>:<port> \
  --iterations 50 \
  --concurrency 16 \
  --concurrent-requests 200
```

### Using the Runner Script

```bash
# Edit host/port in the script first
chmod +x run_tensorrt_benchmark.sh
./run_tensorrt_benchmark.sh
```

Results are saved to `benchmark_results/tensorrt_benchmark_<timestamp>.json`.

---

## Actual Results (RTX A4000)

Benchmarked on Vast.ai RTX A4000 (16GB VRAM), February 15, 2026.
Server-side metrics from Triton `/metrics` endpoint (no network overhead):

| Metric | ONNX Runtime (CUDA EP) | TensorRT EP (FP16) | Speedup |
|--------|----------------------|-------------------|---------|
| **GPU compute** | 31.1ms | **4.7ms** | **6.6x** |
| Server request | 41.9ms | 15.9ms | 2.6x |
| Input processing | 1.2ms | 0.5ms | 2.4x |

### Key Achievement
TRT EP's 4.7ms compute time is likely **faster than native PyTorch** (~10-15ms estimated from Step 5A), closing and surpassing the 5.2x gap found in Step 5A.

**Full results**: See [STEP_5B_GPU_RESULTS.md](STEP_5B_GPU_RESULTS.md)

**Key question answered**: ✅ TensorRT EP closes the gap with native PyTorch and likely exceeds it for GPU compute.

---

## Troubleshooting

### TRT engine build fails

```
Error: trtexec failed (exit code 1)
```

**Possible causes:**
- Insufficient VRAM during build (need ~4GB free)
- ONNX model has unsupported ops
- Try: `SKIP_TRT_BUILD=1` to fall back to ONNX-only

### TRT model won't load in Triton

```
Error: model 'openclip_vit_b32_trt' is not ready
```

**Possible causes:**
- Engine built on different GPU architecture
- Delete `model.plan` and restart container to rebuild
- Check Triton logs: `docker logs <container-id>`

### FP16 accuracy concerns

For image embeddings, FP16 is generally safe:
- Cosine similarity between FP32 and FP16 outputs: >0.999
- For exact reproducibility, set `TRT_FP16=0`

### Container takes too long to start

First startup includes TRT build (2-10 min). Subsequent starts are instant.
The health check has `start-period=600s` to accommodate this.

---

## Files Reference

```
model_repository/
├── openclip_vit_b32/           # ONNX model (Step 4)
│   ├── config.pbtxt            # ONNX Runtime CUDA EP config
│   └── 1/
│       └── model.onnx          # ONNX model file (336MB)
│
└── openclip_vit_b32_trt/       # TensorRT EP model (Step 5B)
    ├── config.pbtxt            # ONNX Runtime TRT EP config (FP16)
    └── 1/
        └── model.onnx          # Symlink → ../openclip_vit_b32/1/model.onnx

scripts/
├── trt_quick_test.py           # Quick A/B benchmark with server-side metrics
├── benchmark_tensorrt.py       # Full 3-way benchmark script
├── triton_trt_entrypoint.sh    # Container entrypoint
└── convert_to_tensorrt.py      # Reference: standalone ONNX → TRT conversion

benchmark_results/
└── tensorrt_ep_results.json    # Benchmark results JSON

Dockerfile.tensorrt             # Docker image definition
deploy_tensorrt_gpu.sh          # Build + push to Docker Hub
run_tensorrt_benchmark.sh       # Benchmark runner
STEP_5B_GPU_RESULTS.md          # Detailed benchmark results
```
