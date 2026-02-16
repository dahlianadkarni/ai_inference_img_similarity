# Step 6A: Implementation Checklist

**Goal:** Deploy all 3 backends (PyTorch, Triton ONNX, Triton TRT) on one GPU instance, benchmark, and declare a winner.

---

## Phase 1: Pre-flight Verification (Local)
- [x] All infrastructure files exist and are substantive
- [x] ONNX model present (335MB)
- [x] Triton configs for both ONNX and TRT backends exist
- [x] PyTorch server code exists (`src/inference_service/server.py`)
- [x] Benchmark scripts exist (`scripts/benchmark_all_three.py`, `scripts/benchmark_all_three_local.py`)
- [x] Analysis script exists (`scripts/analyze_step6a_results.py`)
- [x] Step 6A Docker image **not yet built**

## Phase 2: Build & Push Docker Image (Local)
- [x] Fix minor Dockerfile comment (says `-f Dockerfile.step6a` but file is `Dockerfile.step6a-all`)
- [x] Build the all-in-one image for `linux/amd64` using `docker buildx` (322s / ~5.4min)
- [x] Push to Docker Hub as `dahlianadkarni/photo-duplicate-step6a:latest` ✅
- [x] Verify push succeeded ✅

> ⏱ Expected: 10-20 minutes (large image: Triton base + PyTorch + ONNX model)

## Phase 3: Deploy to Vast.ai ✅
- [x] Rented **A100 SXM4 80GB** on Vast.ai (Instance 31477749)
- [x] Docker image: `dahlianadkarni/photo-duplicate-step6a:latest`
- [x] All 7 ports exposed and mapped
- [x] Container started all 3 backends automatically

## Phase 4: Health Check & Warmup ✅
- [x] Health check PyTorch backend → HTTP 200 (31ms RTT)
- [x] Health check Triton ONNX → HTTP 200 (22ms RTT)
- [x] Health check Triton TRT → HTTP 200 (24ms RTT)
- [x] TRT engine compilation on first inference call (~56s)
- [x] All 3 backends verified returning valid 512-dim embeddings

## Phase 5: Run Benchmarks ✅
- [x] Remote benchmark: 30 iterations per backend, batch sizes 1-32
- [x] Captured Triton server-side metrics (GPU compute, total request)
- [x] TRT EP batch 8+ timed out (per-batch-size engine recompilation)
- [x] Results: `benchmark_results/step6a_a100_remote.json`

## Phase 6: Analyze & Document ✅
- [x] Comprehensive analysis: `STEP_6A_A100_RESULTS.md`
- [x] **Winners:** PyTorch (client latency), Triton ONNX (GPU compute), TRT (not recommended)

---

## Port Layout Reference

| Backend          | HTTP  | gRPC  | Metrics |
|-----------------|-------|-------|---------|
| PyTorch FastAPI  | 8002  | —     | —       |
| Triton ONNX     | 8010  | 8011  | 8012    |
| Triton TRT      | 8020  | 8021  | 8022    |
