# Photo Near-Duplicate Detection (macOS Photos)

Finds groups of near-duplicate photos using OpenCLIP embeddings, then lets you review them in a local web UI and optionally delete selected photos from Apple Photos.

This is a personal project focussed on learning AI Inference frameworks and GPU Accleration

**🎬 Project Presentation:**
- **[presentation/README.md](presentation/README.md)** — Complete presentation materials (slides, executive summary, technical deep-dive)
- **[presentation/slides.html](presentation/slides.html)** — Interactive HTML slide deck (18 slides, open in browser)

**📚 Documentation:**
- **[COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md)** — Quick cheatsheet commands to start servers, etc.
- **[PLAN.md](PLAN.md)** — 8-step infrastructure learning plan (Steps 1–7 complete ✅, Step 8 in progress 🔄)
- **[K8S_PLAN.md](K8S_PLAN.md)** — Step 8: Local Kubernetes phased implementation plan (kind, HPA, PDB, Helm)
- **[STEP_8_K8S_RESULTS.md](STEP_8_K8S_RESULTS.md)** — Step 8: Live cluster output, pod startup, HPA events
- **[DOCKER_README.md](DOCKER_README.md)** — Step 2: Docker containerization guide with security best practices
- **[GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md)** — Step 3: Cloud GPU deployment guide (Vast.ai, RunPod, Lambda Labs)
- **[TRITON_SETUP.md](TRITON_SETUP.md)** — Step 4: Triton Inference Server setup, benchmarking, and trade-offs
- **[STEP_5A_FINDINGS.md](STEP_5A_FINDINGS.md)** — Step 5A: ONNX Runtime optimization report (binary protocol fix, profiling)
- **[STEP_5B_TENSORRT.md](STEP_5B_TENSORRT.md)** — Step 5B: TensorRT EP setup guide and deployment
- **[STEP_5B_GPU_RESULTS.md](STEP_5B_GPU_RESULTS.md)** — Step 5B: TensorRT vs ONNX benchmark results
- **[STEP_6A_A100_RESULTS.md](STEP_6A_A100_RESULTS.md)** — Step 6A: 3-way backend comparison on A100 SXM4
- **[STEP_6A_RTX4080_RESULTS.md](STEP_6A_RTX4080_RESULTS.md)** — Step 6A: RTX 4080 comparison (consumer vs datacenter GPU)
- **[STEP_6B_RESULTS.md](STEP_6B_RESULTS.md)** — Step 6B: 4x RTX 4080 multi-GPU scaling study
- **[STEP_7_GRPC_RESULTS_A100.md](STEP_7_GRPC_RESULTS_A100.md)** — Step 7: 5-way protocol comparison on A100 SXM4
- **[STEP_7_GRPC_RESULTS_RTX_4090.md](STEP_7_GRPC_RESULTS_RTX_4090.md)** — Step 7B: RTX 4090 repeat + cross-GPU comparison
- **[DEMO_SETUP_CLEAN.md](DEMO_SETUP_CLEAN.md)** — Demo mode setup (separate server on port 8081)
- **[docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)** — Project history and implementation notes

---

## 📊 Key Learnings: Backend Comparison Summary

After benchmarking 3 inference backends (PyTorch, Triton ONNX, Triton TensorRT) across multiple remote GPUs — and also benchmarking in-process locally on macOS Apple Silicon — here are the key takeaways:

### Performance by Backend

| Backend | Best For | GPU Compute (A100) | Client Latency (A100) | When to Use |
|---------|----------|:------------------:|:---------------------:|-------------|
| **PyTorch FastAPI** | Remote clients | 56.9ms | **56.9ms** ⚡ | Client-facing API (accepts JPEG) |
| **Triton ONNX CUDA EP** | Server-side | **4.4ms** ⚡ | 182.9ms | Local inference, batch processing |
| **Triton TensorRT EP** | Consumer GPUs | 2.0ms (RTX 4080) | 336ms | RTX GPUs, max GPU efficiency |
| **PyTorch MPS (local)** | Local macOS use | **~12ms** (M-series) | 0ms (in-process) | No server needed; auto-selected by app |

### Top 5 Insights

1. **Protocol Design > Compute Optimization**: PyTorch wins for remote clients (56.9ms) despite Triton being 12.8× faster at GPU compute (4.4ms) — because PyTorch accepts 10KB JPEGs while Triton requires 602KB float tensors.

2. **GPU Architecture Matters**: TensorRT is 2.9× faster on RTX 4080 but 6.5× slower on A100. Consumer vs datacenter GPUs have opposite optimization profiles.

3. **Serialization Bottlenecks Are Real**: JSON `.tolist()` was 1,000–4,800× slower than binary encoding, masking all GPU optimizations. Always profile the full pipeline.

4. **Multi-GPU Scaling Isn't Free**: 4× RTX 4080 delivered only 1.8× throughput (45% efficiency) — network/CPU became the bottleneck, not GPU.

5. **Production Recommendation**: Use PyTorch FastAPI for client-facing APIs + Triton ONNX as a local inference backend for batch workloads.

6. **Local Apple Silicon Beats Remote GPU End-to-End**: PyTorch MPS (~12ms in-process) is 4.7× faster than a remote PyTorch FastAPI call (56.9ms), even though A100 Triton is 2.7× faster at raw GPU compute (4.4ms). Zero network overhead wins. The app auto-selects MPS on macOS.

**📈 Full benchmark data:** [presentation/04_BENCHMARK_RESULTS.md](presentation/04_BENCHMARK_RESULTS.md)  
**🍎 macOS local benchmark script:** `scripts/benchmark_macos_local.py` → `benchmark_results_macOS_local/`

---

## What It Does

- **Scans photos** from Apple Photos (via AppleScript) or any directory
- **Detects duplicates** at three levels:
  - Exact duplicates (MD5 hash matching)
  - Perceptual duplicates (dHash - finds resized/edited versions)
  - AI similar groups (OpenCLIP embeddings - finds semantically similar photos)
- **Review UI** with separate workflows:
  - **Exact Duplicates**: Auto-selected by default, batch delete with one click
  - **Perceptual Duplicates**: Manual review, check groups to activate
  - **AI Groups**: Fine-grained selection, teach the AI with feedback
- **Smart features**:
  - Keeps oldest photo by default (based on EXIF date)
  - Shows disk savings in real-time
  - Full-screen modal with keyboard navigation
  - Individual image toggle within groups

## Quickstart

See DEMO_SETUP.md and DEMO_SOLUTION.md for demo

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start UI only (port 8000)
python -m src.ui.main

# Or start both inference + UI
python start_services.py

# Or use demo dataset (UI on port 8081; inference runs on port 8002)
python start_services.py --ui-demo
```

Open http://127.0.0.1:8080 (or the demo UI at http://127.0.0.1:8081). The inference service runs on http://127.0.0.1:8002.

## End-to-End Workflow (How the Pieces Fit)

### Step 1 — Scan

Produces a JSON list of images to analyze.

- Output: scan_for_embeddings.json
- UI: Use the “Control Panel → Start Scan” button
- CLI (optional):

```bash
source venv/bin/activate
python -m src.scanner.main --photos-library --use-applescript --keep-export --limit 200
```

Tip: With `--keep-export`, AppleScript exports into `.cache/photos_export` by default and subsequent scans can be incremental (it skips re-exporting files that are already present).

macOS will prompt “Terminal wants to control Photos” on first run. Allow it.

I have local photos stored in photos_local cache.
Restore scan_for_embeddings.json from real photos (fast, no Photos app needed):
```source venv/bin/activate
python -m src.scanner.main .cache/photos_local \
  --output scan_for_embeddings.json \
  --duplicates-output scan_duplicates.json \
  --cache-file .cache/scan_cache.json
  ```

If you want estimates before running a full scan of your Photos “originals” (Full Disk Access required), run:

```bash
source venv/bin/activate
python -m src.scanner.main --photos-library --estimate --estimate-sample 200
```

### Step 2 — Generate Embeddings + Similarity Groups

Computes embeddings and writes similarity groups to disk.

- Output directory: embeddings/
- Group output: embeddings/similar_groups.json
- UI: Use “Control Panel → Generate Embeddings” and set your threshold
- CLI (optional):

```bash
source venv/bin/activate
python -m src.embedding.main scan_for_embeddings.json --output embeddings --similarity-threshold 0.85
```

Notes:
- Higher thresholds (e.g., 0.95–0.99) are stricter.
- Embeddings are cleared before regenerating so old data doesn’t accumulate.

### Step 3 — Review Groups (Delete / Keep / Feedback)

The UI has three tabs:

**Scanner Results Tab:**
- **Exact Duplicates** (green theme): All groups checked by default, oldest photo kept
  - Uncheck groups to exclude from batch delete
  - Click thumbnails to toggle individual images (KEEP ★ / DEL ✗)
  - Click Delete Selected to remove marked photos
- **Perceptual Duplicates** (orange theme): Manual review workflow
  - Check groups to activate them
  - Same toggle functionality as exact duplicates
  - Review and delete in batches

**AI Results Tab:**
- Side-by-side comparison of similar groups
- Delete Selected: deletes chosen photos from Apple Photos
- Keep All: marks the group reviewed
- Not Similar (Teach AI): stores "negative examples" in embeddings/feedback.pkl

**Universal Features:**
- 🔍 Click to open full-screen modal with prev/next navigation
- Real-time disk savings calculation
- Keyboard shortcuts in modal (← → arrows, ESC to close)

### Step 4 — Re-analyze With Feedback

After giving feedback, click “Re-analyze with Feedback” (or regenerate embeddings) to apply the learned penalties/boosts during grouping.

## Outputs

- scan_for_embeddings.json — scan results used for embedding generation
- embeddings/embeddings.npy + embeddings/metadata.json — stored embeddings
- embeddings/similar_groups.json — groups to review
- embeddings/feedback.pkl — persisted feedback examples

## Architecture

The app uses a **client-service architecture** — the same pattern used by production inference systems like Triton, TorchServe, and vLLM.

```
┌──────────────────────┐         ┌──────────────────────┐
│   Client/UI          │         │ Inference Service    │
│  (Port 8000)         │─HTTP──→ │  (Port 8002)         │
│                      │         │                      │
│ • Scan photos       │         │ • Load model         │
│ • Call service      │         │ • Generate embeddings│
│ • Store embeddings  │         │ • Return JSON        │
│ • Group results     │         │ • Stateless API      │
│ • Display UI        │         │                      │
└──────────────────────┘         └──────────────────────┘
```

**Inference Backends (switchable via env vars):**

| Backend | Docker Image | Ports | Status |
|---------|-------------|-------|--------|
| PyTorch + FastAPI | `Dockerfile.gpu` | 8002 | ✅ Step 3 |
| NVIDIA Triton (ONNX CUDA EP) via HTTP | `Dockerfile.triton` | 8003 (HTTP), 8004 (gRPC), 8005 (metrics) | ✅ Step 4/5A |
| NVIDIA Triton (TensorRT EP) via HTTP | `Dockerfile.tensorrt` | 8003 (HTTP), 8004 (gRPC), 8005 (metrics) | ✅ Step 5B |
| All 3 backends (unified) | `Dockerfile.step6a-all` | 8002 (PyTorch), 8003/8004/8005 (Triton) | ✅ Step 6A |
| NVIDIA Triton (ONNX CUDA EP) via gRPC | same `Dockerfile.triton` | 8004 (gRPC) | ✅ Step 7 |
| NVIDIA Triton (TRT EP) via gRPC | same `Dockerfile.tensorrt` | 8021 (gRPC, step6a) | ✅ Step 7 |
| PyTorch FastAPI (Kubernetes / kind) | `photo-duplicate-inference:k8s-cpu` | 8092 (NodePort, host) | ✅ Step 8 |

```bash
# Switch backends with environment variables:
export INFERENCE_BACKEND=pytorch       # plain PyTorch FastAPI
export INFERENCE_BACKEND=triton        # Triton via HTTP binary protocol
export INFERENCE_BACKEND=triton_grpc   # Triton via gRPC (Step 7)

export INFERENCE_SERVICE_URL=http://localhost:8002   # PyTorch
export INFERENCE_SERVICE_URL=http://localhost:8003   # Triton HTTP
# gRPC URL is auto-derived as HTTP_PORT+1 (e.g. 8003 → 8004)
# Override with: export TRITON_GRPC_URL=localhost:8004
python -m src.ui.main
```

See [TRITON_SETUP.md](TRITON_SETUP.md) for Triton setup, benchmarking, and trade-offs evaluation.

**Key Principles:**
- **Separation of Concerns**: Client handles photos/metadata, service handles ML inference
- **Stateless Service**: Each request is independent, enabling horizontal scaling
- **Clean HTTP Boundary**: Services communicate via JSON/REST APIs
- **Independent Deployment**: Can run on same machine or separate GPU server

**Three Embedding Modes:**
```bash
# Local mode (original behavior - model loads inline)
python -m src.embedding.main_v2 scan_for_embeddings.json --mode local

# Remote mode (calls inference service)
python -m src.embedding.main_v2 scan_for_embeddings.json --mode remote

# Auto mode (tries remote, falls back to local)
python -m src.embedding.main_v2 scan_for_embeddings.json
```

See [DOCKER_README.md](DOCKER_README.md) for containerization details.

## Repo Map

- src/scanner/ — scanning + AppleScript Photos export
- src/embedding/ — OpenCLIP embedding generation + storage
- src/inference_service/ — stateless FastAPI inference server (client + server)
- src/grouping/ — clustering + feedback learner
- src/ui/ — FastAPI UI server
- scripts/ — benchmarking, profiling, deployment, and analysis tools
- model_repository/ — Triton model configs (ONNX CUDA EP + TensorRT EP)
- benchmark_results/ — raw benchmark JSON data from all steps

## Safety Notes

- Everything runs locally by default.
- Deletion uses AppleScript to delete photos from the Photos library; start by testing on a small limit.



Step	Time	Resource	Bottleneck
1. Scanner	1-5 min	I/O + CPU	File reading, hash computation
2. Embedding	10-30 sec	GPU/MPS	Neural network (MAIN BOTTLENECK)
3. Grouping	<1 sec	CPU	Just math on vectors
4. UI	Instant	Memory	Display