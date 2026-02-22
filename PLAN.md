
Step 1 (Most important): Split your app into "client" and "inference service" - ✅ DONE

Step 2: Containerize the inference service - ✅ DONE
- Create a Dockerfile for the inference service ✅
- Expose:
    - /embed ✅
    - /healthz ✅
- Make model choice configurable via env vars ✅
- Run it locally via Docker ✅
- See DOCKER_STEP2.md for details

Step 3: Move the same container to NVIDIA GPU - ✅ DONE
- Run the container on:
    - Cloud GPU VM (T4 / A10 / RTX PRO 6000) ✅
    - OR internal Akamai GPU if available
- Use:
    - NVIDIA container runtime ✅
    - CUDA-enabled base image ✅
- See GPU_DEPLOYMENT.md for details

Step 4: Introduce one inference framework. Eg: NVIDIA Triton Inference Server - ✅ DONE
- Serve your CLIP/OpenCLIP model via ONNX on Triton ✅
- Use dynamic batching ✅
- Call it via HTTP or gRPC ✅
- Compare Triton vs "plain Python service" ✅
- Evaluate trade-offs: model loading latency, dynamic batching, GPU utilization ✅
- See TRITON_SETUP.md for details


Step 5A: ONNX Runtime/Triton Optimization (No TensorRT) - ✅ COMPLETE
    - Enable and test CUDA graphs in Triton config for GPU inference ✅
    - Profile ONNX Runtime for slow ops or CPU fallbacks (use ONNX Runtime profiling tools) ✅
    - Experiment with `max_queue_delay_microseconds` and client concurrency to maximize dynamic batching (document impact on throughput/latency) ✅
    - Document any serialization/deserialization overheads found in ONNX Runtime ✅
    - Observe and record GPU memory usage for ONNX Runtime and PyTorch backends ✅
    - Summarize findings and update documentation with recommendations ✅
    - GPU validation complete: Binary protocol fix confirmed, but native PyTorch outperforms ONNX by 5.2x for batch inference ✅
    - See STEP_5A_FINDINGS.md for complete report ✅
    - **Recommendation**: Continue with PyTorch backend (superior performance) or proceed to Step 5B (TensorRT) if Triton features are required


Step 5B (Optional): TensorRT Optimization - ✅ COMPLETE
    - Convert your model to TensorRT (if compatible) ✅ (pivoted to TRT Execution Provider — uses TRT 8.6 built into Triton)
    - Serve the TensorRT-optimized model via Triton ✅ (Dockerfile.tensorrt with TRT EP config, FP16)
    - Compare latency and throughput vs PyTorch and ONNX Runtime ✅ (scripts/trt_quick_test.py with server-side metrics)
    - Observe and record GPU memory usage for all backends ✅
    - Document any serialization/deserialization overheads found in TensorRT backend ✅
    - Summarize findings and update documentation with recommendations ✅
    - **Result**: TRT EP achieves **6.6x faster GPU compute** (4.7ms vs 31.1ms) over ONNX CUDA EP
    - **Key finding**: TRT EP likely matches or exceeds native PyTorch (~10-15ms estimated), closing the 5.2x gap from Step 5A
    - See STEP_5B_TENSORRT.md for setup guide, STEP_5B_GPU_RESULTS.md for benchmark results

Step 6: Final Comparison & Multi-GPU Scaling (Optional)

**Step 6A: Single-Machine 3-Way Comparison** - ✅ COMPLETE
    - Deployed all 3 backends on **A100 SXM4 80GB** (Vast.ai) for apples-to-apples comparison:
        - PyTorch FastAPI (Step 3) ✅
        - Triton ONNX CUDA EP (Step 4/5A) ✅
        - Triton ONNX TensorRT EP (Step 5B) ✅
    - **Results (30 iterations, remote benchmark):**
        - **PyTorch wins client-side:** 56.9ms single-image, 64.3 img/s batch-32
        - **Triton ONNX wins server-side:** 4.4ms GPU compute (12.8× faster than PyTorch estimate)
        - **TRT EP not recommended:** 29.1ms GPU compute (6.5× slower than ONNX), engine recompilation issues
    - **Key insight:** Network/serialization overhead dominates remote benchmarks.
      Triton ONNX achieves 4.4ms GPU compute but 182.9ms client-side (due to 602KB tensor transfer).
      PyTorch achieves 56.9ms client-side by accepting efficient base64 JPEG (~10KB).
    - **Production recommendation:** PyTorch FastAPI for client-facing API + Triton ONNX as local inference backend
    - See `STEP_6A_A100_RESULTS.md` for full analysis, `benchmark_results/step6a_a100_remote.json` for raw data

**Step 6B: Multi-GPU Scaling Study** - ✅ COMPLETE
    - Tested 4x RTX 4080 multi-GPU scaling on Vast.ai ✅
    - **Results:**
        - **Peak throughput:** 43.2 img/s at concurrency 32 (1.8× vs single GPU, only 45% efficiency) ✅
        - **Scaling inefficiency:** Network overhead dominates (97% of latency), GPU compute only 8.4ms ✅
        - **Cost-efficiency:** 2.2× more expensive per image than 1x GPU ✅
        - **Bottlenecks:** 602KB tensor payload, PCIe Gen4 interconnect, low dynamic batching (22%) ✅
        - **Saturation point:** Concurrency 32; throughput drops at 64+ due to queue time buildup ✅
    - **Key finding:** Multi-GPU is **not recommended** for remote inference due to network bottleneck
    - **Recommendation:** Use 1x A100 or horizontal scaling (3-4× single-GPU instances) for production
    - **Optimization paths:** JPEG base64 input (5-6× faster), local deployment (14× faster), NVLink GPUs (2-3× better scaling)
    - See `STEP_6B_RESULTS.md` for complete analysis
    - **Value**: Proves that multi-GPU doesn't solve network-bound workloads; validated infrastructure decision-making

**Step 7: 5-Way Protocol Comparison with gRPC** - ✅ COMPLETE
    - Motivation: gRPC (HTTP/2 + protobuf) should reduce per-call overhead vs HTTP/1.1 for the 602KB tensor payload bottleneck identified in Step 6A
    - **5 backends compared on same GPU instance (step6a docker-compose):**
        1. PyTorch FastAPI — HTTP/1.1 + base64 JPEG (~10KB/image)
        2. Triton ONNX HTTP — HTTP/1.1 + binary FP32 (~602KB/image)
        3. Triton ONNX gRPC — HTTP/2 + binary FP32 (~602KB/image)
        4. Triton TRT  HTTP — HTTP/1.1 + binary FP32 (~602KB/image)
        5. Triton TRT  gRPC — HTTP/2 + binary FP32 (~602KB/image)
    - **Why 5-way?** GPU and TRT vs ONNX differences are already known from Step 6A.
      The new question is: within each Triton backend, does gRPC reduce the 178ms transport overhead?
      Including PyTorch gives a reference point for what an optimized input format can achieve.
    - **Client changes:**
        - Added `triton_grpc` backend to `src/inference_service/client.py` using `tritonclient.grpc`
        - gRPC URL auto-derived as HTTP_PORT+1 (e.g. 8003 HTTP → 8004 gRPC), or set via `TRITON_GRPC_URL` env var
    - **Benchmark script:** `scripts/benchmark_grpc_vs_http.py`
        - All 5 backends benchmarked on the same Vast.ai instance
        - Batch sizes: 1, 4, 8, 16, 32  |  Concurrency: 1, 8, 16
        - Records p50/p95/p99 latency, img/s throughput, server-side GPU compute from Triton metrics
        - Saves raw JSON to `benchmark_results/step7_5way_<timestamp>.json`
    - **Expected findings to measure:**
        - gRPC vs HTTP overhead isolation (same GPU compute, different transport)
        - Whether HTTP/2 multiplexing improves concurrent throughput
        - PyTorch JPEG input as lower bound on client-side latency (10KB vs 602KB)
    - **Requirements:** `tritonclient[http,grpc]>=2.41.0` (updated in requirements.txt)
    - **Test instance:** Vast.ai A100 SXM4 80GB — IP `207.180.148.74` (Massachusetts, USA) — Instance ID 31781954
    - **Port mappings:** PyTorch 47150, ONNX HTTP 47088, ONNX gRPC 47037, TRT HTTP 47008, TRT gRPC 47045
    - **Results (30 iterations, A100 SXM4 80GB, 2026-02-20):**
        - **PyTorch wins outright:** 64.2ms batch-1, 48.5 img/s batch-32 — 3× faster than best Triton serial
        - **gRPC is slower for batch=1:** 0.86–0.96× vs HTTP; TCP handshake is NOT the bottleneck
        - **gRPC wins from batch=4+:** 1.4–1.7× speedup for ONNX; HTTP/2 framing pays off at multi-MB sizes
        - **HTTP scales better concurrently:** ONNX HTTP 43.6 img/s at conc=16 vs ONNX gRPC 7.5 img/s (client channel contention)
        - **TRT engine anomaly:** TRT HTTP shows ~1658ms GPU metric (engine compiling); TRT gRPC (post-cache) = 3.6ms
    - **Key finding:** gRPC does NOT reduce the 178ms transport overhead at batch=1. The bottleneck is 602KB payload bandwidth. Input format (JPEG base64) is the only path to <100ms remote latency.
    - Raw data: `benchmark_results/step7_5way_20260220_221313.json`
    - See `STEP_7_GRPC_RESULTS_A100.md` for full A100 analysis

**Step 7B: RTX 4090 Repeat — Pennsylvania** - ✅ COMPLETE
    - Repeating 5-way benchmark on consumer-grade RTX 4090 to compare against A100
    - Slower GPU compute → transport overhead is a smaller fraction of total latency — different gRPC tradeoff expected
    - **Test instance:** Vast.ai RTX 4090 — IP `173.185.79.174` (Pennsylvania, USA) — $0.391/hr
    - **Port mappings:** PyTorch 50616, ONNX HTTP 50680, ONNX gRPC 50764, TRT HTTP 50048, TRT gRPC 50286
    - **Results (30 iterations, RTX 4090, 2026-02-20):**
        - **PyTorch concurrent peak:** 49.1 img/s at conc=16 (vs 30.5 on A100) — FastAPI async + JPEG scales well
        - **gRPC worse at batch=1 than A100:** 11–17% slower vs HTTP (vs 4–14% on A100); H5 confirmed
        - **gRPC ONNX wins at batch=8+:** 2.0× speedup (only from b=8 vs b=4 on A100); slower GPU shifts crossover point
        - **HTTP still wins under concurrency:** ONNX HTTP 32.2 img/s vs ONNX gRPC 4.0 img/s at conc=16 (8× gap vs 5.8× on A100)
        - **Cost-efficiency:** RTX 4090 latency is 1.3–2.1× worse than A100 at 2.3× lower cost — broadly fair
    - Raw data: `benchmark_results/step7_5way_20260220_232454.json`
    - See `STEP_7_GRPC_RESULTS_RTX_4090.md` for full analysis with cross-GPU comparison

**Step 8: Local Kubernetes (kind)** - 🔄 IN PROGRESS
    - Goal: deploy the same inference container into a local Kubernetes cluster (kind) to learn K8s orchestration patterns
    - **Tool**: kind (Kubernetes-in-Docker) + kubectl + metrics-server + hey load generator
    - **Image**: `photo-duplicate-inference:k8s-cpu` — same `Dockerfile`, new tag, ARM64-native, never pushed to Docker Hub
    - **Port**: `localhost:8092` (NodePort 30092 via kind extraPortMappings — avoids collision with existing 8002/8003/8004)
    - **Phase 1 ✅**: kind v0.31.0 cluster created, metrics-server installed + patched for kind TLS, image built and loaded
    - **Phase 2 ✅**: All manifests applied (`kubectl apply -k k8s/`); 2/2 pods Running; `GET /health → {"status":"ok"}`
    - **Phase 3 ✅**: HPA deployed (2–6 replicas, 60% CPU target, 3-min scaledown stabilization)
    - **Phase 4**: kubectl observability practice (logs, exec, top, rolling update, rollback) — pending
    - **Phase 5**: PodDisruptionBudget + ResourceQuota — deployed alongside Phase 2 ✅
    - **Phase 6** (optional): Helm chart for multi-environment deploy
    - See `K8S_PLAN.md` for full phased plan and coexistence notes
    - See `STEP_8_K8S_RESULTS.md` for live cluster output