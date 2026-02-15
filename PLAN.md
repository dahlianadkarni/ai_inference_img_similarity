
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

**Step 6A: Single-Machine 3-Way Comparison** (Recommended)
    - Deploy all 3 backends on **one GPU instance** for apples-to-apples comparison:
        - PyTorch FastAPI (Step 3)
        - Triton ONNX CUDA EP (Step 4/5A)
        - Triton ONNX TensorRT EP (Step 5B)
    - Use Docker Compose to run 3 containers with different ports on same instance:
        - PyTorch: port 8002
        - Triton ONNX: ports 8010/8011/8012 (HTTP/gRPC/Metrics)
        - Triton TRT: ports 8020/8021/8022 (HTTP/gRPC/Metrics)
        - See `docker-compose-step6a.yml` and `deploy_step6a.sh`
    - Two benchmarking approaches (run both for comprehensive analysis):
        - **Remote**: `scripts/benchmark_all_three.py` from your Mac
            - Includes network latency (~200-400ms) but relative comparison still valid
            - Captures server-side metrics from Triton `/metrics` to show pure GPU time
            - No SSH required, easier to run
        - **Local**: `scripts/benchmark_all_three_local.py` after SSH into instance
            - Eliminates network latency for true backend performance
            - More accurate absolute timings
            - Requires SSH access and pip install dependencies
    - Measure and compare:
        - Server-side GPU compute time (from Triton metrics, estimated for PyTorch)
        - Single-image latency (p50, p95, p99)
        - Batch throughput (images/sec at different batch sizes)
        - Memory footprint (VRAM usage per backend)
        - Cold-start/warmup time (TRT EP takes 2-5 min first load)
        - Concurrent throughput (req/sec under same client load)
    - Document in `STEP_6A_RESULTS.md` with comparison tables and winner declaration
    - Instance choice: RTX A4000 or A5000 (~$0.30-0.50/hr)
    - **Value**: Definitive answer on which backend is fastest for single-GPU production

**Step 6B: Multi-GPU Scaling Study** (Optional)
    - After determining single-GPU winner, test Triton's multi-GPU scaling
    - Update Triton model config (`config.pbtxt`) to set:
            instance_group [ { kind: KIND_GPU, count: 2 } ]  # or 4, 8
    - Ensure the container is started with access to all GPUs (e.g., `--gpus all`)
    - No Docker rebuild needed if only `config.pbtxt` changes; just update the file in the model repository
    - Test on 2x, 4x, and/or 8x GPU instances (e.g., 4x RTX A6000, 8x A100)
    - Run benchmarking scripts with increasing concurrent load to saturate GPUs
    - Measure throughput scaling: Does 4x GPU = 4x throughput?
    - Find saturation point: When does adding GPUs stop helping?
    - Compare cost-efficiency: $/1000 images at each scale
    - Document scaling limitations, bottlenecks, and cost analysis
    - Document in `STEP_6B_MULTI_GPU_RESULTS.md`
    - **Value**: Validates Triton's multi-GPU scaling story; answers "when do I need more GPUs?"