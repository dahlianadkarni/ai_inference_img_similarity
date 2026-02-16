
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