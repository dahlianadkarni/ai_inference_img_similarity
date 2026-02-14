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


Step 5A: ONNX Runtime/Triton Optimization (No TensorRT)
    - Enable and test CUDA graphs in Triton config for GPU inference
    - Profile ONNX Runtime for slow ops or CPU fallbacks (use ONNX Runtime profiling tools)
    - Experiment with `max_queue_delay_microseconds` and client concurrency to maximize dynamic batching (document impact on throughput/latency)
    - Document any serialization/deserialization overheads found in ONNX Runtime
    - Observe and record GPU memory usage for ONNX Runtime and PyTorch backends
    - Summarize findings and update documentation with recommendations

Step 5B (Optional): TensorRT Optimization
    - Convert your model to TensorRT (if compatible)
    - Serve the TensorRT-optimized model via Triton
    - Compare latency and throughput vs PyTorch and ONNX Runtime
    - Observe and record GPU memory usage for all backends
    - Document any serialization/deserialization overheads found in TensorRT backend
    - Summarize findings and update documentation with recommendations