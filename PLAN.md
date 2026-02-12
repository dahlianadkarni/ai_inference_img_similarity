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

Step 4: Introduce one inference framework. Eg: NVIDIA Triton Inference Server
- Serve your CLIP/OpenCLIP model
- Use dynamic batching
- Call it via HTTP or gRPC
- Compare Triton vs “plain Python service”

Step 5 (Optional but strong): TensorRT optimization
- Convert your model to TensorRT
- Compare latency vs PyTorch
- Observe memory usage