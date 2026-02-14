# Step 4: NVIDIA Triton Inference Server Setup

> **Goal:** Replace the plain PyTorch FastAPI service with NVIDIA Triton Inference Server for production-grade model serving with dynamic batching and optimized inference.

## Overview

NVIDIA Triton Inference Server provides:
- **Dynamic Batching**: Automatically batches requests for better GPU utilization
- **Multi-model Support**: Serve multiple models/versions simultaneously
- **Multiple Backends**: PyTorch, TensorFlow, ONNX, TensorRT
- **HTTP & gRPC APIs**: Industry-standard protocols
- **Performance Metrics**: Built-in Prometheus metrics
- **Model Versioning**: Hot-swap models without downtime

## Architecture Philosophy

**Independent Backends, Unified Client**

- **Step 3**: `Dockerfile.gpu` → PyTorch + FastAPI (port 8002)
- **Step 4**: `Dockerfile.triton` → Triton Server (ports 8000/8001/8002 in cloud)
- **Step 5**: `Dockerfile.tensorrt` → TensorRT + FastAPI (port 8002)

### Port Allocation Strategy

**Local Development (Mac):**
```
8000 → Main UI (Photos Library)
8001 → Demo UI
8002 → PyTorch Inference Server (Step 3)
8003 → Triton HTTP endpoint (Step 4)
8004 → Triton gRPC endpoint (Step 4)
8005 → Triton Metrics (Step 4)
```

**Cloud Deployment (Vast.ai):**
```
One GPU instance at a time, each uses standard ports:
- PyTorch:  8002
- Triton:   8000 (HTTP), 8001 (gRPC), 8002 (metrics)
- TensorRT: 8002
```

Each backend is a **separate Docker image**. You deploy ONE at a time on your GPU instance.

The **client code** (UI + inference_service/client.py) works with ALL backends via environment variables:

```bash
# Local Testing
export INFERENCE_SERVICE_URL=http://localhost:8002  # PyTorch
export INFERENCE_SERVICE_URL=http://localhost:8003  # Triton (mapped from 8000)
export INFERENCE_BACKEND=triton  # Required for Triton

# Cloud Deployment (Vast.ai)
export INFERENCE_SERVICE_URL=http://142.170.89.112:63292  # PyTorch
export INFERENCE_SERVICE_URL=http://142.170.89.112:26500  # Triton (mapped port)
export INFERENCE_BACKEND=triton  # Required for Triton
```

### Architecture Comparison

**Current (Step 3): PyTorch + FastAPI**
```
Client → FastAPI (port 8002) → PyTorch Model → GPU
         - Custom batching
         - Manual optimization
         - Single model at a time
         - Easy to debug
```

**Step 4: Triton Inference Server**
```
Client → Triton Server (HTTP:8000, gRPC:8001) → ONNX Model → GPU
         - Dynamic batching (automatic)
         - Optimized inference pipeline
         - Multiple models/versions
         - Built-in metrics
```

**Step 5: TensorRT (Future)**
```
Client → FastAPI (port 8002) → TensorRT Engine → GPU
         - Maximum throughput
         - INT8 quantization
         - Graph optimization
```

---

## Implementation Plan

### Phase 1: Model Export to ONNX
1. Export OpenCLIP ViT-B-32 to ONNX format
2. Verify ONNX model outputs match PyTorch
3. Optimize ONNX model (optional: quantization, graph optimization)

### Phase 2: Triton Model Repository
1. Create model repository structure
2. Write `config.pbtxt` for model configuration
3. Set up dynamic batching parameters

### Phase 3: Triton Server Deployment
1. Create Dockerfile for Triton
2. Deploy on GPU instance (Vast.ai)
3. Expose HTTP (8000) and gRPC (8001) endpoints

### Phase 4: Client Updates
1. Create Triton client wrapper
2. Support both backends (PyTorch vs Triton)
3. Add backend selection in UI

### Phase 5: Benchmarking
1. Measure latency (single request)
2. Measure throughput (concurrent requests)
3. Compare PyTorch vs Triton performance
4. Document results

---

## Prerequisites

- Completed Step 3 (GPU deployment)
- NVIDIA Triton Inference Server Docker image
- `onnx` and `onnxruntime` Python packages

---

## Quick Start

```bash
# 1. Export model to ONNX
python scripts/export_to_onnx.py

# 2. Build Triton Docker image
docker build -f Dockerfile.triton -t photo-duplicate-triton:latest .

# 3. Test locally
docker run --rm -p 8003:8000 -p 8004:8001 -p 8005:8002 \
  photo-duplicate-triton:latest

# 4. Test Triton endpoint
curl http://localhost:8003/v2/health/ready

# 5. Test with client
export INFERENCE_BACKEND=triton
export INFERENCE_SERVICE_URL=http://localhost:8003
python -m src.ui.main  # Start UI on port 8000

# 6. Deploy to Vast.ai (once local testing passes)
docker buildx build --platform linux/amd64 \
  -f Dockerfile.triton \
  -t yourdockeruser/photo-duplicate-triton:gpu-linux-amd64 \
  --push .

## Vast.ai Deployment Instructions

**1. Use the image:**
  yourdockeruser/photo-duplicate-triton:gpu-linux-amd64

**2. Expose ports:**
  - 8000 (HTTP)
  - 8001 (gRPC)
  - 8002 (metrics)
    
  In Vast.ai, set up port mapping so that external ports map to container ports 8000, 8001, 8002. For example:
  - External 8000 → Container 8000
  - External 8001 → Container 8001
  - External 8002 → Container 8002

**3. Environment variables:**
  - For Triton deployment, you do NOT need to set HOST, MODEL_NAME, MODEL_PRETRAINED, or LOG_LEVEL. These are only required for the PyTorch FastAPI backend.
  - For the client/UI, set:
    - `INFERENCE_BACKEND=triton`
    - `INFERENCE_SERVICE_URL=http://<vast-ip>:<external-port>` (use the external port mapped to 8000)

**4. Test the deployment:**
  - Health: `curl http://<vast-ip>:<external-port>/v2/health/ready`
  - Model info: `curl http://<vast-ip>:<external-port>/v2/models/openclip_vit_b32`

**5. Connect your UI:**
  - `export INFERENCE_BACKEND=triton`
  - `export INFERENCE_SERVICE_URL=http://<vast-ip>:<external-port>`
  - `python -m src.ui.main`

**Note:**
  - Only one backend should run at a time on a Vast.ai instance.
  - Port mapping is required for external access.
  - No extra environment variables are needed for Triton.
```

---

## Model Repository Structure

```
model_repository/
└── openclip_vit_b32/
    ├── config.pbtxt              # Model configuration
    ├── 1/                        # Version 1
    │   └── model.onnx           # ONNX model file
    └── labels.txt               # Optional: class labels
```

### config.pbtxt Example

```protobuf
name: "openclip_vit_b32"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "embedding"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

---

## Triton Endpoints

### HTTP API (Port 8000)
- **Health Check**: `GET /v2/health/ready`
- **Model Metadata**: `GET /v2/models/{model_name}`
- **Inference**: `POST /v2/models/{model_name}/infer`

### gRPC API (Port 8001)
- More efficient for high-throughput scenarios
- Binary protocol (lower overhead than HTTP)

### Metrics (Port 8002)
- Prometheus-compatible metrics
- Request counts, latency, GPU utilization

---

## Docker Setup

### Triton Server Dockerfile

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /models

# Copy model repository
COPY model_repository/ /models/

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]
```

### Build & Push

```bash
# Build for linux/amd64
docker buildx build --platform linux/amd64 \
  -f Dockerfile.triton \
  -t yourdockeruser/photo-duplicate-triton:gpu-linux-amd64 \
  .

# Push to Docker Hub
docker push yourdockeruser/photo-duplicate-triton:gpu-linux-amd64
```

---

## Client Code Updates

**Goal**: Extend `src/inference_service/client.py` to support both PyTorch and Triton backends without breaking existing code.

### Current client.py (Step 3)
```python
# Works with PyTorch FastAPI backend
class InferenceClient:
    def embed_base64_images(self, images: List[str]) -> List[List[float]]:
        response = requests.post(f"{self.base_url}/embed/base64", json={"images": images})
        return response.json()["embeddings"]
```

### Enhanced client.py (Step 4)
```python
import os
import requests
import tritonclient.http as httpclient

class InferenceClient:
    def __init__(self, base_url: str = None, backend: str = None):
        self.base_url = base_url or os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8002")
        self.backend = backend or os.getenv("INFERENCE_BACKEND", "pytorch")  # "pytorch" or "triton"
        
        if self.backend == "triton":
            # Extract host:port from URL
            host_port = self.base_url.replace("http://", "").replace("https://", "")
            self.triton_client = httpclient.InferenceServerClient(url=host_port)
    
    def embed_base64_images(self, images: List[str]) -> List[List[float]]:
        if self.backend == "pytorch":
            return self._embed_pytorch(images)
        elif self.backend == "triton":
            return self._embed_triton(images)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _embed_pytorch(self, images: List[str]) -> List[List[float]]:
        # Existing PyTorch code (unchanged)
        response = requests.post(f"{self.base_url}/embed/base64", json={"images": images})
        return response.json()["embeddings"]
    
    def _embed_triton(self, images: List[str]) -> List[List[float]]:
        # New Triton code
        # Convert base64 → numpy array → Triton inference
        # ... (implementation in Phase 4)
        pass
```

### Usage remains the same
```python
# Automatically uses INFERENCE_BACKEND environment variable
client = InferenceClient()
embeddings = client.embed_base64_images(image_list)
```

### Switching backends
```bash
# Use PyTorch (Step 3) - NO CHANGES NEEDED
python src/ui/app.py  # Uses INFERENCE_SERVICE_URL automatically

# Use Triton (Step 4)
export INFERENCE_BACKEND=triton
export INFERENCE_SERVICE_URL=http://triton-ip:8000
python src/ui/app.py

# Or pass directly
python src/ui/app.py --inference-backend triton --inference-url http://triton-ip:8000
```

---

## Performance Expectations

### PyTorch (Step 3)
- **Single Request**: ~100-500ms
- **Batch Size 32**: ~1-2 seconds
- **Throughput**: ~20-50 images/sec

### Triton (Step 4)
- **Single Request**: ~80-400ms (similar or better)
- **Batch Size 32**: ~0.8-1.5 seconds (10-20% faster)
- **Throughput**: ~40-100 images/sec (2-3x better with dynamic batching)
- **Concurrent Requests**: Significantly better (dynamic batching shines here)
: Non-Breaking, Additive

**Key Principle**: Step 3 (PyTorch) continues working exactly as-is. Step 4 (Triton) is purely additive.

### What Stays the Same
- ✅ `Dockerfile.gpu` (PyTorch backend) - **NO CHANGES**
- ✅ `src/inference_service/server.py` (FastAPI) - **NO CHANGES**
- ✅ `src/ui/app.py` (UI code) - **NO CHANGES** (uses client.py)
- ✅ Existing Docker image: `dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64`
- ✅ Existing Vast.ai deployment (if you want to keep it)

### What Gets Added
- ➕ `Dockerfile.triton` (new Triton backend)
- ➕ `model_repository/` (Triton model files)
- ➕ `scripts/export_to_onnx.py` (ONNX conversion)
- ➕ `src/inference_service/client.py` - **EXTENDED** (add Triton support, keep PyTorch)
- ➕ `scripts/benchmark_backends.py` (compare PyTorch vs Triton)

### Deployment Options

```bash
# Option A: Continue using PyTorch (Step 3)
docker run --gpus all -p 8002:8002 \
  dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64

# Option B: Switch to Triton (Step 4)
docker run --gpus all -p 8000:8000 -p 8001:8001 \
  dahlianadkarni/photo-duplicate-triton:gpu-linux-amd64

# Option C: Switch to TensorRT (Step 5 - future)
docker run --gpus all -p 8002:8002 \
  dahlianadkarni/photo-duplicate-tensorrt:gpu-linux-amd64
```

### Development Workflow
1. **Keep Step 3 running** on Vast.ai if you want
2. **Test Step 4 locally** on a new Vast.ai instance
3. **Benchmark** both backends
4. **Keep both images** for easy switching/demos
5. **Destroy/recreate** Vast.ai instances as needed (each costs ~$0.20/hour)

4. **GPU Utilization**: Monitor `nvidia-smi` during tests
   ```bash
   # Triton should show higher GPU utilization
   ```

---

## Migration Strategy

### Week 1: Development
- Export model to ONNX
- Set up local Triton testing
- Verify inference correctness

### Week 2: Integration
- Update client code
- Add backend selection
- Test both backends

### Week 3: Deployment
- Deploy Triton on Vast.ai
- Run benchmarks
- CEasy Switching Between Steps

**For Demos or Development**

```bash
# Demo Step 3: PyTorch + FastAPI
vastai create instance <pytorch-image-id>
export INFERENCE_SERVICE_URL=http://<pytorch-ip>:8002
python src/ui/app.py

# Demo Step 4: Triton
vastai destroy instance <pytorch-instance-id>
vastai create instance <triton-image-id>
export INFERENCE_BACKEND=triton
export INFERENCE_SERVICE_URL=http://<triton-ip>:8000
python src/ui/app.py

# Go back to Step 3 anytime
vastai destroy instance <triton-instance-id>
vastai create instance <pytorch-image-id>
export INFERENCE_BACKEND=pytorch  # or unset it (default)
export INFERENCE_SERVICE_URL=http://<pytorch-ip>:8002
python src/ui/app.py
```

**Cost**: ~$0.20/hour per instance. Destroy when not in use.

## Rollback Plan

If Triton doesn't provide sufficient benefits:
- ✅ Keep using PyTorch backend (Step 3)
- ✅ All Step 3 code remains unchanged
- ✅ Triton code is isolated (no risk)
- ✅ Document Triton learnings for portfolio/interviews
- ✅ Easy to switch back: just change environment variable

---

## Success Criteria

- ✅ ONNX model matches PyTorch accuracy (within 0.1% difference)
- ✅ Triton server successfully deployed on GPU
- ✅ Client can switch between PyTorch and Triton backends
- ✅ Benchmarks show 20%+ throughput improvement with concurrent requests
- ✅ Dynamic batching reduces average latency under load

---

## Rollback Plan

If Triton doesn't provide sufficient benefits:
- Keep PyTorch backend (Step 3) as primary
- Document Triton learnings for portfolio
- Still valuable experience for job interviews

---

## Resources

- **Triton Documentation**: https://github.com/triton-inference-server/server
- **ONNX Runtime**: https://onnxruntime.ai/
- **Model Repository Guide**: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html
- **Dynamic Batching**: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher

---

## Next Steps

Run through the implementation:

```bash
# 1. Install dependencies
pip install onnx onnxruntime-gpu tritonclient[all]

# 2. Export model
python scripts/export_model_to_onnx.py

# 3. Test locally (CPU)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# 4. Test inference
python scripts/test_triton_inference.py

# 5. Deploy to GPU
./deploy_triton_gpu.sh
```

---

**Ready to start implementation?** Let me know which phase you'd like to begin with!
