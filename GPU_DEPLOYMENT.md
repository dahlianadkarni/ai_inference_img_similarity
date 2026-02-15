# GPU Deployment Guide (Step 3)

> **Step 3 Status: ✅ COMPLETE**  
> Successfully deployed containerized inference service on cloud GPU infrastructure.

## Overview

This guide covers deploying your Docker container to cloud GPU instances (Vast.ai, RunPod, Lambda Labs, etc.).

## Prerequisites

- Docker Hub account
- Docker with buildx support (for multi-platform builds)
- Cloud GPU provider account (Vast.ai, RunPod, Lambda Labs)

---

## Part 1: Build for linux/amd64

**Important:** Cloud GPU instances run on `linux/amd64`, not `linux/arm64` (Mac M1/M2).

### Build GPU Image

```bash
# Build for linux/amd64 platform
docker buildx build --platform linux/amd64 \
  -f Dockerfile.gpu \
  -t dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64 \
  .
```

### Push to Docker Hub

```bash
# Login to Docker Hub (one-time)
docker login

# Push the image
docker push dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64
```

**Image is now available at:**
```
dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64
```

---

## Part 2: Deploy on Vast.ai

### Step 1: Create Account & Add Payment

1. Go to https://vast.ai/
2. Sign up and add payment method
3. Navigate to "Console" → "Create"

### Step 2: Select GPU Instance

**Recommended specs:**
- **GPU:** RTX 3090, RTX 4090, A40, or A10 (anything with 16GB+ VRAM), eg: 1x RTX PRO 6000
- **CPU:** Any (AMD EPYC or Xeon preferred)
- **RAM:** 32GB+ recommended
- **Disk:** 30GB+ for container + model weights
- **Bandwidth:** Any

**Cost estimate:** $0.15-0.80/hour depending on GPU

### Step 3: Configure Container

1. **Image & Config:**
   - Select **"Edit Image & Config"**
   - **Docker Image:** `dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64`
   - **Launch Mode:** Docker

2. **Environment Variables:**
   Add these environment variables:
   ```
   HOST=0.0.0.0
   MODEL_NAME=ViT-B-32
   MODEL_PRETRAINED=openai
   LOG_LEVEL=info
   ```

3. **On-Start Script:** Leave empty (container auto-starts)

4. **Disk Space:** 30 GB minimum

5. **Ports:**
   - Expose port **8002** (inference API - usually auto-detected)
   - Expose port **22** (SSH access for debugging - optional but recommended)
   
   > **Note**: Ports must be specified at instance creation time. They cannot be added later to a running instance.

### Step 4: Launch Instance

1. Click **"Rent"**
2. Wait 2-3 minutes for instance to boot
3. Model loading takes 60-90 seconds after boot

### Step 5: Get Connection Details

Once running, you'll see:
```
Instance ID:       31226006
Public IP:         69.63.236.192
Instance Port:     26872 -> 8002/tcp
SSH Port:          27000
```

Your service URL:
```
http://<public-ip>:<mapped-port>
```

Example: `http://69.63.236.192:26872`

---

## Part 3: Test Deployment

### Health Check

```bash
curl http://<public-ip>:<port>/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "ViT-B-32",
  "device": "cuda"
}
```

### Model Info

```bash
curl http://<public-ip>:<port>/model-info
```

### API Documentation

Open in browser:
```
http://<public-ip>:<port>/docs
```

### Test Embedding Generation

```bash
curl -X POST http://<public-ip>:<port>/embed/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["<base64-encoded-image>"],
    "model_name": "ViT-B-32",
    "pretrained": "openai"
  }'
```

---

## Part 4: Connect UI Client

Point your local UI to the remote GPU service:

```bash
# On your local machine
export INFERENCE_SERVICE_URL=http://<public-ip>:<port>

# Start UI
python -m src.ui.main
```

Or edit the UI code to point to your GPU instance.

---

## Part 5: Monitor & Debug

### Check Container Logs (via SSH)

```bash
ssh -p <ssh-port> root@<public-ip>

# View logs
docker logs -f <container-id>

# Check GPU usage
nvidia-smi

# Check service status
curl http://localhost:8002/health
```

### Common Issues

#### ❌ Connection Refused
- **Cause:** Container not started or HOST=127.0.0.1
- **Fix:** Check logs, ensure HOST=0.0.0.0 in environment variables

#### ❌ 404 Not Found
- **Cause:** Wrong port mapping
- **Fix:** Check Vast.ai port mapping (e.g., 26872 -> 8002)

#### ❌ Out of Memory
- **Cause:** GPU VRAM exhausted
- **Fix:** Use smaller model or instance with more VRAM

#### ❌ Slow Response
- **Cause:** Model loading still in progress
- **Fix:** Wait 60-90 seconds after container starts

---

## Part 6: Cost Management

### Monitoring Costs

- Check Vast.ai dashboard for current spending
- Most instances: $0.15-0.80/hour
- 24 hours of testing ≈ $4-20

### Stopping Instance

**Always stop instances when not in use!**

```bash
# Via Vast.ai console
1. Go to "Instances"
2. Click "Destroy" on your instance

# Billing stops immediately
```

### Budget Tips

- Use spot instances (cheaper but can be interrupted)
- Prefer RTX 3090/4090 over A100 (similar performance, lower cost)
- Test locally on CPU first, then deploy to GPU
- Only run GPU inference for actual workload testing

---

## Part 7: Alternative Providers

### RunPod.io

Similar setup:
```
1. Go to https://www.runpod.io/
2. Select GPU pod
3. Use custom Docker image: dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64
4. Set HOST=0.0.0.0 in environment
5. Expose port 8002
```

### Lambda Labs

```
1. Go to https://lambdalabs.com/service/gpu-cloud
2. Launch instance (pre-installed Docker + CUDA)
3. SSH in and run:
   docker run -d -p 8002:8002 --gpus all \
     -e HOST=0.0.0.0 \
     dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64
```

---

## Architecture Comparison

### Local CPU (Step 1-2)
- **Device:** MacBook (MPS or CPU)
- **Speed:** ~2-5 seconds/image
- **Cost:** Free
- **Use case:** Development, testing

### Cloud GPU (Step 3)
- **Device:** NVIDIA T4/A10/RTX 6000 Ada/H100/L40S
- **Speed:** ~0.1-0.5 seconds/image (5-50x faster)
- **Cost:** $0.15-0.80/hour
- **Use case:** Production, batch processing

**GPU Compatibility:** Our Docker image (CUDA 12.4 + PyTorch cu124) supports:
- ✅ **Ada Lovelace:** RTX 6000 Ada, RTX 4090, L40/L40S (compute 8.9)
- ✅ **Hopper:** H100 (compute 9.0)
- ✅ **Ampere:** A100, A10, RTX 3090 (compute 8.0-8.6)
- ✅ **Turing:** T4, RTX Titan (compute 7.5)
- ✅ **Volta:** V100 (compute 7.0)

---

## Key Learnings from Step 3

✅ **Multi-platform builds are critical** for Mac → Cloud deployment  
✅ **HOST=0.0.0.0 is required** for external access in containers  
✅ **Port mapping** can differ from exposed ports (e.g., 26872 → 8002)  
✅ **Model loading takes time** (~60-90s on first request)  
✅ **GPU acceleration** provides 10-50x speedup for embedding generation  
✅ **Cost-effective testing** possible with pay-per-hour GPU instances  
✅ **CUDA 12.4 + PyTorch cu124** required for Ada/Hopper GPU support  

---

## Next Steps

- ✅ **Step 1-3 Complete:** Client-service architecture, containerization, GPU deployment
- 🔜 **Step 4:** Introduce inference framework (NVIDIA Triton Server)
- 🔜 **Step 5:** TensorRT optimization for production

---

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] Docker image built with `--platform linux/amd64`
- [ ] Image pushed to Docker Hub successfully
- [ ] Vast.ai instance shows "Running" status
- [ ] Port 8002 is exposed and mapped correctly
- [ ] Environment variable `HOST=0.0.0.0` is set
- [ ] Waited 60-90 seconds for model to load
- [ ] Health check returns 200 OK: `curl http://<ip>:<port>/health`
- [ ] Model-info shows `"device": "cuda"`: `curl http://<ip>:<port>/model-info`

### Common Issues

#### ❌ "CUDA error: no kernel image is available for execution on the device"
**Cause:** PyTorch doesn't have pre-compiled kernels for your GPU architecture (common with newer Ada/Hopper GPUs)  
**Solution:** Rebuild with CUDA 12.4 + PyTorch cu124 (already in updated Dockerfile.gpu)  
**How to verify:** Check Dockerfile.gpu uses `nvidia/cuda:12.4.0-runtime` and `--index-url https://download.pytorch.org/whl/cu124`

#### ❌ Connection refused
**Cause:** HOST=127.0.0.1 instead of 0.0.0.0  
**Solution:** Set environment variable `HOST=0.0.0.0` when launching instance

#### ❌ Service returns `"device": "cpu"` instead of `"device": "cuda"`
**Cause:** GPU not available to container  
**Solution:** Ensure instance has GPU and Docker has `--gpus all` flag (Vast.ai handles this automatically)

---

## Summary

You've successfully:
- Built a cross-platform Docker image for GPU deployment
- Deployed to cloud GPU infrastructure (Vast.ai)
- Tested remote inference service endpoints
- Achieved 10-50x speedup over CPU inference

**Step 3 Complete! 🎉**

See [PLAN.md](PLAN.md) for next steps (Triton Inference Server + TensorRT optimization).
