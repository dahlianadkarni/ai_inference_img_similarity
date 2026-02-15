# GPU Verification Guide

## Problem: 0% GPU Usage on Vast.ai Dashboard

If you're seeing 0% GPU usage during benchmarks, the model might be running on CPU instead of GPU.

## Quick Verification Steps

### Option 1: Use the GPU Info Endpoint (After Redeploying)

After deploying the updated server code:

```bash
# Check PyTorch instance
curl http://142.112.39.215:50912/gpu-info | python3 -m json.tool

# Check Triton instance (if updated with similar endpoint)
curl http://142.112.39.215:50919/gpu-info | python3 -m json.tool
```

**Expected Output (GPU working)**:
```json
{
  "cuda_available": true,
  "cuda_version": "12.4",
  "device_count": 1,
  "devices": [
    {
      "id": 0,
      "name": "NVIDIA RTX 3090",
      "total_memory_gb": 24.0,
      "current_device": true
    }
  ],
  "model_device": "cuda:0",
  "model_on_gpu": true
}
```

**Bad Output (CPU fallback)**:
```json
{
  "cuda_available": false,
  "device_count": 0,
  "model_on_gpu": false,
  "warning": "CUDA not available - running on CPU"
}
```

### Option 2: Use the Verification Script

```bash
python verify_gpu.py http://142.112.39.215:50912
```

### Option 3: SSH into Container (If Port 22 Exposed)

```bash
# Get SSH port from Vast.ai dashboard
ssh root@142.112.39.215 -p <SSH_PORT>

# Inside container, check:
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Check nvidia-smi
nvidia-smi
```

---

## Common Issues & Fixes

### Issue 1: CUDA Not Available in Container

**Symptoms**:
- `cuda_available: false`
- Model reports `device: cpu`

**Causes**:
- Container not started with `--gpus all` flag
- Base image doesn't have CUDA support
- NVIDIA container runtime not configured

**Fix for Vast.ai**:
Vast.ai automatically adds `--gpus all` when you select a GPU instance. Verify:
1. Instance shows GPU in specs
2. Docker image uses CUDA base (`nvidia/cuda:12.4.0-runtime-ubuntu22.04`)

### Issue 2: Model on CPU Despite CUDA Available

**Symptoms**:
- `cuda_available: true`
- But `model_on_gpu: false`

**Causes**:
- Code explicitly sets `device='cpu'`
- CUDA_VISIBLE_DEVICES="-1"
- Model failed to move to GPU silently

**Fix**:
Check server logs at startup for GPU verification messages:
```
GPU Configuration:
  CUDA Available: YES
  CUDA Version: 12.4
  Device Count: 1
  GPU 0: NVIDIA RTX 3090 (24.00 GB)
Model loaded on device: cuda:0
✓ Model successfully loaded on GPU
```

### Issue 3: 0% GPU Usage But Model is on GPU

**Symptoms**:
- `model_on_gpu: true`
- GPU name shows correctly
- But Vast.ai dashboard shows 0%

**Possible Causes**:
1. **Sampling timing**: Vast.ai samples GPU% every few seconds. Brief inference bursts (168ms) might be missed.
2. **Network-bound**: Most time spent waiting for network requests, GPU idles between bursts.
3. **Low GPU utilization**: Small model + small batch doesn't fully utilize GPU.

**Verification**:
Run a sustained load test:
```bash
# In one terminal, watch GPU usage in real-time
ssh root@instance "watch -n 0.5 nvidia-smi"

# In another terminal, run continuous benchmark
python scripts/benchmark_backends.py \
  --backend pytorch \
  --pytorch-url http://142.112.39.215:50912 \
  --iterations 100 \
  --concurrent-requests 500
```

During the benchmark, you should see GPU% spike above 0% if GPU is actually being used.

---

## Redeploying with GPU Verification

To deploy the updated server with GPU verification:

### For PyTorch Service:

```bash
# Rebuild and push
docker buildx build --platform linux/amd64 \
  -f Dockerfile.gpu \
  -t dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64 \
  --push \
  .

# Restart Vast.ai instance or stop/start container
```

### For Triton Service:

Triton doesn't expose custom endpoints, but you can check logs:

```bash
# SSH into container
ssh root@instance "docker logs triton-container"

# Look for GPU initialization messages from Triton
```

---

## Next Steps if Model is on CPU

1. **Verify Dockerfile.gpu** has CUDA base image
2. **Check CUDA_VISIBLE_DEVICES** env var (should be 0, not -1)
3. **Review server logs** for CUDA initialization errors
4. **Test nvidia-smi** in container to verify driver access
5. **Redeploy** with `--gpus all` flag explicitly

---

## Understanding Vast.ai GPU Metrics

Vast.ai samples GPU metrics periodically (every 5-10 seconds). For workloads with:
- Brief inference bursts (< 1 second)
- Long idle periods between requests
- Network-bound operations

You may see:
- **0% GPU** most of the time
- **Spikes to 20-80%** during active inference
- **0% even during benchmarks** if sampling misses the bursts

This doesn't necessarily mean GPU isn't being used — verify with `/gpu-info` endpoint instead.
