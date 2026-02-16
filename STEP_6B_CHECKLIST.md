# Step 6B: Multi-GPU Scaling Study - Implementation Checklist

**Date:** February 15, 2026  
**Status:** Ready to deploy  
**Prerequisites:** Step 6A complete (single-GPU baseline established)

---

## 🎯 Objectives

1. **Validate Triton's multi-GPU scaling** — Does 4x GPU = 4x throughput?
2. **Find saturation point** — When does adding GPUs stop helping?
3. **Cost-efficiency analysis** — What's the $/1000 images at each scale?
4. **Document bottlenecks** — Network, CPU, memory, or something else?

---

## 📋 Pre-Deployment Checklist

### 1. GPU Instance Selection
- [ ] Select 4x RTX 4080 instance (recommended for apples-to-apples with previous results)
- [ ] Optionally, test 4x A100 SXM4 for direct comparison
- [ ] Verify all GPUs are same model (homogeneous config)
- [ ] Check GPU interconnect type (PCIe Gen4 for RTX 4080, NVLink for A100)
- [ ] Verify CUDA driver version ≥ 12.0
- [ ] Check Docker GPU runtime is available

### 2. Config Preparation
- [ ] Review current single-GPU config ([openclip_vit_b32/config.pbtxt](model_repository/openclip_vit_b32/config.pbtxt))
- [ ] Use only the 4x config ([openclip_vit_b32/config.multigpu.4x.pbtxt](model_repository/openclip_vit_b32/config.multigpu.4x.pbtxt))
- [ ] Ensure `max_batch_size` and `preferred_batch_size` remain optimized

### 3. Docker Image & Deployment
- [ ] Build custom 4x GPU Docker image: `docker buildx build --platform linux/amd64 -f Dockerfile.step6b-4gpu -t dahlianadkarni/photo-duplicate-step6b-4gpu:latest --push .`
- [ ] Verify image is pushed to Docker Hub: `docker pull dahlianadkarni/photo-duplicate-step6b-4gpu:latest`
- [ ] Image includes pre-configured 4x GPU config.pbtxt (no manual config needed)
- [ ] Test health check endpoints before benchmarking

### 4. Benchmarking Setup
- [ ] Install benchmark script on local machine ([scripts/benchmark_multigpu.py](scripts/benchmark_multigpu.py))
- [ ] Prepare test image dataset (use existing synthetic images)
- [ ] Decide concurrency levels to test: [1, 2, 4, 8, 16, 32, 64, 128]
- [ ] Plan iteration counts: 100 iterations per config for statistical significance

### 5. Monitoring & Metrics
- [ ] Verify Triton metrics endpoint is accessible: `/metrics`
- [ ] Plan to capture per-GPU utilization via `nvidia-smi`
- [ ] Prepare to collect Triton queue times and compute times
- [ ] Set up method to monitor GPU memory usage over time

---

## 🚀 Deployment Steps

### Step 1: Provision Multi-GPU Instance
```bash
# Example: Vast.ai 4x RTX 4080 (PCIe Gen4)
# - Search for 4x RTX 4080 (16GB) instance (PCIe interconnect)
# - For direct A100 comparison, repeat with 4x A100 SXM4 (NVLink)
# - Select region with low network latency to your location
# - Ensure Docker is pre-installed
# - Note: GPU instances with NVLink are 2-3x faster for multi-GPU than PCIe
```

### Step 2: Deploy Triton Container (Single-GPU Baseline)
```bash
# SSH into instance
ssh root@<INSTANCE_IP>

# Pull Docker image (4x GPU pre-configured)
docker pull dahlianadkarni/photo-duplicate-step6b-4gpu:latest

# Start with 1 GPU first (baseline - will only use 1 of 4 instances)
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --name triton-4gpu \
  dahlianadkarni/photo-duplicate-step6b-4gpu:latest

# Wait 30s for model load
sleep 30

# Verify health
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/openclip_vit_b32/ready
```

### Step 3: Run Single-GPU Baseline Benchmark
```bash
# On local machine
python scripts/benchmark_multigpu.py \
  --triton-url http://<INSTANCE_IP>:8000 \
  --config-name "1-gpu-baseline" \
  --concurrency 1,2,4,8,16,32 \
  --iterations 100 \
  --output benchmark_results/step6b_1gpu_baseline.json
```

### Step 4: Run 4x GPU Benchmark
```bash
# The container is already configured for 4x GPUs
# Run benchmark from your local machine
python scripts/benchmark_multigpu.py \
  --triton-url http://<INSTANCE_IP>:8000 \
  --config-name "4-gpu-rtx4080" \
  --gpu-name rtx4080 \
  --concurrency 1,4,8,16,32,64,128 \
  --iterations 100 \
  --output benchmark_results/step6b_rtx4080_4gpu.json

# For A100 comparison (if testing on 4x A100 instance):
python scripts/benchmark_multigpu.py \
  --triton-url http://<INSTANCE_IP>:8000 \
  --config-name "4-gpu-a100" \
  --gpu-name a100 \
  --concurrency 1,4,8,16,32,64,128 \
  --iterations 100 \
  --output benchmark_results/step6b_a100_4gpu.json
```

### Step 5: Collect Triton Metrics
```bash
# During benchmarks, periodically capture metrics
curl http://<INSTANCE_IP>:8002/metrics > metrics_2gpu_c32.txt

# Key metrics to track:
# - nv_inference_request_success (total requests)
# - nv_inference_queue_duration_us (queue time)
# - nv_inference_compute_infer_duration_us (GPU compute time)
# - nv_gpu_utilization (per-GPU utilization)
# - nv_gpu_memory_used_bytes (per-GPU memory)
```

### Step 6: Capture GPU Utilization
```bash
# On remote instance during benchmark
watch -n 1 nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

# Or log to file
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 1 > gpu_utilization_4gpu.csv
```

---

## 📊 Analysis Steps

### 1. Compare Throughput Scaling
```python
# Use scripts/analyze_multigpu_results.py (created below)
python scripts/analyze_multigpu_results.py \
  --results benchmark_results/step6b_*.json \
  --output STEP_6B_RESULTS.md
```

**Expected Metrics:**
- Throughput (images/sec) at each concurrency level
- Latency (p50, p95, p99) vs concurrency
- GPU utilization per device
- Memory usage per device
- Cost per 1000 images (based on instance pricing)

### 2. Identify Scaling Efficiency
- **Linear scaling:** 4x GPU = 4x throughput?
- **Saturation point:** At what concurrency does adding GPUs stop helping?
- **Bottlenecks:** CPU preprocessing, network I/O, memory bandwidth?

### 3. Cost-Efficiency Analysis
```
Cost per 1000 images = (Instance $/hr ÷ Throughput images/sec) × 1000 ÷ 3600

Example:
- 1x A100: $1.50/hr, 200 img/s → $2.08 per 1000 images
- 4x A100: $5.00/hr, 700 img/s → $1.98 per 1000 images (5% better)
- 8x A100: $9.00/hr, 900 img/s → $2.78 per 1000 images (33% worse!)
```

### 4. Document Findings
- [ ] Update [STEP_6B_RESULTS.md](STEP_6B_RESULTS.md) with:
  - Throughput vs GPU count (table + chart)
  - Latency vs concurrency (per GPU config)
  - GPU utilization and memory usage
  - Cost-efficiency analysis
  - Recommended production config
  - When to use multi-GPU (saturation threshold)

---

## 🧪 Test Matrix

| GPU Count | Instance Group Config | Concurrency Levels | Expected Throughput* | Instance Cost/hr** |
|:---------:|:---------------------:|:------------------:|:--------------------:|:------------------:|
| **1x** | `count: 1, gpus: [0]` | 1, 2, 4, 8, 16, 32 | ~200-250 img/s | $1.50 |
| **2x** | `count: 2` | 1, 2, 4, 8, 16, 32, 64 | ~400-500 img/s | $2.50 |
| **4x** | `count: 4` | 1, 4, 8, 16, 32, 64, 128 | ~700-900 img/s | $5.00 |
| **8x** | `count: 8` | 1, 8, 16, 32, 64, 128, 256 | ~1200-1600 img/s | $9.00 |

*Expected throughput based on single-GPU 4.4ms compute time (227 img/s theoretical max per GPU)  
**Pricing is approximate for Vast.ai A100 SXM4 80GB instances as of Feb 2026

---

## ⚠️ Common Issues & Solutions

### Issue 1: Not All GPUs Utilized
**Symptoms:** Only GPU 0 shows high utilization, others idle  
**Solutions:**
- Verify config has `instance_group [ { kind: KIND_GPU, count: N } ]` (not `gpus: [0]`)
- Check Docker was started with `--gpus all`, not `--gpus 0`
- Increase concurrency to saturate all instances

### Issue 2: Throughput Doesn't Scale Linearly
**Symptoms:** 4x GPU = only 2.5x throughput  
**Root Causes:**
- **Network bottleneck:** Binary protocol overhead (602KB per image)
  - *Solution:* Use local benchmark on instance, or optimize to JPEG input
- **CPU bottleneck:** Preprocessing/deserialization on single CPU core
  - *Solution:* Increase instance vCPU count, or use DALI for preprocessing
- **PCIe bottleneck:** GPUs on PCIe Gen3 instead of NVLink
  - *Solution:* Use instances with NVLink (A100 SXM4, H100 SXM5)

### Issue 3: High Queue Times
**Symptoms:** `nv_inference_queue_duration_us` >> compute time  
**Solutions:**
- Increase `max_queue_delay_microseconds` to allow larger batches
- Reduce `preferred_batch_size` to process smaller batches faster
- Add more model instances if CPUs are available

### Issue 4: OOM (Out of Memory)
**Symptoms:** Triton crashes, CUDA OOM errors  
**Solutions:**
- Reduce `max_batch_size` from 32 to 16 or 8
- Use smaller `preferred_batch_size` values
- Ensure `--shm-size` is sufficient (2g for 4 GPUs, 4g for 8 GPUs)

---

## 📈 Success Criteria

- [ ] **Baseline established:** 1-GPU throughput matches Step 6A results (~200-250 img/s)
- [ ] **Scaling validated:** 4x GPU achieves ≥3.0x throughput improvement (≥600 img/s)
- [ ] **Saturation documented:** Found concurrency level where adding GPUs no longer helps
- [ ] **Cost analysis complete:** Calculated $/1000 images for each config
- [ ] **Bottlenecks identified:** CPU, network, memory, or GPU compute?
- [ ] **Production recommendation:** Clear guidance on when to use multi-GPU

---

## 🔗 Related Files

- **Config Templates:** [model_repository/openclip_vit_b32/config.multigpu.*.pbtxt](model_repository/openclip_vit_b32/)
- **Benchmark Script:** [scripts/benchmark_multigpu.py](scripts/benchmark_multigpu.py)
- **Analysis Script:** [scripts/analyze_multigpu_results.py](scripts/analyze_multigpu_results.py)
- **Results Document:** [STEP_6B_RESULTS.md](STEP_6B_RESULTS.md)
- **Previous Step:** [STEP_6A_A100_RESULTS.md](STEP_6A_A100_RESULTS.md)

---

## 💡 Pro Tips

1. **Start small:** Test 1-GPU baseline first to verify deployment works
2. **Incremental scaling:** Test 2x before 4x, 4x before 8x
3. **Monitor in real-time:** Keep `nvidia-smi` and `/metrics` open during benchmarks
4. **Save everything:** All results, metrics, logs — they tell the scaling story
5. **Network matters:** If remote, test localhost benchmark on instance for true GPU performance
6. **Cost vs performance:** More GPUs ≠ better value. Find the sweet spot.

---

**Ready to proceed?** Follow the deployment steps above and run benchmarks! 🚀
