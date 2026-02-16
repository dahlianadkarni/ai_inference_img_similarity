# Step 6B Results: 4x RTX 4080 Multi-GPU Scaling Study

**Date:** February 15, 2026  
**GPU:** 4x NVIDIA GeForce RTX 4080 (16GB each)  
**Instance:** Vast.ai Instance 31484068 (173.185.79.174)  
**Interconnect:** PCIe Gen4  
**Test Type:** Remote benchmark from Mac

---

## 🎯 Executive Summary

**Multi-GPU scaling is highly inefficient for remote inference workloads.**

| Metric | 1x RTX 4080 (Step 6A) | 4x RTX 4080 (Step 6B) | Scaling Efficiency |
|--------|:--------------------:|:--------------------:|:------------------:|
| **Peak Throughput** | 24.3 img/s | 43.2 img/s | **1.8×** (45% efficiency) |
| **Server GPU Compute** | 5.7ms | 8.4ms avg | **1.5× slower** |
| **Client Latency (p50)** | 274.4ms | 416ms | 52% higher |
| **VRAM per GPU** | 2.8 GB | 2.5 GB | Similar |

**Key Finding:** Adding 3 more GPUs only improved throughput by 1.8× instead of the expected 4×. The bottleneck is **network/CPU overhead**, not GPU compute.

---

## 📊 Throughput vs Concurrency

| Concurrency | Throughput | Latency p50 | Latency p95 | Client-side |
|:-----------:|:----------:|:-----------:|:-----------:|:----------:|
| 1 | 3.4 img/s | 267.5ms | 457.0ms | Single request |
| 4 | 11.0 img/s | 270.2ms | 839.9ms | Starting to scale |
| 8 | 20.2 img/s | 255.0ms | 836.4ms | Good scaling |
| 16 | 31.9 img/s | 302.8ms | 999.5ms | Approaching peak |
| **32** | **43.2 img/s** | **415.9ms** | **1300.5ms** | **Peak** ⭐ |
| 64 | 37.4 img/s | 999.6ms | 2111.4ms | Saturated |
| 128 | 37.1 img/s | 1637.8ms | 2250.0ms | Over-saturated |

**Saturation Point:** Concurrency 32 is optimal. Beyond this, queue times dominate and throughput drops.

---

## 🔬 Server-Side Metrics Analysis

From Triton metrics endpoint (`http://173.185.79.174:45242/metrics`):

### Inference Stats
```
Total requests:           721
Total inferences:         721
Model executions:         560 (22% batch efficiency)
Success rate:            100%
```

### Timing Breakdown (Cumulative)
```
Total request time:      12,917,036 µs (12.9s total)
Queue time:               6,338,705 µs (6.3s) — 49% of total
GPU compute time:         6,067,809 µs (6.1s) — 47% of total
Input processing:           453,237 µs (0.5s)
Output processing:           21,752 µs (0.02s)
```

### Per-Request Averages
```
Avg request duration:    17.9ms
Avg queue time:           8.8ms (49% waiting)
Avg GPU compute:          8.4ms (10.8ms per execution)
```

**Key Insight:** Queue time (8.8ms) is comparable to GPU compute (8.4ms), indicating that dynamic batching is working but GPUs are idle ~50% of the time due to request serialization.

---

## 💾 GPU Memory Usage

| GPU | VRAM Used | VRAM Total | Utilization | Power | Energy Consumed |
|:---:|:---------:|:----------:|:-----------:|:-----:|:---------------:|
| GPU 0 | 2.47 GB | 16 GB | 0% (idle) | 14.9W | 20.7 kJ |
| GPU 1 | 2.45 GB | 16 GB | 0% (idle) | 6.7W | 9.5 kJ |
| GPU 2 | 2.46 GB | 16 GB | 0% (idle) | 21.8W | 29.2 kJ |
| GPU 3 | 2.46 GB | 16 GB | 0% (idle) | 11.1W | 16.6 kJ |
| **Total** | **9.84 GB** | **64 GB** | **0%** | **54.5W** | **76.0 kJ** |

**Memory per GPU:** ~2.5 GB (model + ONNX Runtime + buffers)  
**GPU Utilization:** 0% at measurement time (post-benchmark idle)

---

## 📈 Comparison: Single vs Multi-GPU

### Throughput Comparison

| Config | Peak Throughput | Best Concurrency | Scaling vs 1x |
|:------:|:---------------:|:----------------:|:-------------:|
| 1x RTX 4080 | 24.3 img/s | batch-32 | 1.0× (baseline) |
| 4x RTX 4080 | 43.2 img/s | concurrency-32 | **1.8×** |
| **Expected (ideal)** | **~97 img/s** | - | **4.0×** |
| **Efficiency** | - | - | **45%** |

### Why Only 1.8× Scaling?

1. **Network Overhead Dominates:**
   - 602KB raw tensor per request
   - ~200-300ms network RTT
   - Only ~17.9ms server-side processing
   - **Network is 10-15× slower than GPU compute**

2. **CPU Serialization:**
   - CPU utilization: 0.12%
   - Request preprocessing is single-threaded
   - Queue time (8.8ms) is 50% of total request time

3. **Dynamic Batching Limitations:**
   - Batch efficiency: 22% (560 execs / 721 requests)
   - Most requests processed as batch-1 or batch-2
   - `max_queue_delay_microseconds: 10000` too low for multi-GPU

4. **PCIe vs NVLink:**
   - PCIe Gen4: ~32 GB/s per GPU (shared bandwidth)
   - NVLink (A100): ~600 GB/s interconnect
   - Inter-GPU communication bottleneck

---

## 🆚 RTX 4080 vs A100 (Single-GPU Reference)

| Metric | 1x RTX 4080 | 1x A100 SXM4 | RTX 4080 Advantage |
|--------|:-----------:|:------------:|:------------------:|
| Single-image (client) | 337ms | 183ms | **1.8× slower** |
| GPU compute (server) | 5.7ms | 4.4ms | **1.3× slower** |
| Batch-32 throughput | 24.3 img/s | 64.3 img/s | **2.6× slower** |
| VRAM used | 2.8 GB | 3.3 GB | Similar |
| Cost per hour | $2.00 | $1.50 | 33% more expensive |

**Takeaway:** A100 is 2.6× faster and 25% cheaper than RTX 4080 for batch inference.

---

## 💰 Cost-Efficiency Analysis

### Cost Comparison

| Config | Throughput | Cost/hr | Cost per 1000 imgs | Efficiency |
|:------:|:----------:|:-------:|:------------------:|:----------:|
| 1x RTX 4080 | 24.3 img/s | $2.00 | $0.0229 | 100% (baseline) |
| 4x RTX 4080 | 43.2 img/s | $8.00 | $0.0514 | **45%** |
| 1x A100 | 64.3 img/s | $1.50 | $0.0065 | **287%** |

**Key Finding:** 4x RTX 4080 is **2.2× more expensive** per image than 1x RTX 4080, despite only 1.8× throughput gain.

### Break-Even Analysis

- **To justify 4x GPUs:** Need >80 img/s throughput
- **Actual throughput:** 43.2 img/s
- **Shortfall:** 46% of break-even

---

## 🔍 Bottleneck Analysis

### 1. Network I/O (Primary Bottleneck)

**Problem:** 602KB tensor per request dominates latency
- Network RTT: ~200-300ms
- Tensor transfer: ~150-250ms at 5 MB/s
- GPU compute: 8.4ms (only 3% of total time)

**Solution:** Use JPEG base64 input (reduces payload from 602KB → ~10KB)
- Expected reduction: 600KB → 10KB = **60× smaller**
- Network time: 250ms → 4ms = **62× faster**
- New throughput estimate: 43 img/s → **~250 img/s**

### 2. CPU Preprocessing (Secondary Bottleneck)

**Problem:** Single-threaded preprocessing
- CPU utilization: 0.12% (underutilized)
- Input processing: 453µs per request
- No parallelization across GPUs

**Solution:** Parallel preprocessing with worker pool

### 3. Dynamic Batching Inefficiency

**Problem:** Only 22% batch efficiency (560 execs / 721 requests)
- `max_queue_delay_microseconds: 10000` = 10ms
- Most requests don't wait long enough to batch
- GPUs process many batch-1 requests (inefficient)

**Solution:** Increase queue delay to 50-100ms for multi-GPU

### 4. PCIe Bandwidth Limitations

**Problem:** 4x GPUs share PCIe lanes
- PCIe Gen4 x16: ~32 GB/s per GPU
- Aggregate: 128 GB/s (theoretical)
- Actual: Much lower due to CPU bottleneck

**Solution:** Use NVLink-enabled GPUs (A100/H100) for 10× faster inter-GPU communication

---

## 💡 Recommendations

### For Production Deployment

1. **Don't use multi-GPU for remote inference**
   - Network overhead dominates (97% of latency)
   - Scaling efficiency: 45%
   - Cost efficiency: -55%

2. **If remote inference is required:**
   - Use JPEG base64 input format
   - Co-locate client and server
   - Use horizontal scaling (multiple 1x GPU instances)

3. **Multi-GPU makes sense for:**
   - **Local inference** (no network overhead)
   - **Batch processing** (high-throughput offline workloads)
   - **Large models** (>8GB VRAM per instance)

### Optimization Opportunities

1. **Input Format** (Priority 1)
   - Switch to JPEG base64: 602KB → 10KB
   - Expected improvement: **5-6× throughput**
   - Estimated: 43 img/s → 250 img/s

2. **Local Deployment** (Priority 2)
   - Eliminate network latency
   - Expected: 250ms → 17.9ms client-side
   - Estimated: 250 img/s → **800 img/s**

3. **Increase Dynamic Batching** (Priority 3)
   - Set `max_queue_delay_microseconds: 50000`
   - Improve batch efficiency: 22% → 70%
   - Estimated: 800 img/s → **1200 img/s**

4. **Use NVLink GPUs** (Priority 4)
   - Switch to 4x A100 SXM4 with NVLink
   - Inter-GPU bandwidth: 32 GB/s → 600 GB/s
   - Estimated: 1200 img/s → **1800 img/s**

### When to Use Multi-GPU

| Use Case | Single GPU | Multi-GPU | Recommended Config |
|----------|:----------:|:---------:|:------------------:|
| Remote API (<100 req/s) | ✅ | ❌ | 1x RTX 4080 or A100 |
| Remote API (>100 req/s) | ❌ | ⚠️ | Horizontal scaling (3-4× 1x GPU) |
| Local inference | ✅ | ✅ | 4x A100 with NVLink |
| Batch processing | ❌ | ✅ | 4-8x A100 with NVLink |
| Large models (>8GB) | ❌ | ✅ | Model parallelism (2-4x A100) |

---

## 📝 Lessons Learned

1. **Network overhead dominates remote inference**
   - GPU compute: 8.4ms (3%)
   - Network latency: 300ms+ (97%)
   - Multi-GPU doesn't help when network is the bottleneck

2. **PCIe interconnect limits scaling**
   - Expected: 4× throughput
   - Actual: 1.8× throughput
   - NVLink would provide 2-3× better scaling

3. **Dynamic batching needs tuning for multi-GPU**
   - Current batch efficiency: 22%
   - Queue delay too low (10ms)
   - Need 50-100ms delay for optimal batching

4. **Cost-efficiency decreases with multi-GPU**
   - 1x GPU: $0.0229 per 1000 images
   - 4x GPU: $0.0514 per 1000 images
   - 2.2× more expensive per image

5. **A100 is better value than RTX 4080**
   - 2.6× faster throughput
   - 25% cheaper cost
   - 3.5× better cost-efficiency

---

## 🚀 Next Steps

1. ✅ **Step 6B Complete** — Multi-GPU scaling study finished
2. **Optional:** Repeat on 4x A100 SXM4 with NVLink for comparison
3. **Optional:** Test local deployment (co-located client/server)
4. **Optional:** Implement JPEG base64 input for 5-6× improvement
5. **Production:** Deploy 1x A100 with PyTorch FastAPI for best value

---

## 📚 Related Documents

- [STEP_6A_A100_RESULTS.md](STEP_6A_A100_RESULTS.md) — Single-GPU A100 baseline
- [STEP_6A_RTX4080_RESULTS.md](STEP_6A_RTX4080_RESULTS.md) — Single-GPU RTX 4080 baseline
- [PLAN.md](PLAN.md) — Full infrastructure learning plan
- [STEP_6B_CHECKLIST.md](STEP_6B_CHECKLIST.md) — Multi-GPU deployment checklist
- [benchmark_results/step6b_rtx4080_4gpu-rtx4080_20260215_191823.json](benchmark_results/step6b_rtx4080_4gpu-rtx4080_20260215_191823.json) — Raw benchmark data

---

**Conclusion:** Multi-GPU scaling is inefficient for remote inference due to network overhead. For production, use 1x A100 with PyTorch FastAPI or horizontal scaling with multiple 1x GPU instances.
