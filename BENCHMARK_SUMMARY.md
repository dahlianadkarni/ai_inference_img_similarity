# Benchmark Summary: PyTorch vs Triton (RTX A4000)

**Date**: February 14, 2026  
**GPU**: NVIDIA RTX A4000 (16GB VRAM)  
**Provider**: Vast.ai  
**Test**: OpenCLIP ViT-B-32 embeddings (512-dim output)

---

## Executive Summary

**Winner: PyTorch** by a significant margin across all metrics.

PyTorch outperformed Triton by:
- **5.5x higher throughput** (47.5 vs 8.7 req/s)
- **3.5x lower latency** (207ms vs 724ms mean)
- **13.3x faster batch processing** (43.9 vs 3.3 img/s for batch-32)
- **82% lower cost per image**

---

## Key Metrics Comparison

### Latency

| Metric | PyTorch | Triton | Speedup |
|--------|---------|--------|---------|
| Cold-start | 530ms | 674ms | **1.3x** |
| Single p50 | 162ms | 564ms | **3.5x** |
| Single p95 | 360ms | 1,875ms | **5.2x** |
| Batch-32 mean | 729ms | 9,649ms | **13.2x** |

### Throughput

| Workload | PyTorch | Triton | Speedup |
|----------|---------|--------|---------|
| Concurrent (req/s) | 47.5 | 8.7 | **5.5x** |
| Batch-32 (img/s) | 43.9 | 3.3 | **13.3x** |

---

## Why PyTorch Won

### 1. **Native PyTorch is faster than ONNX Runtime**
- No conversion overhead
- Direct tensor operations optimized for GPU
- ONNX Runtime adds serialization/deserialization costs

### 2. **Network latency dominates**
- ~150-200ms round-trip from local Mac to Vast.ai
- Affects both backends but proportionally hurts slower one more
- Triton's queue delays (5ms) are negligible compared to network latency

### 3. **Batch processing highly optimized in PyTorch**
- Native batch operations on GPU
- Minimal overhead between batch sizes
- ONNX Runtime struggles with larger batches

### 4. **Dynamic batching didn't help Triton**
- Requires concurrent requests from multiple clients
- Single-client sequential testing doesn't trigger batching
- Network latency prevents queue from filling up

---

## When to Use Each Backend

### Use **PyTorch** when:
✅ Maximum performance is critical  
✅ Single model serving  
✅ Operational simplicity matters  
✅ Cost per inference is a concern  
✅ You control the deployment environment

### Use **Triton** when:
✅ Serving multiple models (GPU sharing)  
✅ Need TensorRT optimization (can close the gap)  
✅ Production-grade observability required (Prometheus metrics)  
✅ Model versioning and hot-swapping needed  
✅ Truly concurrent workloads (multi-client)  
✅ Language-agnostic inference (gRPC/HTTP)

---

## Cost Analysis


## Actual Cost (One-Time Run)

| Backend   | Total Cost (one run) |
|-----------|----------------------|
| PyTorch   | $0.092               |
| Triton    | $0.092               |

> Both benchmarks were run on the same RTX A4000 instance. Total cost for the full benchmarking session (including both backends) was **$0.092** (Vast.ai spot pricing, February 2026).

**Note:** For cost-per-image or per-1K image analysis, divide by the number of images processed in your run.

---

## Recommendations

### For This Project (Photo Duplicate Detection)
**Use PyTorch backend** because:
- Single model deployment
- Batch processing is common (scan entire photo library)
- Cost efficiency matters for personal project
- Simpler deployment (no ONNX conversion step)

### Future Optimizations
If you need to use Triton for production:
1. **Try TensorRT backend** (could be 2-3x faster than ONNX)
2. **Optimize network latency** (deploy closer to clients)
3. **Use true concurrent clients** to benefit from dynamic batching
4. **Tune batch sizes** and queue delays for your workload

---

## Files

- Full results: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)
- Raw data:
  - `benchmark_results/pytorch_rtx3070_20260214_090742.json`
  - `benchmark_results/triton_rtx3070_20260214_083702.json`
- Visualization: `benchmark_results/comparison_chart.png` (generated)
