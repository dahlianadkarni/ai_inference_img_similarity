# Benchmark Results (Template)

Use this file to summarize results from `scripts/benchmark_backends.py` runs.

- Triton results JSON: `benchmark_results/triton_<GPU>_<DATE>.json`
- PyTorch results JSON: `benchmark_results/pytorch_<GPU>_<DATE>.json`

-## Environment

-- Date: **2026-02-14**
-- Provider (e.g., Vast.ai): **Vast.ai**
-- Instance type / notes: **RTX A4000 (16GB VRAM), Remote GPU instances**
- Docker image tags / git commit: `dahlianadkarni/photo-duplicate-triton:gpu-linux-amd64`, `dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64`

-### GPU

-- Record from the JSON `gpu_specs` (and/or `nvidia-smi`).

-- - GPU model (Triton run): **RTX A4000** (Instance 31423333)
-- - GPU model (PyTorch run): **RTX A4000** (Instance 31424216)
- Driver version: **Not captured (remote instance)**
- VRAM total: **~8GB**

> Both runs used the same GPU model for fair comparison. Network latency from Mac to Vast.ai affects absolute numbers but relative comparison is valid.

## Benchmark Config

Paste from JSON `config` (must match for fair comparison):

- iterations: **50**
- batch sizes: **1, 4, 8, 16, 32**
- concurrency: **16**
- concurrent requests: **200**
- pytorch url: `http://142.112.39.215:50968`
- triton url: `http://142.112.39.215:5257`
- triton metrics url: `http://142.112.39.215:5354/metrics`

## Results Summary

Fill these from JSON fields:
- cold start: `*.cold_start.first_request_ms`
- single image: `*.single.{mean_ms,p50_ms,p95_ms,p99_ms}`
- batch: `*.batch_<N>.mean_ms` and derived images/sec
- concurrent: `*.concurrent.{throughput_rps,p95_ms}`

### Cold-start / First Request

| Metric | PyTorch | Triton | Notes |
|---|---:|---:|---|
| First request (ms) | **530** | 674 | PyTorch 21% faster |

### Single-image Latency

| Metric | PyTorch (ms) | Triton (ms) | Δ |
|---|---:|---:|---:|
| Mean | **207** | 724 | 3.5x slower |
| p50 | **162** | 564 | 3.5x slower |
| p95 | **360** | 1,875 | 5.2x slower |
| p99 | **892** | 2,032 | 2.3x slower |

### Batch Latency / Throughput

Compute images/sec as $\text{batch\_size} / (\text{mean\_ms}/1000)$.

| Batch | PyTorch mean (ms) | Triton mean (ms) | PyTorch img/s | Triton img/s | Winner |
|---:|---:|---:|---:|---:|---|
| 4 | **319** | 1,704 | **12.5** | 2.3 | PyTorch 5.4x faster |
| 8 | **534** | 2,610 | **15.0** | 3.1 | PyTorch 4.8x faster |
| 16 | **549** | 5,301 | **29.2** | 3.0 | PyTorch 9.7x faster |
| 32 | **729** | 9,649 | **43.9** | 3.3 | PyTorch 13.3x faster |

### Concurrency / Dynamic Batching

| Metric | PyTorch | Triton | Notes |
|---|---:|---:|---|
| Throughput (req/s) | **47.5** | 8.7 | PyTorch 5.5x faster |
| p95 latency (ms) | **502** | 4,495 | PyTorch 9x faster |
| Wall time (s) | **4.2** | 23.0 | PyTorch 5.5x faster |

### Triton Metrics (if captured)

From JSON `triton.server_metrics`.
 **459**
- queue_duration_us: **1,835,633** (1.84s total queue time)
- compute_infer_duration_us: **13,690,887** (13.69s total compute time)
- compute_input_duration_us: **1,566,833** (1.57s input processing)
- compute_output_duration_us: **41,357** (0.04s output processing)

## Observations / Trade-offs (writeup-ready)

### Model loading / cold start

- **PyTorch**: 530ms first request (21% faster)
- **Triton**: 674ms first request (includes ONNX graph optimization)
- **Verdict**: Comparable cold-start; PyTorch slightly faster

### Latency (single image)

- **PyTorch dominates**: 3.5x faster mean latency (207ms vs 724ms)
- **p95 latency**: PyTorch 5.2x faster (360ms vs 1,875ms)
- **Root cause**: ONNX Runtime overhead + network serialization adds latency
- **Network impact**: ~150-200ms round-trip to Vast.ai affects both but proportionally hurts slower backend more

### Throughput & dynamic batching

- **Batch processing**: PyTorch dramatically faster across all batch sizes
  - Batch-32: PyTorch 43.9 img/s vs Triton 3.3 img/s (13.3x difference!)
- **Concurrent throughput**: PyTorch 47.5 req/s vs Triton 8.7 req/s (5.5x difference)
- **Dynamic batching**: Triton's advantage did NOT materialize because:
  - ONNX Runtime backend slower than native PyTorch for this model
  - Network latency dominates request time (~500ms)
  - Queue delays (5ms) don't help when requests are sequential from single client
- **Verdict**: Native PyTorch significantly faster for this workload

### GPU utilization

- GPU metrics not captured (remote instance; nvidia-smi unavailable from local benchmark)
- Triton metrics show: ~13.7s compute time across 459 requests = ~30ms/request GPU compute
- Suggests GPU is underutilized; network and serialization dominate latency

### Cost / ops notes

- **Cost efficiency**: PyTorch 5.5x higher throughput = 82% lower cost per image
- **For 10,000 images**:
  - PyTorch: ~3.5 minutes @ 47.5 req/s
  - Triton: ~19 minutes @ 8.7 req/s
- **Operational simplicity**: PyTorch requires fewer dependencies (no ONNX export/conversion)
- **When to use Triton**:
  - Multi-model serving (not relevant here)
  - TensorRT optimization (could close the gap significantly)
  - Production-grade metrics/observability (Prometheus built-in)
  - Model versioning and hot-swapping
- **Recommendation for this use case**: Stick with PyTorch unless you need Triton's production features
- 

## Quick extraction helpers

These snippets help pull key numbers from a JSON file.

```bash
python - <<'PY'
import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else None
if not p:
    raise SystemExit('usage: python - <path/to/results.json>')
d = json.load(open(p))
backend = 'triton' if 'triton' in d else 'pytorch'
if backend not in d:
    raise SystemExit('No backend section found')

b = d[backend]
print('backend:', backend)
print('gpu_specs:', d.get('gpu_specs'))
print('cold_start_ms:', b.get('cold_start', {}).get('first_request_ms'))
print('single_p50_ms:', b.get('single', {}).get('p50_ms'))
print('single_p95_ms:', b.get('single', {}).get('p95_ms'))
print('concurrent_rps:', b.get('concurrent', {}).get('throughput_rps'))
print('concurrent_p95_ms:', b.get('concurrent', {}).get('p95_ms'))
PY
```
