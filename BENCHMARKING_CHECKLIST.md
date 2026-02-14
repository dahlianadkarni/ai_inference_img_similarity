# Benchmarking Checklist (PyTorch vs Triton)

Goal: produce **comparable** benchmark JSON files for Triton and PyTorch on the **same GPU model** (or document the difference), capturing:
- cold-start / first-request latency
- single-image latency percentiles
- batch latency by batch size
- concurrency/throughput (dynamic batching impact)
- GPU specs + utilization snapshot (`nvidia-smi`, if available)
- Triton Prometheus metrics (if available)

This repo’s benchmark runner: `scripts/benchmark_backends.py`
- Always prints to stdout
- Always saves JSON results (default under `./benchmark_results/`)

---

## 0) Decide “fair comparison” settings

- [ ] Use the **same GPU model** for both runs (ideal)
- [ ] If not possible, record GPU model + VRAM for each run and **do not compare costs/throughput directly**
- [ ] Keep the same benchmark parameters for both runs:
  - [ ] `--iterations`
  - [ ] `--batch-sizes`
  - [ ] `--concurrency`
  - [ ] `--concurrent-requests`

Recommended baseline (adjust as needed):
- iterations: `50`
- batch sizes: `1,4,8,16,32`
- concurrency: `16`
- concurrent requests: `200`

---

## 1) Remote machine prep

**Access your container:**
- [ ] Go to Vast.ai instance page → Click "CONNECT" → "Open Terminal" (web-based shell)
  - This connects directly to your running container with GPU access enabled
  - No SSH setup needed; Vast.ai automatically configures `--gpus all`

**Verify GPU is accessible:**
- [ ] Run `nvidia-smi` in the web terminal to confirm GPU is visible
  - Should show GPU model, driver version, and VRAM

Commands:
```bash
nvidia-smi
# Should show your RTX 3070 with ~8GB VRAM
```

- [ ] Get the repo on the machine and set up Python env (if running benchmarks from host)

Example:
```bash
git clone <your-repo-url>
cd inference_1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Triton run (do this first)

### 2.1 Start Triton

- [ ] Build / pull the Triton image (whatever workflow you’re using)
- [ ] Start Triton container with ports exposed
- [ ] Wait for readiness

Quick readiness checks:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8003/v2/health/ready
curl -s http://localhost:8003/v2/models/openclip_vit_b32 | head
```

If you have metrics enabled:
```bash
curl -s http://localhost:8005/metrics | head
```

### 2.2 Run Triton benchmark

- [ ] Run benchmark (Triton only)
- [ ] Save output JSON (either default path or explicit)

Recommended command (explicit output filename):
```bash
source venv/bin/activate
python scripts/benchmark_backends.py \
  --backend triton \
  --triton-url http://localhost:8003 \
  --triton-metrics-url http://localhost:8005/metrics \
  --iterations 50 \
  --batch-sizes 1,4,8,16,32 \
  --concurrency 16 \
  --concurrent-requests 200 \
  --output benchmark_results/triton_<GPU>_<DATE>.json
```

- [ ] Verify the JSON file exists
```bash
ls -lh benchmark_results | tail
```

- [ ] (Optional) Snapshot GPU state right after the run
```bash
nvidia-smi
```

---

## 3) PyTorch run (same GPU box)

### 3.1 Stop Triton (to avoid resource contention)

- [ ] Stop Triton container
- [ ] Confirm ports are free (optional)

---

### 3.2 Start PyTorch inference service

- [ ] Start the PyTorch FastAPI server (either Docker or `python -m ...`)
- [ ] Confirm it’s healthy

Health check:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8002/health
curl -s http://localhost:8002/model-info | head
```

---

### 3.3 Run PyTorch benchmark

- [ ] Run benchmark (PyTorch only)
- [ ] Save output JSON

Recommended command (explicit output filename):
```bash
source venv/bin/activate
python scripts/benchmark_backends.py \
  --backend pytorch \
  --pytorch-url http://localhost:8002 \
  --iterations 50 \
  --batch-sizes 1,4,8,16,32 \
  --concurrency 16 \
  --concurrent-requests 200 \
  --output benchmark_results/pytorch_<GPU>_<DATE>.json
```

- [ ] Verify the JSON file exists
```bash
ls -lh benchmark_results | tail
```

---

## 4) Sanity checks on results

- [ ] Confirm both JSON files include `gpu_specs` (on GPU boxes) and expected sections:
  - [ ] `pytorch.single`, `pytorch.concurrent`, `pytorch.batch_*`
  - [ ] `triton.single`, `triton.concurrent`, `triton.batch_*`, `triton.server_metrics` (if metrics reachable)

Quick peek:
```bash
python -c "import json; p='benchmark_results/triton_<GPU>_<DATE>.json'; d=json.load(open(p)); print(d.keys()); print('gpu_specs' in d, 'triton' in d)"
```

- [ ] If GPU models differ between runs, record that fact prominently before interpreting comparisons

---

## 5) Record the trade-offs (for resume / doc writeup)

- [ ] Loading / first request: compare `cold_start.first_request_ms`
- [ ] Single-image latency: compare `single.p50_ms` and `single.p95_ms`
- [ ] Batch scaling: compare `batch_4/8/16/32 mean_ms` and images/sec
- [ ] Concurrency: compare `concurrent.throughput_rps` and `concurrent.p95_ms`
- [ ] Utilization: compare `gpu_before/gpu_after` (if captured)
- [ ] Triton-only: note `server_metrics` queue/compute durations and how they change under concurrency

---

## Notes

- Triton dynamic batching shows its biggest win when you send **many concurrent single-image requests**, not just a single large batch request.
- For the cleanest comparison, run Triton and PyTorch **separately** (not simultaneously) on the same box.
