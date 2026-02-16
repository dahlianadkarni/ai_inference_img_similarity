# Apples-to-Apples Multi-GPU Comparison Guidance

**Goal:** Ensure your 4x RTX 4080 multi-GPU results are directly comparable to previous single-GPU and A100 benchmarks.

---

## Why 4x RTX 4080?
- You have already benchmarked both A100 and RTX 4080 single-GPU results.
- 4x RTX 4080 is cost-effective and widely available.
- Provides a true apples-to-apples scaling study using the same model, batch sizes, and scripts.
- PCIe Gen4 interconnect is typical for 4080s; this is a realistic production scenario.

## How to Ensure a Fair Comparison

1. **Use the Same Model and Config:**
   - Use the same ONNX model and config.pbtxt as in previous single-GPU tests.
   - For multi-GPU, use only the 4x config: `config.multigpu.4x.pbtxt`.

2. **Match Batch Sizes and Concurrency:**
   - Use the same `max_batch_size` and `preferred_batch_size` as before.
   - Run benchmarks at the same concurrency levels (e.g., 1, 4, 8, 16, 32, 64, 128).

3. **Use the Updated Benchmark Script:**
   - Always pass `--gpu-name rtx4080` when running `benchmark_multigpu.py`.
   - This ensures result files are named with the GPU type for clarity.

4. **Document All Settings:**
   - Record instance specs (GPU model, VRAM, interconnect, vCPU, RAM).
   - Note any differences in Docker image, driver, or Triton version.

5. **Repeat for A100 if Needed:**
   - If you want to compare scaling efficiency, repeat the same process on a 4x A100 SXM4 instance.
   - Use `--gpu-name a100` for those runs.

## What to Look For in Results

- **Scaling Efficiency:**
  - Does 4x RTX 4080 achieve close to 4x the throughput of single RTX 4080?
  - Compare with 4x A100 scaling if available.

- **Bottlenecks:**
  - Is scaling limited by PCIe bandwidth, CPU, or network?
  - Are queue times or compute times the limiting factor?

- **Cost-Efficiency:**
  - Calculate $/1000 images for each config.
  - Is 4x RTX 4080 more cost-effective than 4x A100?

## Example Result File Names

- `benchmark_results/step6b_rtx4080_4gpu_20260215_153000.json`
- `benchmark_results/step6b_a100_4gpu_20260215_170000.json`

## Final Recommendation

- Start with 4x RTX 4080 for your main scaling study.
- Only run 4x A100 if you want to compare NVLink vs PCIe scaling or need maximum throughput.
- Always use the `--gpu-name` flag for clarity in your results and analysis.

---

**Ready to benchmark?**
- Use the checklist in [STEP_6B_CHECKLIST.md](STEP_6B_CHECKLIST.md)
- Run: `python scripts/benchmark_multigpu.py --gpu-name rtx4080 ...`
- Analyze: `python scripts/analyze_multigpu_results.py ...`
