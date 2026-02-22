# Presentation Materials

> All materials for presenting the ML Inference Infrastructure project.

---

## Quick Start

| Format | File | Best For |
|--------|------|----------|
| **🖥 Slide Deck** | [slides.html](slides.html) | Live presentation (open in browser, use arrow keys) |
| **📄 Executive Summary** | [01_EXECUTIVE_SUMMARY.md](01_EXECUTIVE_SUMMARY.md) | 5-minute overview, quick read |
| **📘 Technical Deep-Dive** | [02_TECHNICAL_DEEP_DIVE.md](02_TECHNICAL_DEEP_DIVE.md) | 20-30 minute walkthrough |
| **📐 Architecture Diagrams** | [03_ARCHITECTURE.md](03_ARCHITECTURE.md) | Visual reference, system evolution |
| **📊 Benchmark Data** | [04_BENCHMARK_RESULTS.md](04_BENCHMARK_RESULTS.md) | Complete numbers, all steps |

---

## How to Use

### For a Live Demo (10-15 min)
1. Open **slides.html** in a browser
2. Use ← → arrow keys (or swipe on touch devices)
3. 20 slides covering the full journey
4. Refer to the deep-dive doc for detailed Q&A

### For a Technical Interview (20-30 min)
1. Walk through **slides.html** for the overview
2. Switch to **02_TECHNICAL_DEEP_DIVE.md** for deep questions
3. Pull up **04_BENCHMARK_RESULTS.md** for specific numbers
4. Show the actual code repo for implementation details

### For Async Review (Read at Own Pace)
1. Start with **01_EXECUTIVE_SUMMARY.md** (2-3 min read)
2. Continue with **02_TECHNICAL_DEEP_DIVE.md** (10-15 min read)
3. Reference **03_ARCHITECTURE.md** and **04_BENCHMARK_RESULTS.md** as needed

---

## Project Structure Reference

```
presentation/
├── README.md                    ← You are here
├── slides.html                  ← Interactive HTML slide deck (20 slides)
├── 01_EXECUTIVE_SUMMARY.md      ← Quick overview + top findings
├── 02_TECHNICAL_DEEP_DIVE.md    ← Full technical walkthrough (Steps 1-8)
├── 03_ARCHITECTURE.md           ← Architecture diagrams (ASCII art)
└── 04_BENCHMARK_RESULTS.md      ← Complete benchmark data tables
```

### Related Documentation (in repo root)

| Doc | Contents |
|-----|----------|
| [PLAN.md](../PLAN.md) | 8-step learning plan with completion status |
| [STEP_5A_FINDINGS.md](../STEP_5A_FINDINGS.md) | ONNX Runtime optimization report |
| [STEP_5B_TENSORRT.md](../STEP_5B_TENSORRT.md) | TensorRT EP setup guide |
| [STEP_6A_A100_RESULTS.md](../STEP_6A_A100_RESULTS.md) | A100 3-way comparison |
| [STEP_6A_RTX4080_RESULTS.md](../STEP_6A_RTX4080_RESULTS.md) | RTX 4080 comparison |
| [K8S_PLAN.md](../K8S_PLAN.md) | Step 8: Kubernetes phased plan |
| [STEP_8_K8S_RESULTS.md](../STEP_8_K8S_RESULTS.md) | Step 8: kind cluster results, HPA load test, CPU baseline |
| [STEP_6B_RESULTS.md](../STEP_6B_RESULTS.md) | Multi-GPU scaling study |
| [STEP_7_GRPC_RESULTS_A100.md](../STEP_7_GRPC_RESULTS_A100.md) | 5-way gRPC vs HTTP, A100 Massachusetts |
| [STEP_7_GRPC_RESULTS_RTX_4090.md](../STEP_7_GRPC_RESULTS_RTX_4090.md) | 5-way gRPC vs HTTP, RTX 4090 Pennsylvania |
| [DOCKER_README.md](../DOCKER_README.md) | Docker containerization guide |
| [GPU_DEPLOYMENT.md](../GPU_DEPLOYMENT.md) | Cloud GPU deployment guide |
| [TRITON_SETUP.md](../TRITON_SETUP.md) | Triton Inference Server setup |
