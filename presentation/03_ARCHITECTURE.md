# Architecture Diagrams

> Visual reference for the system architecture at each step of the project.

---

## 1. Starting Point: Monolithic Application

```
┌──────────────────────────────────────────────────┐
│                 macOS Application                 │
│                                                  │
│  ┌──────────┐  ┌───────────────┐  ┌───────────┐ │
│  │ Scanner  │→ │ Model Loading │→ │ Embedding │ │
│  │          │  │ + Inference   │  │ Storage   │ │
│  └──────────┘  └───────────────┘  └───────────┘ │
│       ↓                                    ↓     │
│  ┌──────────┐                        ┌─────────┐ │
│  │ Hashing  │                        │Grouping │ │
│  │ MD5+dHash│                        │Cosine   │ │
│  └──────────┘                        └─────────┘ │
│                       ↓                          │
│              ┌─────────────────┐                 │
│              │   Web UI        │                 │
│              │   (FastAPI)     │                 │
│              │   Port 8000     │                 │
│              └─────────────────┘                 │
│                                                  │
│  Problem: Model loads inline, can't scale,       │
│  can't deploy to GPU independently               │
└──────────────────────────────────────────────────┘
```

---

## 2. After Step 1: Client–Service Split

```
┌─────────────────────────────┐      ┌─────────────────────────────┐
│        Client / UI          │      │     Inference Service       │
│        (Port 8000)          │      │     (Port 8002)             │
│                             │      │                             │
│  ┌──────────┐               │      │  ┌───────────────────────┐  │
│  │ Scanner  │               │      │  │  OpenCLIP ViT-B/32    │  │
│  │          │               │      │  │  (loaded once on       │  │
│  └──────────┘               │      │  │   startup)             │  │
│       ↓                     │      │  └───────────────────────┘  │
│  ┌──────────┐  ┌─────────┐ │      │                             │
│  │ Hashing  │  │Embedding│ │ HTTP │  Endpoints:                 │
│  │ MD5+dHash│  │ Storage │◀┼──────┼─ POST /embed                │
│  └──────────┘  └─────────┘ │      │  GET  /healthz              │
│       ↓            ↓       │      │                             │
│  ┌──────────┐  ┌─────────┐ │      │  Stateless:                 │
│  │ Grouping │  │ Web UI  │ │      │  • No session state         │
│  │ Cosine   │  │ FastAPI │ │      │  • Each request independent │
│  └──────────┘  └─────────┘ │      │  • Horizontally scalable    │
│                             │      │                             │
│  Three modes:               │      │  Configurable via env vars: │
│  • local  (inline model)    │      │  • MODEL_NAME               │
│  • remote (calls service)   │      │  • DEVICE (cpu/cuda/mps)    │
│  • auto   (remote→local)    │      │  • PORT                     │
└─────────────────────────────┘      └─────────────────────────────┘
```

---

## 3. After Step 2–3: Containerized GPU Deployment

```
 ┌─ macOS (local) ───────────────┐       ┌─ Cloud GPU (Vast.ai) ──────────┐
 │                               │       │                                │
 │  ┌───────────────────────┐    │       │  ┌──────────────────────────┐  │
 │  │   Client / UI App     │    │       │  │   Docker Container       │  │
 │  │   (Port 8000)         │    │ HTTP  │  │                          │  │
 │  │                       │────┼───────┼─▶│   Inference Service      │  │
 │  │   python -m src.ui    │    │       │  │   (Port 8002)            │  │
 │  │                       │    │       │  │                          │  │
 │  └───────────────────────┘    │       │  │   ┌────────────────────┐ │  │
 │                               │       │  │   │ OpenCLIP ViT-B/32  │ │  │
 │  Sends: base64 JPEG images   │       │  │   │ PyTorch + CUDA     │ │  │
 │  Receives: 512-dim embeddings │       │  │   └────────────────────┘ │  │
 │                               │       │  │                          │  │
 └───────────────────────────────┘       │  │   FROM nvidia/cuda:12.1  │  │
                                         │  │   NVIDIA Container Toolkit│  │
                                         │  └──────────────────────────┘  │
                                         │                                │
                                         │  GPU: T4 / A10 / RTX / A100   │
                                         └────────────────────────────────┘
```

---

## 4. After Step 4: Triton Inference Server (Side-by-Side)

```
                           ┌─ Cloud GPU ─────────────────────────────────────┐
                           │                                                 │
 ┌─ macOS ───────────┐    │  ┌─ PyTorch Backend ──────┐                     │
 │                   │    │  │  FastAPI Server         │                     │
 │  Client App       │    │  │  Port 8002              │                     │
 │                   │    │  │                          │                     │
 │  Supports both:   │    │  │  POST /embed             │                     │
 │  • PyTorch backend│────┼──│  (base64 JPEG → embed)  │                     │
 │  • Triton backend │    │  │                          │                     │
 │                   │    │  │  Model: PyTorch native   │                     │
 │  env vars:        │    │  └──────────────────────────┘                     │
 │  INFERENCE_BACKEND│    │                                                   │
 │  INFERENCE_URL    │    │  ┌─ Triton Backend ──────────────────────────────┐│
 │                   │    │  │  NVIDIA Triton Inference Server 24.01        ││
 └───────────────────┘    │  │  Port 8003 (HTTP) / 8004 (gRPC) / 8005 (metrics)│
                           │  │                                              ││
                           │  │  Model Repository:                           ││
                           │  │  ┌─────────────────────────────────────┐     ││
                           │  │  │  openclip_vit_b32/                 │     ││
                           │  │  │  ├── config.pbtxt                  │     ││
                           │  │  │  │   • ONNX Runtime backend        │     ││
                           │  │  │  │   • Dynamic batching enabled    │     ││
                           │  │  │  │   • CUDA EP (GPU execution)     │     ││
                           │  │  │  └── 1/model.onnx  (335 MB)       │     ││
                           │  │  └─────────────────────────────────────┘     ││
                           │  │                                              ││
                           │  │  Features:                                   ││
                           │  │  • Dynamic batching (auto-groups requests)   ││
                           │  │  • Prometheus metrics (/metrics endpoint)    ││
                           │  │  • Model versioning & health checks          ││
                           │  └──────────────────────────────────────────────┘│
                           └─────────────────────────────────────────────────┘
```

---

## 5. After Step 5B: Three Execution Providers

```
┌─ Triton Inference Server ────────────────────────────────────────────────┐
│                                                                          │
│  ┌─ openclip_vit_b32 ──────────────┐  ┌─ openclip_vit_b32_trt ────────┐│
│  │                                  │  │                                ││
│  │  ONNX Runtime                    │  │  ONNX Runtime                  ││
│  │  CUDA Execution Provider         │  │  TensorRT Execution Provider   ││
│  │  (FP32)                          │  │  (FP16)                        ││
│  │                                  │  │                                ││
│  │  ┌─────────┐                     │  │  ┌─────────┐                   ││
│  │  │model.onnx│ ← 335 MB          │  │  │model.onnx│ ← symlink!       ││
│  │  └─────────┘                     │  │  └─────────┘                   ││
│  │                                  │  │                                ││
│  │  GPU Compute: 31.1ms (A4000)     │  │  GPU Compute: 4.7ms (A4000)   ││
│  │               4.4ms  (A100)      │  │               2.0ms (RTX 4080)││
│  │               5.7ms  (RTX 4080)  │  │              29.1ms (A100)    ││
│  │                                  │  │                                ││
│  │  Pros:                           │  │  Pros:                         ││
│  │  • Portable, stable              │  │  • Fastest on consumer GPUs    ││
│  │  • Best on datacenter GPUs       │  │  • 6.6× faster than CUDA EP   ││
│  │                                  │  │                                ││
│  │  Cons:                           │  │  Cons:                         ││
│  │  • Slower than PyTorch native    │  │  • 2-5 min first-load time    ││
│  │                                  │  │  • Recompiles per batch shape  ││
│  └──────────────────────────────────┘  └────────────────────────────────┘│
│                                                                          │
│  Same ONNX file, different Execution Providers                           │
│  Switchable via config.pbtxt (no code changes)                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Step 6A: Unified 3-Backend Container

```
┌─ Docker: Dockerfile.step6a-all ──────────────────────────────────────────┐
│                                                                          │
│  ┌─ Supervisor (manages all processes) ────────────────────────────────┐ │
│  │                                                                      │ │
│  │  ┌─ PyTorch FastAPI ────┐  ┌─ Triton Inference Server ────────────┐ │ │
│  │  │  Port 8002           │  │  Port 8003 (HTTP)                    │ │ │
│  │  │                      │  │  Port 8004 (gRPC)                    │ │ │
│  │  │  POST /embed         │  │  Port 8005 (Prometheus metrics)      │ │ │
│  │  │  POST /embed-batch   │  │                                      │ │ │
│  │  │  GET  /healthz       │  │  Models:                             │ │ │
│  │  │  GET  /gpu-info      │  │  ├─ openclip_vit_b32 (ONNX CUDA EP) │ │ │
│  │  │                      │  │  └─ openclip_vit_b32_trt (TRT EP)   │ │ │
│  │  │  Input: base64 JPEG  │  │                                      │ │ │
│  │  │  (~10 KB per image)  │  │  Input: FP32 tensors                 │ │ │
│  │  │                      │  │  (~602 KB per image)                 │ │ │
│  │  └──────────────────────┘  └──────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Total VRAM: 3.3 GB (all 3 backends)                                     │
│  Base: nvcr.io/nvidia/tritonserver:24.01-py3 + PyTorch                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Step 6B: Multi-GPU Architecture

```
┌─ 4× RTX 4080 Instance (Vast.ai) ──────────────────────────────────────┐
│                                                                        │
│  ┌─ Triton Inference Server ─────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  instance_group [                                                  │ │
│  │    { count: 1, kind: KIND_GPU, gpus: [0] },                       │ │
│  │    { count: 1, kind: KIND_GPU, gpus: [1] },                       │ │
│  │    { count: 1, kind: KIND_GPU, gpus: [2] },                       │ │
│  │    { count: 1, kind: KIND_GPU, gpus: [3] }                        │ │
│  │  ]                                                                 │ │
│  │                                                                    │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                          │ │
│  │  │GPU 0 │  │GPU 1 │  │GPU 2 │  │GPU 3 │                          │ │
│  │  │16 GB │  │16 GB │  │16 GB │  │16 GB │                          │ │
│  │  │2.5 GB│  │2.5 GB│  │2.5 GB│  │2.5 GB│  (model loaded on each) │ │
│  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                          │ │
│  │     │         │         │         │                                │ │
│  │     └─────────┴─────────┴─────────┘                                │ │
│  │              PCIe Gen4 Interconnect                                 │ │
│  │              (~32 GB/s per GPU)                                     │ │
│  │                                                                    │ │
│  │  Dynamic Batching:                                                 │ │
│  │  • Requests queued centrally                                       │ │
│  │  • Dispatched to first available GPU                               │ │
│  │  • Batch efficiency: only 22% (most processed as batch-1)         │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  Result: 43.2 img/s peak at concurrency 32                             │
│          1.8× scaling (45% efficiency vs ideal 4×)                     │
│          Bottleneck: Network transfer, not GPU compute                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Data Flow Comparison: PyTorch vs Triton

```
PyTorch Flow (56.9ms total):
┌────────┐  base64 JPEG  ┌──────────┐  decode   ┌──────────┐  inference  ┌──────────┐
│ Client │──── ~10 KB ──▶│  FastAPI  │─────────▶│ Preprocess│───────────▶│  Model   │
│        │               │ (server)  │           │ (server)  │            │ (GPU)    │
└────────┘               └──────────┘           └──────────┘            └──────────┘
  ~25ms network            ~2ms                   ~5ms                    ~10-15ms


Triton Flow (182.9ms total):
┌────────┐  preprocess   ┌──────────┐  binary    ┌──────────┐  inference  ┌──────────┐
│ Client │─── locally ──▶│ Encode   │── 602KB ──▶│  Triton  │───────────▶│  Model   │
│        │               │ (client) │  tensor    │ (server)  │            │ (GPU)    │
└────────┘               └──────────┘            └──────────┘            └──────────┘
  ~0.5ms client-side       ~1.7ms binary          ~150ms network          ~4.4ms

Key difference: WHERE preprocessing happens determines WHAT crosses the network.
  PyTorch: raw image crosses network (~10 KB), server preprocesses
  Triton:  preprocessed tensor crosses network (~602 KB), client preprocesses
```

---

## 9. Step 8: Local Kubernetes (kind)

```
 ┌─ macOS ────────────────────────────────────────────────────────────┐
 │                                                                    │
 │  ┌─────────────────────────────────────────────────────────────┐  │
 │  │  kind cluster (single Docker container — "inference-cluster"│  │
 │  │                                                             │  │
 │  │  ┌─ K8s Service ───────────────────────────────────────┐   │  │
 │  │  │  NodePort 30092 ← host port 8092                   │   │  │
 │  │  │                                                     │   │  │
 │  │  │  ┌─ Deployment (HPA: 2–6 replicas) ──────────────┐ │   │  │
 │  │  │  │                                                │ │   │  │
 │  │  │  │  Pod 1: photo-duplicate-inference:k8s-cpu     │ │   │  │
 │  │  │  │  Pod 2: photo-duplicate-inference:k8s-cpu     │ │   │  │
 │  │  │  │  Pod 3–6: (HPA scales up under load)          │ │   │  │
 │  │  │  │                                                │ │   │  │
 │  │  │  │  Resources:  cpu: 500m req / 2000m limit      │ │   │  │
 │  │  │  │              mem: 1Gi  req / 3Gi  limit       │ │   │  │
 │  │  │  │                                                │ │   │  │
 │  │  │  │  Probes: readiness (GET /health, delay:20s)   │ │   │  │
 │  │  │  │          liveness  (GET /health, delay:60s)   │ │   │  │
 │  │  │  └────────────────────────────────────────────────┘ │   │  │
 │  │  │                                                     │   │  │
 │  │  │  ┌─ HPA ──────────────────────────────────────────┐ │   │  │
 │  │  │  │  target: 60% CPU utilization                   │ │   │  │
 │  │  │  │  scaleUp stabilization:   30s                  │ │   │  │
 │  │  │  │  scaleDown stabilization: 180s                 │ │   │  │
 │  │  │  └────────────────────────────────────────────────┘ │   │  │
 │  │  │                                                     │   │  │
 │  │  │  ┌─ PDB ──────────┐  ┌─ ResourceQuota ───────────┐ │   │  │
 │  │  │  │ minAvailable:1 │  │ pods:10, cpu req:4        │ │   │  │
 │  │  │  └────────────────┘  └────────────────────────────┘ │   │  │
 │  │  └──────────────────────────────────────────────────────┘   │  │
 │  └─────────────────────────────────────────────────────────────┘  │
 │                                                                    │
 │  Port isolation vs docker-compose:                                 │
 │  • docker-compose PyTorch: 8002                                    │
 │  • docker-compose Triton:  8003/8004                               │
 │  • kind K8s:               8092  ← no conflicts                   │
 └────────────────────────────────────────────────────────────────────┘
```

**HPA Scale Events (from load test):**
```
SuccessfulRescale: 2→4 replicas (cpu: 210% of request → 397%)
SuccessfulRescale: 4→6 replicas (still above 60% target, 30s later)
```

---

## 10. Evolution Summary

```
Step 1      Step 2        Step 3         Step 4          Step 5          Step 6
──────      ──────        ──────         ──────          ──────          ──────

Monolith → Split →     Container →   Add Triton →   Optimize →     Compare &
                        on GPU                                       Scale

 ┌───┐    ┌───┐ ┌───┐  ┌───┐ ┌───┐   ┌───┐ ┌───┐   ┌───┐ ┌───┐   ┌───┐ ┌───┐
 │ALL│    │CLI│ │SRV│  │CLI│ │GPU│   │CLI│ │TRI│   │CLI│ │TRI│   │CLI│ │4×G│
 │   │    │   │ │   │  │   │ │SRV│   │   │ │TON│   │   │ │+TRT│  │   │ │   │
 └───┘    └───┘ └───┘  └───┘ └───┘   └───┘ └───┘   └───┘ └───┘   └───┘ └───┘
                        Docker        ONNX export    Binary proto    A100 vs
                        NVIDIA RT     Dyn. batching  TensorRT EP     RTX 4080
                                      gRPC           Profiling       Multi-GPU
                                      Prometheus     FP16            Scaling

Step 7              Step 8
──────              ──────

gRPC vs HTTP →   Kubernetes

 ┌───┐ ┌───┐    ┌───┐ ┌───┐
 │CLI│ │5×B│    │CLI│ │K8s│
 │   │ │END│    │   │ │HPA│
 └───┘ └───┘    └───┘ └───┘
 A100 +         kind cluster
 RTX 4090       2–6 pods
 5 protocols    HPA / PDB
 gRPC vs HTTP   ResourceQuota
```

---

*See [04_BENCHMARK_RESULTS.md](04_BENCHMARK_RESULTS.md) for the complete benchmark data behind these architectures.*
