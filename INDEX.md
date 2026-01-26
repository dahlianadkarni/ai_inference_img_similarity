# Photo Duplicate Detection: Client-Service Architecture

## 📚 Documentation Index

Read these in order:

1. **[00_START_HERE.md](00_START_HERE.md)** ← Start here!
   - 5-minute quick overview
   - Common commands
   - Success metrics

2. **[QUICKSTART_ARCHITECTURE.md](QUICKSTART_ARCHITECTURE.md)** 
   - Fast setup (choose your startup style)
   - Using remote embeddings
   - Quick troubleshooting

3. **[REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md)**
   - What changed and why
   - Getting started options
   - File reference

4. **[ARCHITECTURE_REFACTOR.md](ARCHITECTURE_REFACTOR.md)**
   - Full technical deep-dive
   - Key architectural principles
   - Migration path

5. **[ARCHITECTURE_DIAGRAM.py](ARCHITECTURE_DIAGRAM.py)**
   - Visual architecture diagram
   - Data flow examples
   - Why this pattern matters

6. **[VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md)**
   - What's been implemented
   - Test results
   - Production readiness

## 🚀 Start Here

```bash
# Everything in one command
python start_services.py

# Or start separately
python -m src.inference_service.server    # Terminal 1
python -m src.ui.main                     # Terminal 2
```

Open http://127.0.0.1:8000

## 🎯 What Is This?

Your photo duplicate detection app has been refactored from a **monolithic design** to a **client-service architecture**.

**Before:**
```
[UI + Scanner + Model + Grouping] → All in one process
```

**After:**
```
[Client/UI] ←HTTP→ [Inference Service]
```

This is the exact pattern used by:
- **Triton Inference Server** (NVIDIA)
- **TorchServe** (PyTorch)
- **vLLM** (LLM serving)
- **Ray Serve** (distributed ML)
- **Kubernetes ML deployments**

## ✅ What Works

✅ Local inference mode (original behavior)
✅ Remote inference mode (via service)  
✅ Auto mode (tries remote, falls back to local)
✅ Both startup options (single command or separate)
✅ Full test suite (all passing)
✅ Clean HTTP API with interactive docs

## 📊 Architecture

```
┌─────────────────────┐      ┌──────────────────────┐
│ Client (port 8000)  │─────→│ Service (port 8001)  │
├─────────────────────┤      ├──────────────────────┤
│ • Scan photos      │  HTTP │ • Load model        │
│ • Call service     │       │ • Generate embeddings│
│ • Store embeddings │ JSON  │ • No state          │
│ • Group results    │       │ • Stateless API     │
│ • Display UI       │       │                      │
└─────────────────────┘      └──────────────────────┘
```

## 💾 New Files

**Core Service:**
- `src/inference_service/server.py` — FastAPI inference API (250 lines)
- `src/inference_service/client.py` — HTTP client (200 lines)

**Refactored Embedding:**
- `src/embedding/main_v2.py` — Local/remote/auto modes (350 lines)

**Startup & Testing:**
- `start_services.py` — Dual startup script (100 lines)
- `test_architecture.py` — Validation suite (350 lines)

**Documentation:**
- All files in this index
- This file (INDEX.md)

## 🔍 Quick Overview

### Three Inference Modes

**Local** (Original behavior, no service needed)
```bash
python -m src.embedding.main_v2 scan.json --mode local
```

**Remote** (Via service)
```bash
python -m src.embedding.main_v2 scan.json --mode remote
```

**Auto** (Tries remote, falls back to local)
```bash
python -m src.embedding.main_v2 scan.json
```

### Key Concepts

**Separation of Concerns**
- Client: Photo management, UI, grouping
- Service: Model loading, inference only

**Statelessness**
- Service knows nothing about photos
- Each request is independent
- Can scale horizontally

**Clean Boundary**
- Communication via HTTP/JSON
- Can deploy to different machines
- Easy to containerize

## 📈 Learning Path

**Week 1:**
- [ ] Read 00_START_HERE.md
- [ ] Run `python test_architecture.py`
- [ ] Start services with `python start_services.py`
- [ ] Generate embeddings in both modes

**Week 2:**
- [ ] Read ARCHITECTURE_REFACTOR.md
- [ ] Add logging to trace requests
- [ ] Measure performance
- [ ] Explore API docs at `/docs`

**Week 3+:**
- [ ] Containerize service (Docker)
- [ ] Deploy on separate machine
- [ ] Learn Triton/TorchServe
- [ ] Plan for GPU inference

## 💻 Common Commands

```bash
# Start everything
python start_services.py

# Start service
python -m src.inference_service.server

# Start UI
python -m src.ui.main

# Test
python test_architecture.py

# Embeddings (local)
python -m src.embedding.main_v2 scan.json --mode local

# Embeddings (remote)
python -m src.embedding.main_v2 scan.json --mode remote

# Embeddings (auto)
python -m src.embedding.main_v2 scan.json

# Health check
curl http://127.0.0.1:8001/health

# API docs (while running)
open http://127.0.0.1:8001/docs
```

## ❓ Why This Matters

This architecture is **not just code organization**. It's the foundational pattern for:

1. **Production ML Systems** — Every serious ML deployment uses this
2. **Horizontal Scaling** — Replicate service instances behind load balancer
3. **Independent Deployment** — Update client and service separately
4. **Framework Agility** — Replace FastAPI with Triton/TorchServe easily
5. **Cloud Ready** — Service on GPU machine, client anywhere else

By learning this now, you understand the core pattern behind:
- Cloud ML platforms (GCP Vertex AI, AWS SageMaker)
- Kubernetes ML deployments
- Enterprise inference systems
- Open source frameworks (Triton, TorchServe, vLLM)

## ✨ Summary

You have:
- ✅ Production-grade architecture
- ✅ Scalable inference pattern
- ✅ Clean HTTP API
- ✅ Flexible deployment
- ✅ Foundation for advanced frameworks

**Status:** ✅ Complete and tested  
**All tests:** Passing (4/4)  
**Ready to use:** Yes

---

## Next Action

**👉 Open [00_START_HERE.md](00_START_HERE.md)**

It's the quickest way to get started!
