# Architecture Refactor: Client-Service Separation

## Summary

Your photo duplicate detection app has been refactored into a **client-service architecture**—the same pattern used by production inference systems like Triton, TorchServe, and vLLM.

**Before:** Monolithic app (scanning + embedding generation + UI in one process)
**After:** Two independent services (client UI + stateless inference service)

This isn't just a refactor for refactor's sake. It's the foundational architecture for learning how real ML systems scale.

## What Changed

### New Components

1. **Inference Service** (`src/inference_service/`)
   - `server.py` — FastAPI application that serves embeddings over HTTP
   - `client.py` — Python client for calling the inference service
   - Clean, stateless HTTP API
   - Handles all model loading and inference

2. **Refactored Embedding Generation** (`src/embedding/main_v2.py`)
   - Three modes: `local`, `remote`, or `auto`
   - Can call inference service or generate embeddings inline
   - Backward compatible with existing code

3. **Startup Scripts**
   - `start_services.py` — Start both services with one command
   - `test_architecture.py` — Validate all components

4. **Documentation**
   - `ARCHITECTURE_REFACTOR.md` — Full technical deep-dive
   - `QUICKSTART_ARCHITECTURE.md` — Quick start guide
   - `ARCHITECTURE_DIAGRAM.py` — Visual diagrams and data flow

## Quick Start

### Option 1: Start Everything at Once

```bash
python start_services.py
```

This starts both services and opens your browser to http://127.0.0.1:8000.

### Option 2: Start Services Separately

```bash
# Terminal 1
python -m src.inference_service.server

# Terminal 2
python -m src.ui.main
```

### Option 3: Just Test the Architecture

```bash
python test_architecture.py
```

## Key Architectural Principles

### 1. Separation of Concerns

- **Client**: Owns photo discovery, metadata, grouping, UI
- **Service**: Owns model loading, inference, hardware optimization

### 2. Statelessness

The inference service doesn't know about:
- Which photos you're analyzing
- How embeddings will be grouped
- Your photos library structure
- UI state or preferences

Each request is independent → can scale horizontally.

### 3. Clean HTTP Boundary

```
Client (http://127.0.0.1:8000)
   ↓
   │ HTTP/POST
   │ images (base64 or multipart)
   ↓
Service (http://127.0.0.1:8001)
   ↓
   │ HTTP/200
   │ embeddings (JSON array)
   ↓
Client (stores locally, groups, displays)
```

## Inference Modes

### Local Mode (Original Behavior)

```bash
python -m src.embedding.main_v2 scan_for_embeddings.json --mode local
```

- No service needed
- Embedding generation inline
- Good for: Testing, single-machine setups

### Remote Mode (New)

```bash
python -m src.embedding.main_v2 scan_for_embeddings.json --mode remote --service-url http://127.0.0.1:8001
```

- Calls inference service over HTTP
- Service can run anywhere
- Good for: Distributed setups, GPU servers

### Auto Mode (Recommended)

```bash
python -m src.embedding.main_v2 scan_for_embeddings.json
```

- Tries remote mode first
- Falls back to local if service unavailable
- Good for: Development (flexible)

## What This Enables

### Short-term (Learning)

- ✓ Understand model serving patterns
- ✓ Learn FastAPI for ML systems
- ✓ See how inference frameworks are designed
- ✓ Experience client-server communication

### Medium-term (Scaling)

- ✓ Run inference service on GPU machine
- ✓ Client on Mac accesses remote service
- ✓ Multiple clients share one service
- ✓ Easy to benchmark and profile independently

### Long-term (Production)

- ✓ Containerize service (Docker)
- ✓ Deploy on Kubernetes with auto-scaling
- ✓ Replace FastAPI with Triton/TorchServe/vLLM
- ✓ Add inference batching, quantization, etc.

## Inference Frameworks to Try Next

Once comfortable with this architecture, you can replace the service with:

### **Triton Inference Server**
Google/NVIDIA's production inference system. Handles batching, versioning, multi-GPU.

### **TorchServe**
PyTorch's official model serving. Built-in metrics, A/B testing, packaging.

### **vLLM**
State-of-the-art LLM serving. (Future, when you add LLMs)

### **BentoML**
Python-first ML serving. Good bridge from research to production.

The beautiful part: **Your client code barely changes**. You just update the service URL or replace the FastAPI server with Triton. That's the power of clean architecture.

## File Changes

### New Files
```
src/inference_service/
├── __init__.py
├── server.py          # Inference API server
└── client.py          # Client for calling service

src/embedding/
└── main_v2.py         # Refactored with local/remote support

start_services.py      # Dual startup script
test_architecture.py   # Validation tests
ARCHITECTURE_REFACTOR.md
QUICKSTART_ARCHITECTURE.md
ARCHITECTURE_DIAGRAM.py
```

### Updated Files
```
requirements.txt       # Added httpx for HTTP client
```

### Unchanged
```
src/ui/                # Works with both local and remote inference
src/embedding/main.py  # Original kept for reference
src/scanner/           # Unchanged
src/grouping/          # Unchanged
```

## How to Use

### 1. Test Architecture
```bash
python test_architecture.py
```

### 2. See It In Action
```bash
# Terminal 1: Start inference service
python -m src.inference_service.server

# Terminal 2 (in another terminal): Generate embeddings
python -m src.embedding.main_v2 scan_for_embeddings.json --mode remote

# Or use UI: python -m src.ui.main
```

### 3. Explore the Code
- `ARCHITECTURE_REFACTOR.md` — Full technical details
- `QUICKSTART_ARCHITECTURE.md` — Quick commands
- `ARCHITECTURE_DIAGRAM.py` — Visual explanations
- `src/inference_service/server.py` — Inference API
- `src/inference_service/client.py` — Client implementation

### 4. Next Steps
- Add logging to trace requests
- Measure inference time vs. network overhead
- Try different batch sizes
- Plan: What if service moved to GPU machine?
- Experiment: How would you scale this?

## Why This Matters

This is **not** just code organization. It's the architectural pattern that powers:

- **Triton** at scale (model serving)
- **TorchServe** deployments (PyTorch serving)
- **vLLM** (LLM inference optimization)
- **Ray Serve** (distributed ML serving)
- **Kubernetes ML deployments** (container orchestration)
- **Cloud ML platforms** (managed inference)

By building this now, you're learning the foundational mental model. When you encounter these frameworks later, they'll feel familiar because you've built their core pattern.

## Questions to Explore

1. What's the network overhead vs. base64/file encoding?
2. How does model loading time compare to inference time?
3. Can you batch requests from multiple clients?
4. What happens if service crashes while client is using it?
5. How would you add authentication to the service?
6. What metrics matter for monitoring an inference service?
7. How would you deploy this service on GPU?
8. What changes when you have multiple inference instances?

## Troubleshooting

### Service won't start
```bash
# Check if port 8001 is in use
lsof -i :8001

# Or use different port
python -m src.inference_service.server --port 8002
```

### Health check fails
```bash
curl http://127.0.0.1:8001/health
```

### Slow first run
- Service startup takes ~10-15s (loading model)
- Subsequent calls are fast
- This is expected and normal

### Can't connect from different machine
- Make sure to use actual IP instead of 127.0.0.1
- Check firewall rules
- Verify service started with `--host 0.0.0.0`

## Learning Path

1. **Week 1:** Get comfortable with both startup options
2. **Week 2:** Add logging, measure performance
3. **Week 3:** Experiment with remote service on different port
4. **Week 4:** Research how Triton/TorchServe work
5. **Week 5:** Build a simple Triton model and try swapping it in

You're building production-grade ML systems. Good luck!
