# Client-Service Architecture Refactor

## Overview

The app has been refactored to separate the **client** (photo management UI) from the **inference service** (embedding generation). This is a critical architectural step for learning inference deployment patterns.

Even though both run on your Mac initially, they communicate via a clean HTTP boundary—exactly like production systems with Triton, TorchServe, or vLLM.

```
┌─────────────────────────────────────────────────┐
│ Client (UI)                                      │
│ - Photo scanning (AppleScript / directory)       │
│ - Manages local embeddings storage              │
│ - Calls inference service via HTTP              │
│ Port: 8000                                       │
└──────────────────┬──────────────────────────────┘
                   │
                   │ HTTP requests
                   │ (image bytes or base64)
                   ↓
┌─────────────────────────────────────────────────┐
│ Inference Service (Stateless)                   │
│ - Loads vision model once at startup             │
│ - Accepts images via HTTP (base64 or files)     │
│ - Returns embeddings as JSON                     │
│ - No knowledge of photos, groups, or metadata   │
│ Port: 8002                                       │
└─────────────────────────────────────────────────┘
```

## Key Architectural Principles

### 1. **Separation of Concerns**

- **Client**: Owns photo discovery, metadata, grouping logic, UI state
- **Service**: Owns model loading, inference, hardware optimization

### 2. **Statelessness**

The inference service doesn't know about:
- Which photos you're analyzing
- How embeddings will be grouped
- Your photos library structure
- UI state or user preferences

Each request is independent and can be served by any instance (key for scaling with Kubernetes).

### 3. **Clean Boundary**

Communication is purely HTTP with JSON/multipart encoding:
- Client loads images from disk
- Encodes them (base64 or multipart)
- Sends to service
- Receives embeddings back
- Stores locally for grouping/analysis

This allows:
- Independent deployment (e.g., GPU service on separate machine)
- Different tech stacks (Python model server, Node.js UI, etc.)
- Easy containerization (each service is a separate container)
- Horizontal scaling (multiple inference instances behind load balancer)

## New Components

### Inference Service (`src/inference_service/server.py`)

FastAPI server that:
- Loads OpenCLIP model at startup
- Exposes HTTP endpoints for embedding generation
- Handles image encoding/decoding
- Returns embeddings as JSON lists

**Endpoints:**
- `GET /health` — Health check
- `GET /model-info` — Current model details
- `POST /embed/base64` — Embed images from base64 strings
- `POST /embed/batch` — Embed images from file uploads

**Key feature:** Model is loaded once per process. Reuses model if same architecture requested.

### Inference Client (`src/inference_service/client.py`)

Python client that:
- Connects to inference service
- Handles image encoding (base64, file upload)
- Manages batching for large image sets
- Provides synchronous API to calling code

Usage:
```python
from src.inference_service.client import InferenceClient

client = InferenceClient("http://127.0.0.1:8002")
embeddings = client.embed_images_files(image_paths)  # Returns numpy array
```

### Refactored Embedding Generation (`src/embedding/main_v2.py`)

CLI tool with three modes:

**Local Mode** (`--mode local`)
- Original behavior: embedding generation inline
- Good for: Testing, small datasets, single-machine setup
```bash
python -m src.embedding.main_v2 scan_for_embeddings.json --mode local
```

**Remote Mode** (`--mode remote`)
- Calls inference service
- Good for: Distributed setup, reusing service across clients
```bash
python -m src.embedding.main_v2 scan_for_embeddings.json --mode remote --service-url http://127.0.0.1:8002
```

**Auto Mode** (`--mode auto`) — Default
- Tries remote first
- Falls back to local if service unavailable
- Good for: Development (flexible setup)
```bash
python -m src.embedding.main_v2 scan_for_embeddings.json --mode auto
```

## Getting Started

### Option 1: Start Both Services Together (Recommended for Development)

```bash
source venv/bin/activate
python start_services.py
```

This:
1. Starts inference service on `http://127.0.0.1:8002`
2. Waits for it to become ready
3. Starts UI client on `http://127.0.0.1:8000`
4. Opens browser to client

Both services run in foreground. Press Ctrl+C to stop both.

### Option 2: Start Services Separately (Recommended for Production/Debugging)

**Terminal 1 — Inference Service:**
```bash
source venv/bin/activate
python -m src.inference_service.server
```
```
INFO:     Uvicorn running on http://127.0.0.1:8002
```

**Terminal 2 — UI Client:**
```bash
source venv/bin/activate
python -m src.ui.main
```
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Testing the Connection

```bash
python -c "
from src.inference_service.client import InferenceClient
client = InferenceClient()
if client.health_check():
    print('✓ Service is healthy')
    print(f'Model: {client.get_model_info()}')
else:
    print('✗ Service not available')
"
```

## Embedding Generation Workflow

### Via UI (New)

1. Click "Control Panel → Generate Embeddings"
2. Set similarity threshold
3. UI calls backend API to start embedding process
4. Backend uses `InferenceClient` to call inference service
5. Results stored in `embeddings/`

### Via CLI (New Modes)

**Remote (new):**
```bash
python -m src.embedding.main_v2 scan_for_embeddings.json \
   --mode remote \
   --service-url http://127.0.0.1:8002
```

**Local (original):**
```bash
python -m src.embedding.main_v2 scan_for_embeddings.json \
  --mode local
```

**Auto (smart):**
```bash
python -m src.embedding.main_v2 scan_for_embeddings.json
# Tries remote, falls back to local
```

## What This Enables

### Short-term (Learning)

- Understand model serving patterns
- Learn FastAPI for ML systems
- Experience client-server communication
- See how inference frameworks are designed

### Medium-term (Scaling)

- Run inference service on GPU machine
- Client on Mac accesses remote service via HTTP
- Multiple clients can share one inference service
- Easy to benchmark and profile service independently

### Long-term (Production)

- Containerize service (Docker)
- Deploy on Kubernetes with auto-scaling
- Replace FastAPI with Triton/TorchServe/vLLM
- Add inference batching, model quantization, etc.

## Inference Frameworks to Try Next

Once you're comfortable with this architecture, you can replace `src/inference_service/server.py` with:

### **Triton Inference Server**
Google/NVIDIA's production inference system. Handles batching, model versioning, multi-GPU.

```yaml
# Simple Triton config that replaces FastAPI
model_repository:
  - name: clip
    platform: "pytorch_libtorch"
    model_path: "./models/clip"
```

### **TorchServe**
PyTorch's official model serving solution. Built-in metrics, A/B testing, model packaging.

```bash
torch-model-archiver --model-name clip --version 1.0 ...
torchserve --start --models clip.mar
```

### **vLLM** (Not needed yet, but for when you add LLMs)
State-of-the-art LLM serving with paged attention. Higher throughput than standard deployments.

### **BentoML**
Python-first ML model serving. Bridges research and production. Good for learning.

## File Structure

```
src/
├── inference_service/           # ← NEW: Stateless inference API
│   ├── __init__.py
│   ├── server.py                # FastAPI app with embedding endpoints
│   └── client.py                # Client for calling the service
│
├── embedding/
│   ├── embedder.py              # Original ImageEmbedder (unchanged)
│   ├── main.py                  # Original CLI (kept for reference)
│   ├── main_v2.py               # ← NEW: Refactored to support local/remote modes
│   └── storage.py               # (unchanged)
│
├── scanner/                     # (unchanged)
├── ui/                          # (unchanged - calls inference service)
└── grouping/                    # (unchanged)

start_services.py                # ← NEW: Dual startup script
```

## Migration Path

You can keep using the old monolithic code while learning the new pattern:

1. **Current:** Original `main.py` still works for local embedding generation
2. **New:** Use `main_v2.py` with `--mode auto` to try service-oriented approach
3. **Transition:** Once comfortable, update UI to always use remote inference
4. **Production:** Deploy service independently, update client for service URL

## Next Steps

1. **Try local-mode embedding generation:**
   ```bash
   python -m src.embedding.main_v2 scan_for_embeddings.json --mode local
   ```

2. **Start both services:**
   ```bash
   python start_services.py
   ```

3. **Try remote-mode embedding generation:**
   ```bash
   python -m src.embedding.main_v2 scan_for_embeddings.json --mode remote
   ```

4. **Experiment in different terminal:**
   - Start inference service in one terminal
   - Start/stop UI in another
   - See how they interact

5. **Prepare for scaling:**
   - Add logging/metrics to inference service
   - Experiment with batch size parameters
   - Test with larger photo libraries
   - Time the inference vs. client communication overhead

## Questions to Explore

- What's the network overhead vs. base64/file encoding?
- How does model loading time compare to inference time?
- Can you batch requests across multiple clients?
- How would you add load balancing with multiple inference instances?
- What monitoring/metrics matter for an inference service?

Good luck! This foundation will make all subsequent ML deployment patterns (Triton, TorchServe, vLLM) feel familiar.
