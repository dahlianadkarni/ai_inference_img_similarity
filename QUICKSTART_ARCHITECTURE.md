# Client-Service Architecture Quick Start

## What Changed?

Your app is now split into two independent services:

1. **Inference Service** (port 8002) — Handles all model inference
   - Stateless
   - Can run on any machine
   - Only knows about: images in, embeddings out

2. **Client/UI** (port 8000) — Your photo review interface
   - Scans photos
   - Calls inference service for embeddings
   - Manages storage and grouping
   - Runs your UI

Even though both run locally now, this is exactly how production systems like Triton, TorchServe, and vLLM work.

## Quick Start (Choose One)

### Option 1: Start Both Services at Once (Easy)

```bash
python start_services.py
```

- ✓ Starts inference service (port 8002)
- ✓ Waits for it to be ready
- ✓ Starts UI client (port 8000)
- ✓ Shows you the URLs

Press `Ctrl+C` to stop both.

**Use Demo Dataset (runs demo UI on port 8001):**
```bash
python start_services.py --ui-demo
# Demo UI: http://127.0.0.1:8001 (demo dataset)
# Inference service: http://127.0.0.1:8002
# Main UI: http://127.0.0.1:8000
```

### Option 2: Start Services Separately (Better for Development)

**Terminal 1:**
```bash
python -m src.inference_service.server
```
```
INFO:     Application startup complete
INFO:     Uvicorn running on http://127.0.0.1:8002
```

**Terminal 2:**
```bash
python -m src.ui.main
```
```
INFO:     Application startup complete
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Now open http://127.0.0.1:8000 in your browser.

### Option 3: Test Just the Architecture

```bash
python test_architecture.py
```

Validates all components without starting the full services.

## Using Remote Embeddings

After starting the inference service, you can use the new embedding generation mode:

### Via CLI (with service)

```bash
python -m src.embedding.main_v2 scan_for_embeddings.json \
   --mode remote \
   --service-url http://127.0.0.1:8002
```

### Via CLI (auto - tries remote, falls back to local)

```bash
python -m src.embedding.main_v2 scan_for_embeddings.json
```

### Via CLI (original local mode)

```bash
python -m src.embedding.main_v2 scan_for_embeddings.json --mode local
```

## File Map

**New Files:**
- `src/inference_service/server.py` — The inference service
- `src/inference_service/client.py` — Client for calling the service
- `src/embedding/main_v2.py` — Refactored with local/remote support
- `start_services.py` — Dual startup script
- `test_architecture.py` — Validation tests
- `ARCHITECTURE_REFACTOR.md` — Full technical details
- `QUICKSTART_ARCHITECTURE.md` — This file

**Unchanged:**
- `src/ui/main.py` — Your UI (unchanged)
- `src/embedding/main.py` — Original CLI (still works)
- All scanning, grouping, storage code

## What to Explore Next

### 1. See How They Talk to Each Other

```bash
# Terminal 1: Start inference service
python -m src.inference_service.server

# Terminal 2: Run test script
python3 -c "
from src.inference_service.client import InferenceClient
client = InferenceClient()
if client.health_check():
    print('✓ Connected!')
    print(f'Model: {client.get_model_info()}')
"
```

### 2. Check Service Endpoints

With service running:
- Health check: http://127.0.0.1:8002/health
- Model info: http://127.0.0.1:8002/model-info
- API docs: http://127.0.0.1:8002/docs (interactive Swagger UI)

### 3. Generate Embeddings with Remote Service

```bash
# Terminal 1: Start inference service
python -m src.inference_service.server

# Terminal 2: Scan photos (if not already done)
python -m src.scanner.main --photos-library --use-applescript --limit 50

# Terminal 3: Generate embeddings via service
python -m src.embedding.main_v2 scan_for_embeddings.json --mode remote
```

### 4. Run Full Workflow

```bash
# Terminal 1: Start both services at once
python start_services.py

# Terminal 2: In your browser
# Open http://127.0.0.1:8000
# Click "Control Panel" → "Start Scan"
# Then "Generate Embeddings"
```

## Why This Matters

This architecture is the foundation for:

- **Triton Inference Server** — NVIDIA's production inference platform
- **TorchServe** — PyTorch's official model serving
- **vLLM** — Ultra-fast LLM serving
- **Ray Serve** — Distributed model serving
- **Kubernetes deployments** — Scale inference independently from client

By building this now, you're learning the mental model that these frameworks assume.

## Troubleshooting

### Service won't start

```bash
# Check if port 8002 is in use
lsof -i :8002

# Or try a different port
python -m src.inference_service.server --port 8002
```

### UI can't find service

```bash
# Check service is running
curl http://127.0.0.1:8002/health

# Or specify service URL when starting UI
python -m src.ui.main --inference-url http://127.0.0.1:8002
```

### Slow embedding generation

- Service startup is slow (loads model) — first time ~10-15s
- Once loaded, subsequent calls are fast
- Check: Is the service reusing the model? (It should be)

## Next Steps

1. **Understand the flow:** Read [ARCHITECTURE_REFACTOR.md](ARCHITECTURE_REFACTOR.md)
2. **Try it out:** Start services and generate embeddings
3. **Experiment:**
   - Add logging to see request/response flow
   - Measure inference time vs. network overhead
   - Try batch processing
4. **Prepare for scaling:** Think about what would change with GPU service on separate machine

Good luck! You're now building production-grade ML systems.
