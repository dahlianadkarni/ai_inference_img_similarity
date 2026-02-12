# Docker Guide - Complete Reference

> **Step 2 Status: ✅ COMPLETE**  
> All containerization requirements from [PLAN.md](PLAN.md) have been successfully implemented.

## Quick Start

```bash
# Option 1: One-command start (Recommended)
./docker-run.sh

# Option 2: Using docker-compose
./docker-compose.sh up

# Option 3: Manual
docker build -t photo-duplicate-inference:latest .
docker run -d -p 8002:8002 --name inference-service photo-duplicate-inference:latest
```

## What Was Built

### Files Created
- **`Dockerfile`** — CPU-optimized production container
- **`Dockerfile.gpu`** — NVIDIA CUDA GPU container
- **`.dockerignore`** — Build optimization
- **`docker-compose.yml`** — Multi-service orchestration
- **`docker-run.sh`** — Quick build & run script
- **`docker-compose.sh`** — Compose wrapper
- **`validate_docker.py`** — Configuration validator

### Requirements Met ✅
- [x] Create Dockerfile for inference service
- [x] Expose `/embed` endpoints (base64 and batch)
- [x] Expose `/health` and `/healthz` endpoints
- [x] Make model configurable via env vars
- [x] Run locally via Docker

## Testing

```bash
# Health check
curl http://localhost:8002/healthz
# or
curl http://localhost:8002/healthz

# Model info
curl http://localhost:8002/model-info

# API docs
open http://localhost:8002/docs
```

## Common Commands

```bash
# View logs
docker logs -f inference-service

# Stop
docker stop inference-service

# Remove
docker rm inference-service

# Resource usage
docker stats inference-service
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `ViT-B-32` | OpenCLIP model architecture |
| `MODEL_PRETRAINED` | `openai` | Pretrained weights |
| `HOST` | `127.0.0.1` | Server bind address (use `0.0.0.0` in containers) |
| `PORT` | `8002` | Server port |
| `LOG_LEVEL` | `info` | Logging level |

### Custom Configuration

```bash
docker run -d \
  -p 8002:8002 \
  -e MODEL_NAME=ViT-L-14 \
  -e MODEL_PRETRAINED=openai \
  -e LOG_LEVEL=debug \
  --name inference-service \
  photo-duplicate-inference:latest
```

## GPU Deployment

### Local GPU (NVIDIA on Linux/Windows)

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t photo-duplicate-inference:gpu .

# Run with GPU
docker run -d \
  -p 8002:8002 \
  --gpus all \
  -e HOST=0.0.0.0 \
  --name inference-service \
  photo-duplicate-inference:gpu
```

### Cloud GPU Deployment (Vast.ai, RunPod, Lambda Labs)

**Important:** Must build for `linux/amd64` architecture on macOS:

```bash
# Build for cloud GPU instances (linux/amd64)
docker buildx build --platform linux/amd64 \
  -f Dockerfile.gpu \
  -t yourdockeruser/photo-duplicate-inference:gpu-linux-amd64 \
  .

# Push to Docker Hub
docker push yourdockeruser/photo-duplicate-inference:gpu-linux-amd64
```

**See [GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md) for complete cloud deployment guide** including:
- Vast.ai setup walkthrough
- Environment variable configuration
- Port mapping and troubleshooting
- Cost management tips

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs inference-service

# Check if port is in use
lsof -i :8002
```

### Health check failing
```bash
# Wait longer (model loading takes time)
sleep 30 && curl http://localhost:8002/health

# Check container is running
docker ps
```

### Out of memory
```bash
# Increase memory limit
docker run -d \
  -p 8002:8002 \
  --memory="8g" \
  --name inference-service \
  photo-duplicate-inference:latest
```

## Files Overview

- `Dockerfile` — CPU-optimized container
- `Dockerfile.gpu` — GPU-enabled container (CUDA)
- `.dockerignore` — Build optimization
- `docker-compose.yml` — Multi-service orchestration
- `docker-run.sh` — Quick build & run script
- `docker-compose.sh` — Compose wrapper script
- `validate_docker.py` — Configuration validator

## Architecture

```
Container (8002)
├── Python 3.11
├── FastAPI
├── PyTorch (CPU or GPU)
├── OpenCLIP
└── Health checks
    
Endpoints:
- GET  /health, /healthz
- GET  /model-info
- POST /embed/base64
- POST /embed/batch
- GET  /docs
```

---

## Security & Best Practices

### Secure Defaults ✅

**Default HOST: 127.0.0.1**
- Container defaults to localhost binding (security-first)
- Prevents accidental public exposure
- Override to `0.0.0.0` only when needed (docker-compose does this)

```bash
# Secure by default
docker run -p 8002:8002 photo-duplicate-inference:latest
# Binds to 127.0.0.1 inside container

# Explicit override for container networking
docker run -e HOST=0.0.0.0 -p 8002:8002 photo-duplicate-inference:latest
```

### Security Features

✅ **Non-Root User** — Both Dockerfiles use `appuser` (UID 1000)  
✅ **Minimal Base Images** — `python:3.11-slim` (CPU), `nvidia/cuda:12.1.0-base` (GPU)  
✅ **PyTorch Protection** — GPU Dockerfile prevents accidental version downgrades  
✅ **Health Endpoints** — Both `/health` and `/healthz` supported  
✅ **Resource Limits** — Memory and CPU caps in docker-compose

### Production Recommendations

**⚠️ No Authentication by Default** — Suitable for private networks only.

For public deployment, add:

```python
# Option 1: API Key middleware
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403)
```

```bash
# Option 2: Use reverse proxy (nginx/Traefik) with:
# - Rate limiting, IP whitelisting, OAuth2/JWT
```

**Network Isolation:**
```yaml
# docker-compose.yml
networks:
  inference-net:
    driver: bridge

services:
  inference-service:
    networks:
      - inference-net
```

**Container Hardening:**
```bash
# Read-only filesystem
docker run --read-only --tmpfs /tmp:rw,noexec,nosuid ...

# Drop capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE ...

# Security scanning
docker scan photo-duplicate-inference:latest
```

---

## Recent Security Fixes

### 1. Changed Default HOST (0.0.0.0 → 127.0.0.1)
**Why**: Prevents accidental exposure without authentication.  
**Impact**: Must explicitly set `HOST=0.0.0.0` for container networking.

### 2. Protected GPU PyTorch Installation
**Why**: Prevent `requirements-ml.txt` from downgrading CUDA torch.  
**How**: Filter out torch entries after CUDA installation.

### 3. Fixed docker-compose Resource Limits
**Why**: `deploy.resources` only works in Swarm mode.  
**Fix**: Use `mem_limit` and `cpus` instead.

### 4. Added /healthz Endpoint
**Why**: Kubernetes convention for health checks.  
**Impact**: Better orchestration compatibility.

---

## Validation

```bash
python validate_docker.py
```

Expected: ✅ All 11 checks passed

---

## Next Steps

**✅ Step 2 Complete** — Containerization done!

**→ Step 3: GPU Deployment**
```bash
# Run on cloud GPU VM (T4/A10/A100)
docker build -f Dockerfile.gpu -t photo-duplicate-inference:gpu .
docker run --gpus all -p 8002:8002 photo-duplicate-inference:gpu
```

**→ Step 4: Triton Inference Server**  
**→ Step 5: TensorRT Optimization**

---

## See Also

- [PLAN.md](PLAN.md) — 5-step learning plan
- [README.md](README.md) — Project overview
