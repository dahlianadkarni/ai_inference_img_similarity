# Commands Cheat Sheet

```bash
source venv/bin/activate
```

---
If scan results get deleted:
I have local photos stored in photos_local cache.
Restore scan_for_embeddings.json from real photos (fast, no Photos app needed):
```bash
python -m src.scanner.main .cache/photos_local \
  --output scan_for_embeddings.json \
  --duplicates-output scan_duplicates.json \
  --cache-file .cache/scan_cache.json
  ```

And for demo:
```bash
python -m src.scanner.main /Users/dnadkarn/demo_photos \
  --output scan_results_demo.json \
  --duplicates-output scan_duplicates_demo.json \
  --cache-file .cache_demo/scan_cache.json
  ```

---
## Local App Server, Without inference servers

## Local App Server (UI only, no inference)
```bash
python -m src.ui.main
# → http://127.0.0.1:8080
```

## Demo UI Only (no inference)
```bash
python start_services.py --ui-demo --no-inference
# → Demo UI:   http://127.0.0.1:8081
```

---
## With inference servers

## Local App Server + Local Inference (PyTorch)
```bash
python start_services.py
# → UI:        http://127.0.0.1:8080
# → Inference: http://127.0.0.1:8002
```

> **Note:** "Local inference" by default means the PyTorch FastAPI server (src/inference_service). Triton and TensorRT require Docker.

## Demo Mode (UI + Local Inference)
```bash
python start_services.py --ui-demo
# → Demo UI:   http://127.0.0.1:8081
# → Inference: http://127.0.0.1:8002
```

## Local Inference Only (PyTorch, separate terminal)
```bash
# Native (venv)
python -m src.inference_service
# → http://127.0.0.1:8002

# Or via Docker
./docker-run.sh
# → http://127.0.0.1:8002
```

---

## Local Inference with Triton/ONNX (Docker, CPU)
> Currently running as `triton-inference-service`. Restart with:
```bash
./build_triton_local.sh          # (re)build image + start container
# or, if image already built:
docker start triton-inference-service

export INFERENCE_BACKEND=triton
export INFERENCE_SERVICE_URL=http://127.0.0.1:8003
python -m src.ui.main
# → Triton HTTP:    http://127.0.0.1:8003
# → Triton gRPC:    http://127.0.0.1:8004
# → Triton metrics: http://127.0.0.1:8005
```

## Local Inference: PyTorch + Triton/ONNX side-by-side
```bash
# Terminal 1 — PyTorch
python -m src.inference_service          # → port 8002

# Terminal 2 — Triton ONNX (already running)
docker start triton-inference-service    # → port 8003
```

## Triton + TensorRT — Remote GPU only (Vast.ai)
> Cannot run locally — requires NVIDIA GPU. Use `./deploy_tensorrt_gpu.sh` to push to a remote instance.
> Image: `dahlianadkarni/photo-duplicate-triton:tensorrt-gpu`

---

## App Server with Remote Inference
```bash
export INFERENCE_SERVICE_URL=http://127.0.0.1:<forwarded-port>
python -m src.ui.main
```

---

## Ports
| Port | Service |
|------|---------|
| 8080 | Main UI |
| 8081 | Demo UI |
| 8002 | Local inference — PyTorch (native or Docker) |
| 8003 | Local inference — Triton HTTP (ONNX, Docker) |
| 8004 | Local inference — Triton gRPC (ONNX, Docker) |
| 8005 | Local inference — Triton metrics (ONNX, Docker) |
| 8092 | Local inference — K8s kind (NodePort, PyTorch CPU) |

---

## Kubernetes (kind) — Step 8

### Cluster status
```bash
export PATH="/opt/homebrew/bin:$PATH"
kubectl get nodes
kubectl get pods -n inference
kubectl get hpa -n inference
```

### Apply / redeploy manifests
```bash
kubectl apply -k k8s/
kubectl rollout status deployment/inference-deploy -n inference
```

### Smoke test
```bash
curl localhost:8092/health
# {"status":"ok"}
```

### Watch HPA live
```bash
watch -n2 kubectl get hpa -n inference
kubectl describe hpa inference-hpa -n inference   # scale events
```

### Load test (triggers HPA scale-up)
```bash
# Requires a valid base64 JPEG payload file:
python -c "
import base64, json
with open('test_image.jpg','rb') as f:
    b64 = base64.b64encode(f.read()).decode()
print(json.dumps({'images': [b64]}))
" > /tmp/payload.json

hey -n 600 -c 30 -m POST \
  -H 'Content-Type: application/json' \
  -D /tmp/payload.json \
  http://localhost:8092/embed/base64
```

### Clean benchmark (50 req, conc=5)
```bash
hey -n 50 -c 5 -m POST \
  -H 'Content-Type: application/json' \
  -D /tmp/payload.json \
  http://localhost:8092/embed/base64
```

### Tear down / recreate cluster
```bash
kind delete cluster --name inference-cluster
kind create cluster --config kind-config.yaml
kind load docker-image photo-duplicate-inference:k8s-cpu --name inference-cluster
kubectl apply -k k8s/
```
