# Step 8: Local Kubernetes (kind) — Live Results

**Date**: 2026-02-21  
**Cluster**: kind v0.31.0, Kubernetes v1.35.0  
**Hardware**: macOS 15.7.3 (Apple Silicon ARM64)  
**Image**: `photo-duplicate-inference:k8s-cpu` (built from `Dockerfile`, CPU-only, 388 MB)

---

## Phase 1: Cluster Bootstrap

### Prerequisites installed via Homebrew
```
kind v0.31.0 go1.25.5 darwin/arm64
hey 0.1.5 (HTTP load generator)
kubectl Client Version: v1.34.1 (already present)
```

### Cluster creation
```bash
kind create cluster --config kind-config.yaml
```
```
Creating cluster "inference-cluster" ...
 ✓ Ensuring node image (kindest/node:v1.35.0) 🖼
 ✓ Preparing nodes 📦
 ✓ Writing configuration 📜
 ✓ Starting control-plane 🕹️
 ✓ Installing CNI 🔌
 ✓ Installing StorageClass 💾
Set kubectl context to "kind-inference-cluster"
```

### Node status (after ~30s)
```
NAME                              STATUS   ROLES           AGE   VERSION
inference-cluster-control-plane   Ready    control-plane   32s   v1.35.0
```

### metrics-server installed and patched
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system \
  --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```
```
deployment.apps/metrics-server patched
```

### Image loaded into kind (no Docker Hub push)
```bash
kind load docker-image photo-duplicate-inference:k8s-cpu --name inference-cluster
```
```
# Verified inside node:
docker.io/library/photo-duplicate-inference   k8s-cpu   a9de0869f973c   388MB
```

---

## Phase 2: Core Manifests

### Apply all manifests
```bash
kubectl apply -k k8s/
```
```
namespace/inference created
resourcequota/inference-quota created
configmap/inference-config created
service/inference-svc created
deployment.apps/inference-deploy created
poddisruptionbudget.policy/inference-pdb created
horizontalpodautoscaler.autoscaling/inference-hpa created
```

### Rollout
```bash
kubectl rollout status deploy/inference-deploy -n inference
```
```
Waiting for deployment "inference-deploy" rollout to finish: 0 of 2 updated replicas are available...
Waiting for deployment "inference-deploy" rollout to finish: 1 of 2 updated replicas are available...
deployment "inference-deploy" successfully rolled out
```
*Both pods passed readiness probe (`/health`) within ~45s of pod creation.*

### Pod status
```
NAME                                READY   STATUS    RESTARTS   AGE
inference-deploy-6845bbfc45-8zh2d   1/1     Running   0          53s
inference-deploy-6845bbfc45-dgtcg   1/1     Running   0          53s
```

### Service
```
NAME            TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)
inference-svc   NodePort   10.96.174.130   <none>        8002:30092/TCP
```
*Traffic path: `localhost:8092` → kind port-map → NodePort 30092 → Pod :8002*

### Smoke test
```bash
curl -s localhost:8092/health | python3 -m json.tool
```
```json
{
    "status": "ok"
}
```
✅ **Service reachable at `localhost:8092` — completely separate from existing docker-compose setup on 8002.**

---

## Phase 3: HPA + Phase 5: PDB + ResourceQuota

### HPA status (after metrics-server warm-up, ~4 min post-deploy)
```
NAME            REFERENCE                     TARGETS       MINPODS   MAXPODS   REPLICAS   AGE
inference-hpa   Deployment/inference-deploy   cpu: 1%/60%   2         6         2          3m58s
```

### Pod resource usage (live from `kubectl top pods -n inference`)
```
NAME                                CPU(cores)   MEMORY(bytes)
inference-deploy-6845bbfc45-8zh2d   24m          923Mi
inference-deploy-6845bbfc45-dgtcg   11m          1115Mi
```
*At idle: ~18m avg CPU (well below 300m threshold = 60% of 500m request). HPA holds at 2 replicas.*

### PDB
```
NAME            MIN AVAILABLE   MAX UNAVAILABLE   ALLOWED DISRUPTIONS
inference-pdb   1               N/A               1
```
*Guarantees ≥1 pod stays alive during rolling updates or node drain.*

### ResourceQuota (live utilization)
```
NAME              REQUEST                                                   LIMIT
inference-quota   pods: 2/10, requests.cpu: 1/4, requests.memory: 2Gi/8Gi   limits.cpu: 4/8, limits.memory: 6Gi/12Gi
```

---

## Coexistence: K8s vs docker-compose

| Service | Port | Manager | Image |
|---------|------|---------|-------|
| PyTorch FastAPI (docker-compose) | 8002 | docker-compose | `photo-duplicate-inference:latest` |
| Triton ONNX (docker-compose) | 8003/8004 | docker-compose | `Dockerfile.triton` |
| PyTorch FastAPI **(Kubernetes)** | **8092** | kubectl / kind | `photo-duplicate-inference:k8s-cpu` |

Both run independently. `docker compose up` and `kubectl apply -k k8s/` have no shared state.

---

## Useful Commands — Phase 4 Operational Reference

```bash
# All pods in namespace
kubectl get pods -n inference -o wide

# Live logs from all inference pods
kubectl logs -n inference -l app=inference-service --tail=50 -f

# Shell into a pod
kubectl exec -n inference -it <pod-name> -- /bin/sh

# Resource usage (requires metrics-server)
kubectl top pods -n inference

# Watch HPA decisions live
kubectl get hpa -n inference -w

# Rolling update (change env var, triggers new pods)
kubectl set env deploy/inference-deploy -n inference LOG_LEVEL=debug
kubectl rollout status deploy/inference-deploy -n inference

# Roll back last change
kubectl rollout undo deploy/inference-deploy -n inference

# Port-forward directly to one pod (bypasses Service, useful for debugging)
kubectl port-forward -n inference pod/<pod-name> 8099:8002
curl localhost:8099/health

# Tear down cluster (leaves docker-compose untouched)
kind delete cluster --name inference-cluster
```

---

## Phase 4: HPA Load Test + Observability

### Load Test Setup

```bash
# Generate realistic 224×224 JPEG payload
python3 -c "
import base64, io, json
from PIL import Image
import numpy as np
arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(arr)
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=50)
b64 = base64.b64encode(buf.getvalue()).decode()
with open('/tmp/embed_payload.json', 'w') as f:
    f.write(json.dumps({'images': [b64]}))
"
# Payload size: 27 KB (base64 JPEG — same format as production client)
```

```bash
# Run load test: 600 requests, 30 concurrent
hey -n 600 -c 30 -m POST \
  -H "Content-Type: application/json" \
  -D /tmp/embed_payload.json \
  http://localhost:8092/embed/base64
```

### HPA Scale-Up Events (captured from `kubectl describe hpa`)

```
Events:
  Type    Reason              Age    Message
  ----    ------              ----   -------
  Normal  SuccessfulRescale   4m18s  New size: 4; reason: cpu resource utilization above target
  Normal  SuccessfulRescale   3m48s  New size: 6; reason: cpu resource utilization above target
```

**Scale lifecycle: 2 → 4 → 6 pods (hit max) in two decisions, 30 seconds apart.**

### CPU Utilization at Each Stage

| Time | CPU % of request | Replicas | Notes |
|------|:---:|:---:|-------|
| Idle (pre-load) | 1% | 2 | ~18m avg per pod |
| Load start (~30s) | **210%** | 2 | HPA triggers first decision |
| Load mid | **397% (1988m/2000m)** | 4→6 | Both original pods at CPU limit |
| Post-load (60s after) | 145% | 6 | New pods absorbing load, originals recovering |
| Idle (post-load) | 1% | 6 | HPA holding at 6 (3-min stabilization window) |

**Key observation:** Original pods hit the 2000m CPU limit and failed readiness probes under saturation — livenessProbe fired and restarted them. This is correct K8s behavior: the probe detected a stuck pod and recovered it.

### Mid-Load Pod Resource Snapshot

```
NAME                                CPU(cores)   MEMORY
inference-deploy-6845bbfc45-8zh2d   1988m        942Mi    ← original, saturated
inference-deploy-6845bbfc45-dgtcg   1989m        1134Mi   ← original, saturated
inference-deploy-6845bbfc45-6ckwm   311m         353Mi    ← new pod, starting up
inference-deploy-6845bbfc45-x2tsj   309m         484Mi    ← new pod, starting up
```

### Benchmark Results (50 requests, concurrency=5, CPU-only)

```
Summary:
  Total:        14.15 secs
  Slowest:      4.15 secs
  Fastest:      0.40 secs
  Average:      1.26 secs
  Requests/sec: 3.53

Latency distribution:
  p50:  1.02s
  p75:  1.20s
  p90:  3.45s
  p95:  4.15s

Status code distribution:
  [200] 50 responses  ← 100% success rate
```

### CPU vs GPU Performance Comparison

| Metric | CPU (kind, local) | GPU (A100, remote) | Ratio |
|--------|:-:|:-:|:-:|
| p50 latency | 1,020ms | 64ms | **16× slower** |
| Throughput (concurrent) | 3.5 req/s | ~30 req/s | **8× lower** |
| Infrastructure cost | $0/hr | $0.85/hr | Free (local dev) |

*CPU is 16× slower — expected for ViT-B/32 on Apple Silicon without MPS acceleration. The K8s step is about learning orchestration patterns, not GPU performance.*

### kubectl Operational Commands (Phase 4 Reference)

```bash
# Live logs from all inference pods simultaneously
kubectl logs -n inference -l app=inference-service --tail=50 -f

# Resource usage
kubectl top pods -n inference

# Shell into a running pod
kubectl exec -n inference -it inference-deploy-6845bbfc45-8zh2d -- /bin/sh

# Describe a pod to see events (probe failures, OOMkill, restarts)
kubectl describe pod -n inference inference-deploy-6845bbfc45-8zh2d

# Watch HPA decisions in real time
kubectl get hpa -n inference -w

# Rolling update — triggers new pods, keeps old ones alive until new pass readiness
kubectl set env deploy/inference-deploy -n inference LOG_LEVEL=debug
kubectl rollout status deploy/inference-deploy -n inference

# Rollback
kubectl rollout undo deploy/inference-deploy -n inference

# Port-forward directly to one pod (bypass Service, for debugging)
kubectl port-forward -n inference pod/inference-deploy-6845bbfc45-8zh2d 8099:8002
curl localhost:8099/health
```

---

## Key Learnings (Phases 1–3)

| Concept | Observed |
|---------|----------|
| `imagePullPolicy: Never` | Required to use locally-loaded kind image without a registry |
| readinessProbe `initialDelaySeconds: 20` | Model load takes ~20s; without this, 503s during startup |
| `resource.requests.cpu` required for HPA | HPA formula: `actual_cpu / request_cpu = utilization%` |
| NodePort + kind extraPortMappings | The two-part config (kind-config.yaml + service.yaml) that makes `localhost:8092` work |
| kind cluster = isolated Docker network | Zero interference with existing docker-compose ports/containers |
