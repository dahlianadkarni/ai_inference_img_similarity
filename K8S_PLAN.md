# Step 8: Local Kubernetes (kind) — Implementation Plan

---

## Will This Break the Existing Setup?

**No. This is 100% additive and non-destructive.**

Here's why:

| Existing setup | K8s (kind) setup |
|------|------|
| `docker-compose up` → container named `inference-service` | kind cluster = a Docker container with its own internal network |
| Ports 8002, 8003, 8004 on `localhost` | Kind exposes a *different* host port (e.g. `8092`) to avoid collision |
| `photo-duplicate-inference:latest` (GPU-capable, linux/amd64) | New tag: `photo-duplicate-inference:k8s-cpu` (CPU-only, built from same `Dockerfile`) |
| Managed by docker-compose | Managed by `kubectl` and kind |

You can run both simultaneously. To restore the old setup at any point: `docker compose up`. To tear down K8s: `kind delete cluster --name inference-cluster`. Nothing shared, nothing overwritten.

---

## Why a New Image Tag?

The existing `photo-duplicate-inference:latest` was built with `docker buildx --platform linux/amd64` for remote GPU deployment on Vast.ai. Reusing it locally on an M-series Mac with kind can cause confusion because:

1. The GPU-targeted image may have been built `linux/amd64` — kind on Apple Silicon needs `linux/arm64` or a multi-arch build
2. `:latest` is already in use by docker-compose; overwriting it mid-session could break `docker compose up` unexpectedly

**New tag**: `photo-duplicate-inference:k8s-cpu`
- Same source `Dockerfile` (CPU path, `python:3.11-slim`)
- Built locally for the native platform (`linux/arm64` on M-series, `linux/amd64` on Intel)
- Loaded directly into kind — never pushed to Docker Hub

---

## Documentation Strategy

Documentation is updated at the end of each phase, not as an afterthought at the end of the project:

| Phase | Docs updated |
|-------|-------------|
| 1 ✅ | `K8S_PLAN.md` — mark Phase 1 complete; `PLAN.md` — add Step 8 entry |
| 2 ✅ | `K8S_PLAN.md` — mark Phase 2 complete; `README.md` — add K8s section to architecture table |
| 3 ✅ | `K8S_PLAN.md` — mark Phase 3 complete; capture HPA scale events in `STEP_8_K8S_RESULTS.md` |
| 4 ✅ | `STEP_8_K8S_RESULTS.md` — kubectl observability examples with real output; presentation docs updated |
| 5 ✅ | PDB/quota deployed alongside Phase 2; noted in `STEP_8_K8S_RESULTS.md` |
| 6 | `STEP_8_K8S_RESULTS.md` — Helm deploy commands; update `PLAN.md` Step 8 as ✅ DONE |

---

## Phases

---

### Phase 1 ✅ — Prerequisites & Cluster Bootstrap
**Status**: COMPLETE — kind v0.31.0, metrics-server installed, `photo-duplicate-inference:k8s-cpu` (388 MB) loaded

**Goal**: kind cluster running, metrics-server installed, image loaded.

**Prerequisites to install (one-time):**
```bash
brew install kind       # Kubernetes-in-Docker
brew install kubectl    # K8s CLI
brew install hey        # HTTP load generator for HPA demo
```

**Steps:**

1. **Create `kind-config.yaml`** — declares port forwarding from host to the cluster's NodePort so `curl localhost:8092/health` "just works":
   ```yaml
   kind: Cluster
   apiVersion: kind.x-k8s.io/v1alpha4
   name: inference-cluster
   nodes:
     - role: control-plane
       extraPortMappings:
         - containerPort: 30092   # NodePort defined in service.yaml
           hostPort: 8092         # Host port (avoids collision with existing 8002)
           protocol: TCP
   ```

2. **Create the cluster**:
   ```bash
   kind create cluster --config kind-config.yaml
   ```

3. **Install metrics-server** (required for HPA CPU metrics):
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   # Patch for kind (disables TLS cert verification for internal comms):
   kubectl patch deployment metrics-server -n kube-system \
     --type=json \
     -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
   ```

4. **Build the K8s image**:
   ```bash
   docker build -t photo-duplicate-inference:k8s-cpu .
   ```

5. **Load image into kind** (no Docker Hub push needed):
   ```bash
   kind load docker-image photo-duplicate-inference:k8s-cpu --name inference-cluster
   ```

**Checkpoint**: `kubectl get nodes` shows `inference-cluster-control-plane   Ready`

**Documentation update:**
```bash
# In K8S_PLAN.md: mark Phase 1 ✅, record actual kind version
# In PLAN.md: add Step 8 entry (in progress)
```

---

### Phase 2 ✅ — Core Manifests (Namespace + ConfigMap + Deployment + Service)**Status**: COMPLETE — 2/2 pods Running; `curl localhost:8092/health → {"status":"ok"}`
**Goal**: Pods running and reachable from `localhost:8092`.

**Files to create under `k8s/`:**

```
k8s/
├── namespace.yaml        # Isolates everything under 'inference' namespace
├── configmap.yaml        # MODEL_NAME, MODEL_PRETRAINED, LOG_LEVEL
├── deployment.yaml       # 2 replicas, probes, resource requests/limits
├── service.yaml          # NodePort 30092 → pod 8002
└── kustomization.yaml    # kubectl apply -k k8s/  (applies all in order)
```

**Key design choices:**

*`deployment.yaml`*
- `replicas: 2` — baseline, HPA takes over in Phase 3
- `imagePullPolicy: Never` — uses the locally loaded image, never hits Docker Hub
- **readinessProbe** on `/health` with `initialDelaySeconds: 20` — model takes ~10–20s to load; K8s won't route traffic until ready
- **livenessProbe** on `/health` with `initialDelaySeconds: 60` — restarts a stuck pod after full startup window
- Resource requests required for HPA:
  ```yaml
  resources:
    requests:
      cpu: "500m"      # HPA calculates: actual_cpu / 500m
      memory: "1Gi"
    limits:
      cpu: "2000m"
      memory: "3Gi"
  ```

*`service.yaml`*
- Type: `NodePort` at port `30092`
- Combined with `kind-config.yaml` port mapping: `localhost:8092 → NodePort 30092 → Pod :8002`

**Apply and verify:**
```bash
kubectl apply -k k8s/
kubectl rollout status deploy/inference-deploy -n inference
curl localhost:8092/health
# → {"status":"healthy","model":"ViT-B-32",...}
```

**Checkpoint**: `curl localhost:8092/health` returns 200. Two pods running.

**Documentation update:**
```bash
# In K8S_PLAN.md: mark Phase 2 ✅, record actual pod startup time
# In README.md: add row to architecture/backend table for K8s deployment
# Create STEP_8_K8S_RESULTS.md with initial health check output
```

---

### Phase 3 ✅ — Horizontal Pod Autoscaler (HPA)
**Goal**: Under synthetic load, pods scale up automatically; idle pods scale back down.

**New file**: `k8s/hpa.yaml`
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-deploy
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60   # Scale up when avg pod CPU > 60% of request
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 180  # Wait 3 min before scaling down (prevents flapping)
    scaleUp:
      stabilizationWindowSeconds: 30
```

**Load test to trigger HPA:**
```bash
# Watch HPA in one terminal:
kubectl get hpa -n inference -w

# Watch pods in another:
kubectl get pods -n inference -w

# Generate load (another terminal):
hey -n 200 -c 20 -m POST \
  -H "Content-Type: application/json" \
  -d '{"images":["<base64_jpeg>"],"mode":"remote"}' \
  http://localhost:8092/embed
```

**Expected behavior:**
1. CPU crosses 60% → HPA adds pods (up to 6)
2. Load ends → 3-min stabilization window → pods scale back to 2

**Checkpoint**: `kubectl describe hpa inference-hpa -n inference` shows scale events.

**Documentation update:**
```bash
# In K8S_PLAN.md: mark Phase 3 ✅
# In STEP_8_K8S_RESULTS.md: paste kubectl describe hpa output showing scale-up/down events
# Record: time-to-scale-up, pod count at peak, time-to-scale-down
```

---

### Phase 4 ✅ — Observability & Operational Skills & Operational Skills**Status**: COMPLETE — load test run, all kubectl workflows exercised, results in STEP_8_K8S_RESULTS.md
**Goal**: Understand what's happening inside the cluster. Practice `kubectl` workflows.

**kubectl commands to learn at this phase:**

```bash
# View logs from all inference pods simultaneously
kubectl logs -n inference -l app=inference-service --tail=50 -f

# Exec into a running pod (like docker exec)
kubectl exec -n inference -it <pod-name> -- /bin/sh

# Resource usage (requires metrics-server from Phase 1)
kubectl top pods -n inference
kubectl top nodes

# Describe a pod (events, probe failures, OOMkill, etc.)
kubectl describe pod -n inference <pod-name>

# Simulate a rolling update (change image tag or env var)
kubectl set env deploy/inference-deploy -n inference LOG_LEVEL=debug
kubectl rollout status deploy/inference-deploy -n inference
kubectl rollout undo deploy/inference-deploy -n inference   # rollback

# Port-forward directly to one pod (bypass Service, for debugging)
kubectl port-forward -n inference pod/<pod-name> 8099:8002
curl localhost:8099/health
```

**Optional add-ons (all installable in kind):**
- **Kubernetes Dashboard**: visual overview of pods/deployments/HPA
- **Prometheus + Grafana** (via `kube-prometheus-stack` Helm chart): CPU/memory graphs that match what cloud providers show you in GKE/EKS consoles

**Documentation update:**
```bash
# In STEP_8_K8S_RESULTS.md: add "Operational Runbook" section with annotated kubectl output
# Record: what kubectl describe pod shows during a probe failure, rolling update, OOMkill
```

---

### Phase 5 ✅ (Deployed with Phase 2) — PodDisruptionBudget + ResourceQuota
**Goal**: Production hardening patterns. Low effort, high signal.

**`k8s/pdb.yaml`** — guarantees at least 1 pod stays alive during `kubectl drain` or rolling updates:
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-pdb
  namespace: inference
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: inference-service
```

**`k8s/resourcequota.yaml`** — caps total resource consumption for the namespace:
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: inference-quota
  namespace: inference
spec:
  hard:
    requests.cpu: "4"
    requests.memory: "8Gi"
    limits.cpu: "8"
    limits.memory: "12Gi"
    pods: "10"
```

---

### Phase 6 (Optional) — Helm Chart
**Goal**: Package the manifests so they can be deployed with a single command and environment-specific overrides (dev vs. prod CPU vs. prod GPU).

```
helm/inference-service/
├── Chart.yaml
├── values.yaml           # defaults: image tag, replicas, cpu requests, HPA settings
├── values-gpu.yaml       # GPU node selector, nvidia.com/gpu: 1
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── hpa.yaml
    ├── configmap.yaml
    └── _helpers.tpl
```

Deploy for local K8s:
```bash
helm install inference ./helm/inference-service -f helm/inference-service/values.yaml
```

Deploy to cloud GPU (EKS/GKE):
```bash
helm install inference ./helm/inference-service \
  -f helm/inference-service/values.yaml \
  -f helm/inference-service/values-gpu.yaml \
  --set image.tag=gpu-linux-amd64
```

**Documentation update:**
```bash
# In STEP_8_K8S_RESULTS.md: add Helm deploy commands with real values.yaml
# In PLAN.md: mark Step 8 as ✅ COMPLETE, add summary of what was learned
# In README.md: update K8s section with final architecture and commands
```

---

## Coexistence Summary

```
localhost:8002  →  docker-compose inference-service    (PyTorch FastAPI, GPU image, unchanged)
localhost:8003  →  docker-compose triton-onnx          (Triton ONNX, unchanged)
localhost:8092  →  kind cluster NodePort → K8s Pod     (PyTorch FastAPI, k8s-cpu image, NEW)

kind cluster:   Docker container "inference-cluster-control-plane"
                  └── completely separate network bridge
                  └── no shared volumes or ports with docker-compose

To stop K8s:    kind delete cluster --name inference-cluster
To restore:     docker compose up      ← nothing changed, works as before
```

---

## File Layout After All Phases

```
k8s/
├── kind-config.yaml         # Cluster definition (port mappings)
├── namespace.yaml
├── configmap.yaml
├── deployment.yaml
├── service.yaml
├── hpa.yaml                 # Phase 3
├── pdb.yaml                 # Phase 5
├── resourcequota.yaml       # Phase 5
└── kustomization.yaml       # kubectl apply -k k8s/

helm/                        # Phase 6 (optional)
└── inference-service/
    ├── Chart.yaml
    ├── values.yaml
    ├── values-gpu.yaml
    └── templates/

K8S_PLAN.md                  # This file
```

---

## What You'll Have Learned by the End

| Concept | Where it appears |
|---------|-----------------|
| Pod / ReplicaSet / Deployment layering | Phase 2 |
| readiness vs liveness probes | Phase 2 — model cold-start handling |
| Resource requests vs limits | Phase 2 — required for HPA |
| NodePort vs ClusterIP vs LoadBalancer | Phase 2 |
| HPA algorithm and stabilization windows | Phase 3 |
| kubectl operational workflows | Phase 4 |
| PodDisruptionBudget patterns | Phase 5 |
| Helm templating and multi-environment values | Phase 6 |
| How cloud K8s (GKE/EKS) maps to this local setup | All phases |
