"""
Demonstration of the client-service architecture.

This file shows the components and how they interact.
"""

# ==============================================================================
# ARCHITECTURE DIAGRAM
# ==============================================================================
"""
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Your Mac)                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         UI Browser (http://127.0.0.1:8000)          │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ Photo Review Interface                                       │  │  │
│  │  │  - Exact Duplicates                                          │  │  │
│  │  │  - Perceptual Duplicates                                     │  │  │
│  │  │  - AI Groups                                                │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                     ↓                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Client Application                       │  │
│  │  src/ui/app_v3.py                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ Responsibilities:                                            │  │  │
│  │  │  • Scan photos (AppleScript or directory)                    │  │  │
│  │  │  • Parse metadata (EXIF, hashes)                             │  │  │
│  │  │  • Call inference service for embeddings                     │  │  │
│  │  │  • Store embeddings locally                                  │  │  │
│  │  │  • Group and cluster images                                  │  │  │
│  │  │  • Manage UI state                                            │  │  │
│  │  │  • Delete photos via AppleScript                             │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                     ↓                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                  Inference Client Module                            │  │
│  │  src/inference_service/client.py                                   │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ class InferenceClient:                                       │  │  │
│  │  │  • health_check()                                            │  │  │
│  │  │  • get_model_info()                                          │  │  │
│  │  │  • embed_images_base64(images) → embeddings                 │  │  │
│  │  │  • embed_image_files_batched(paths) → embeddings            │  │  │
│  │  │                                                              │  │  │
│  │  │ Smart batching, error handling, retries                     │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                  Local Data Storage                                 │  │
│  │  • embeddings/                                                      │  │
│  │    ├── embeddings.npy        (raw embedding vectors)               │  │
│  │    ├── metadata.json          (file paths, model info)             │  │
│  │    └── similar_groups.json    (grouped and scored)                │  │
│  │  • scan_results.json           (from scanner)                      │  │
│  │  • .cache/                     (temp files, cache)                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                    ╔═══════════════════╬═══════════════════╗
                    │ HTTP (REST API)   │ (JSON/multipart)  │
                    ║ http://127.0.0.1:8001                ║
                    ╚═══════════════════╬═══════════════════╝
                                        ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                   INFERENCE SERVICE (Stateless)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Server                                   │  │
│  │  src/inference_service/server.py                                   │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ Endpoints:                                                   │  │  │
│  │  │  • GET /health                  [health check]              │  │  │
│  │  │  • GET /model-info              [current model details]     │  │  │
│  │  │  • POST /embed/base64           [from base64 strings]       │  │  │
│  │  │  • POST /embed/batch            [from file uploads]         │  │  │
│  │  │                                                              │  │  │
│  │  │ Each request is independent                                 │  │  │
│  │  │ No photo metadata stored                                    │  │  │
│  │  │ No state persistence                                        │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                     ↓                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                  Model Management (Singleton)                       │  │
│  │  src/inference_service/server.py:InferenceService                  │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ load_model(model_name, pretrained)                           │  │  │
│  │  │  • Loads OpenCLIP model on first request                     │  │  │
│  │  │  • Reuses same instance for subsequent requests              │  │  │
│  │  │  • No reload if model already loaded                         │  │  │
│  │  │                                                              │  │  │
│  │  │ embed_images(images, batch_size)                            │  │  │
│  │  │  • Generates embeddings in batches                           │  │  │
│  │  │  • Normalizes output (L2)                                    │  │  │
│  │  │  • Returns numpy array                                       │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                     ↓                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                   PyTorch + OpenCLIP Model                          │  │
│  │  src/embedding/embedder.py:ImageEmbedder                           │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ Architecture: ViT-B-32 (OpenAI pretrained)                   │  │  │
│  │  │  • Vision Transformer (ViT) encoder                          │  │  │
│  │  │  • 512-dimensional embeddings                                │  │  │
│  │  │  • Runs on device: MPS (Apple Silicon) / CUDA / CPU          │  │  │
│  │  │                                                              │  │  │
│  │  │ Input: RGB image (224×224)                                  │  │  │
│  │  │ Output: Normalized embedding vector (512,)                  │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
"""

# ==============================================================================
# DATA FLOW EXAMPLES
# ==============================================================================

# Example 1: Scanning + Embedding Generation (Remote Mode)
"""
1. User clicks "Generate Embeddings" in UI

2. UI Backend (app_v3.py):
   ├─ Reads scan_for_embeddings.json
   ├─ Creates InferenceClient("http://127.0.0.1:8001")
   └─ For each batch of images:
      ├─ Load images from disk
      ├─ Encode to base64
      ├─ POST to /embed/base64
      └─ Receive embeddings as JSON

3. Inference Service:
   ├─ Receives HTTP POST request
   ├─ Decode base64 images
   ├─ Load model (if not already loaded)
   ├─ Generate embeddings
   └─ Return JSON: {"embeddings": [[...], [...]], "model_info": {...}}

4. UI Backend:
   ├─ Store embeddings in embeddings/
   ├─ Run grouping algorithm
   ├─ Save similar_groups.json
   └─ Return results to browser

5. User sees groups in UI
"""

# Example 2: Local Mode (Original Behavior)
"""
1. User runs: python -m src.embedding.main_v2 scan_for_embeddings.json --mode local

2. CLI (main_v2.py):
   ├─ Load scan_for_embeddings.json
   ├─ Load ImageEmbedder directly (no HTTP)
   ├─ Load images from disk
   ├─ Generate embeddings in-process
   ├─ Save to embeddings/
   ├─ Run grouping algorithm
   └─ Save similar_groups.json

3. No network, no service
"""

# Example 3: Auto Mode (Smart Fallback)
"""
1. User runs: python -m src.embedding.main_v2 scan_for_embeddings.json

2. CLI (main_v2.py):
   ├─ Try to connect to service at http://127.0.0.1:8001
   │  ├─ If available:
   │  │  └─ Use remote mode (see Example 1)
   │  │
   │  └─ If not available:
   │     └─ Fall back to local mode (see Example 2)
   │
   └─ Run grouping algorithm
"""

# ==============================================================================
# WHY THIS ARCHITECTURE MATTERS
# ==============================================================================

"""
This architecture is exactly what these production systems assume:

1. TRITON INFERENCE SERVER (NVIDIA)
   - Stateless inference API
   - Model management (loading, versioning)
   - Batch processing
   - Multi-GPU support
   - Model composition
   → Your code: Replace /embed/batch with Triton gRPC

2. TORCHSERVE (PyTorch)
   - Python-based model serving
   - Custom handlers
   - Batching, caching
   - Model versioning
   - Custom metrics
   → Your code: Wrap model in TorchServe handler

3. VLLM (LLM Serving)
   - Ultra-fast LLM inference
   - Continuous batching
   - GPU utilization optimization
   - OpenAI-compatible API
   → Your code: Call vLLM instead of your service

4. KUBERNETES DEPLOYMENTS
   - Client and service as separate containers
   - Inference service can scale independently
   - Load balancing across service instances
   - Different resource limits per service
   → Your code: No changes needed, just deploy as containers

5. DISTRIBUTED SYSTEMS
   - Service on different machine
   - Client-service over actual network
   - Independent deployment cycles
   - Different tech stacks (Python service, Node.js UI, etc.)
   → Your code: Change service URL, everything else works

KEY INSIGHT:
By building this NOW, you're learning the mental model that powers
production ML systems. The architecture doesn't change when you:
- Add GPU inference
- Scale to multiple users
- Deploy to cloud
- Use different frameworks

It's the same: client → (HTTP) → service → (inference) → embeddings
"""

# ==============================================================================
# NEXT STEPS FOR LEARNING
# ==============================================================================

"""
1. UNDERSTAND THE CURRENT SETUP
   ✓ Start both services: python start_services.py
   ✓ Generate embeddings via remote mode
   ✓ Observe the request/response flow
   ✓ Read ARCHITECTURE_REFACTOR.md

2. ADD OBSERVABILITY
   □ Add logging to see request details
   □ Time how long embedding takes
   □ Compare: local mode vs. remote mode
   □ Check if service reuses model efficiently

3. EXPERIMENT WITH DISTRIBUTION
   □ Start inference service on different port: --port 8002
   □ Start UI pointing to different service URL
   □ Add network latency simulation
   □ What's the overhead of HTTP vs. direct calls?

4. PREPARE FOR REAL DISTRIBUTION
   □ Extract inference service to separate repo
   □ Create Dockerfile for service
   □ Create Docker Compose for local development
   □ Plan: What would change for GPU server?

5. LEARN PRODUCTION PATTERNS
   □ Add request/response validation
   □ Add error handling and retries
   □ Add metrics (request count, latency, errors)
   □ Add API authentication (if needed)
   □ Add rate limiting

6. TRY FRAMEWORKS
   □ Replace FastAPI service with Triton
   □ Replace FastAPI service with TorchServe
   □ See how little changes in your client code

This is how you build production AI systems. Start simple, iterate,
add layers as needed. You're on the right path.
"""

if __name__ == "__main__":
    import sys
    
    print(__doc__)
    
    print("\n" + "="*70)
    print("This is a documentation file. Read the diagrams and explanations above.")
    print("="*70)
    print("\nTo get started:")
    print("  1. Read ARCHITECTURE_REFACTOR.md for full details")
    print("  2. Read QUICKSTART_ARCHITECTURE.md for quick commands")
    print("  3. Run: python start_services.py")
    print("  4. Open: http://127.0.0.1:8000")
    print("="*70)
