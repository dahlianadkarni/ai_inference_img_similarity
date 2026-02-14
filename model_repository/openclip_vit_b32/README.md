# Triton Model Repository — OpenCLIP ViT-B-32

This directory contains the ONNX model and Triton configuration for serving OpenCLIP ViT-B-32 embeddings via NVIDIA Triton Inference Server.

## Structure

```
model_repository/
└── openclip_vit_b32/          # Model name (must match config.pbtxt "name")
    ├── config.pbtxt           # Model configuration (batching, instances, I/O)
    ├── 1/                     # Version 1
    │   └── model.onnx        # ONNX model file (~352MB, git-ignored)
    └── README.md             # This file
```

## Setup

1. **Export model to ONNX:**
   ```bash
   python scripts/export_to_onnx.py --test
   ```

2. **Verify structure:**
   ```bash
   ls -la model_repository/openclip_vit_b32/1/model.onnx
   ```

3. **Test locally with Triton:**
   ```bash
   ./build_triton_local.sh
   ```

4. **Test Triton endpoints:**
   ```bash
   python scripts/test_triton_client.py
   ```

## Configuration Details

See `config.pbtxt` for the full annotated configuration.

### Dynamic Batching

| Setting | Value | Rationale |
|---------|-------|-----------|
| `preferred_batch_size` | 4, 8, 16, 32 | Progressive sizes from interactive to batch workloads |
| `max_queue_delay` | 5ms | Balance between latency and batch accumulation |
| `max_batch_size` | 32 | ~2.5GB VRAM at FP32; safe for 8GB+ GPUs |

**Trade-off:** Higher `max_queue_delay` → larger batches → better throughput, but adds latency. Tune to 1ms for interactive use, up to 50ms for batch-heavy workloads.

### Configuration Profiles

| Profile | `instance_group` | `optimization` | Use Case |
|---------|-------------------|----------------|----------|
| **Local (CPU)** | `KIND_CPU, count: 1` | None | Mac testing |
| **Cloud (GPU)** | `KIND_GPU, gpus: [0], count: 1` | `cuda { graphs: true }` | Vast.ai deployment |

The `deploy_triton_gpu.sh` script automatically swaps the config for GPU deployment.

### Input / Output

| Direction | Name | Type | Shape | Description |
|-----------|------|------|-------|-------------|
| Input | `image` | FP32 | `[batch, 3, 224, 224]` | Preprocessed image tensor (CHW, [0,1]) |
| Output | `embedding` | FP32 | `[batch, 512]` | L2-normalizable embedding vector |

## Model Versioning

To add a new version:
```bash
mkdir -p model_repository/openclip_vit_b32/2
cp new_model.onnx model_repository/openclip_vit_b32/2/model.onnx
```

Triton loads new versions automatically without restart (controlled by `version_policy` in config).
