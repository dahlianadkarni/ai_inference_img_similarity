# Triton Model Repository

This directory contains model configurations for NVIDIA Triton Inference Server.

## Structure

```
model_repository/
└── openclip_vit_b32/          # Model name
    ├── config.pbtxt           # Model configuration
    ├── 1/                     # Version 1
    │   └── model.onnx        # ONNX model file (generated)
    └── README.md             # This file
```

## Setup

1. **Export model to ONNX:**
   ```bash
   python scripts/export_to_onnx.py
   ```

2. **Verify structure:**
   ```bash
   tree model_repository/
   ```

3. **Test with Triton:**
   ```bash
   docker run --rm -p 8003:8000 \
     -v $(pwd)/model_repository:/models \
     nvcr.io/nvidia/tritonserver:24.01-py3 \
     tritonserver --model-repository=/models
   ```

## Configuration Details

### Dynamic Batching
- **Preferred batch sizes**: 4, 8, 16, 32
- **Max queue delay**: 5ms
- Triton automatically batches requests for better GPU utilization

### Input
- **Name**: `image`
- **Type**: FP32
- **Shape**: `[batch_size, 3, 224, 224]`

### Output
- **Name**: `embedding`
- **Type**: FP32
- **Shape**: `[batch_size, 512]`

### Instance Configuration
- **GPU**: 1 instance on GPU 0
- **CUDA Graphs**: Enabled for performance

## Model Versioning

To add a new version:
```bash
mkdir -p model_repository/openclip_vit_b32/2
cp new_model.onnx model_repository/openclip_vit_b32/2/model.onnx
```

Triton will automatically load new versions without restart.
