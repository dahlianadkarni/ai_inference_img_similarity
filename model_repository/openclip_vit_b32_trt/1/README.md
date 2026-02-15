# TensorRT Engine Placeholder

The `model.plan` file is generated at container startup by converting
the ONNX model to TensorRT format. This ensures the engine is built
for the exact GPU architecture of the deployment target.

## How the engine is created

1. The ONNX model from `openclip_vit_b32/1/model.onnx` is used as source
2. `trtexec` converts it to a TensorRT engine with FP16 precision
3. The engine is saved as `model.plan` in this directory
4. Triton loads the engine and serves it

## Manual conversion

To build the engine manually (on a GPU machine):

```bash
python scripts/convert_to_tensorrt.py --verify
```

Or using trtexec directly:

```bash
trtexec \
  --onnx=model_repository/openclip_vit_b32/1/model.onnx \
  --saveEngine=model_repository/openclip_vit_b32_trt/1/model.plan \
  --fp16 \
  --minShapes=image:1x3x224x224 \
  --optShapes=image:16x3x224x224 \
  --maxShapes=image:32x3x224x224
```
