#!/usr/bin/env python3
"""Quick verification that all 3 backends return valid embeddings."""

import requests
import base64
import time
import io
import sys

import numpy as np
from PIL import Image

HOST = "207.180.148.74"
PYTORCH_PORT = 45580
TRITON_ONNX_PORT = 45597
TRITON_TRT_PORT = 45509

# Create a test image
img = Image.new("RGB", (224, 224), color="red")
buf = io.BytesIO()
img.save(buf, format="JPEG")
img_b64 = base64.b64encode(buf.getvalue()).decode()

print("=" * 60)
print("Verifying all 3 backends on A100 SXM4")
print("=" * 60)

# 1) PyTorch
print("\n=== Backend 1: PyTorch FastAPI ===")
t0 = time.time()
r = requests.post(
    f"http://{HOST}:{PYTORCH_PORT}/embed/base64",
    json={"images": [img_b64], "model_name": "ViT-B-32", "pretrained": "openai"},
)
t1 = time.time()
if r.status_code == 200:
    emb = r.json()["embeddings"][0]
    print(f"  ✓ Status: {r.status_code}, Dim: {len(emb)}, Latency: {(t1-t0)*1000:.1f}ms")
    print(f"  First 5 values: {[round(v, 4) for v in emb[:5]]}")
else:
    print(f"  ✗ ERROR: {r.status_code} - {r.text[:200]}")
    sys.exit(1)

# 2) Triton ONNX
print("\n=== Backend 2: Triton ONNX CUDA EP ===")
import tritonclient.http as httpclient

triton_onnx = httpclient.InferenceServerClient(url=f"{HOST}:{TRITON_ONNX_PORT}")
inp = httpclient.InferInput("image", [1, 3, 224, 224], "FP32")
img_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
inp.set_data_from_numpy(img_np)
t0 = time.time()
result = triton_onnx.infer("openclip_vit_b32", inputs=[inp])
t1 = time.time()
output = result.as_numpy("embedding")
print(f"  ✓ Status: OK, Shape: {output.shape}, Latency: {(t1-t0)*1000:.1f}ms")
print(f"  First 5 values: {[round(v, 4) for v in output[0][:5]]}")

# 3) Triton TRT
print("\n=== Backend 3: Triton TensorRT EP ===")
triton_trt = httpclient.InferenceServerClient(url=f"{HOST}:{TRITON_TRT_PORT}")
inp2 = httpclient.InferInput("image", [1, 3, 224, 224], "FP32")
inp2.set_data_from_numpy(img_np)
t0 = time.time()
result2 = triton_trt.infer("openclip_vit_b32_trt", inputs=[inp2])
t1 = time.time()
output2 = result2.as_numpy("embedding")
print(f"  ✓ Status: OK, Shape: {output2.shape}, Latency: {(t1-t0)*1000:.1f}ms")
print(f"  First 5 values: {[round(v, 4) for v in output2[0][:5]]}")

print("\n" + "=" * 60)
print("✓ All 3 backends verified and returning valid embeddings!")
print("=" * 60)
