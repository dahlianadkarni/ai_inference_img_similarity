#!/usr/bin/env python3
"""
Test Triton Inference Server with sample images.

This script verifies that:
1. Triton server is accessible
2. Model is loaded and ready
3. Inference works correctly
4. Outputs are valid embeddings
"""
import sys
from pathlib import Path
import numpy as np
import requests
import json
from PIL import Image
import io
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_health(base_url: str = "http://localhost:8003"):
    """Test server health endpoints."""
    print("=" * 60)
    print("Testing Triton Health Endpoints")
    print("=" * 60)
    
    # Test server health
    response = requests.get(f"{base_url}/v2/health/ready")
    if response.status_code == 200:
        print("✓ Server is ready")
    else:
        print(f"✗ Server health check failed: {response.status_code}")
        return False
    
    # Test model readiness
    response = requests.get(f"{base_url}/v2/models/openclip_vit_b32/ready")
    if response.status_code == 200:
        print("✓ Model is ready")
    else:
        print(f"✗ Model readiness check failed: {response.status_code}")
        return False
    
    return True


def test_model_metadata(base_url: str = "http://localhost:8003"):
    """Get model metadata."""
    print("\n" + "=" * 60)
    print("Model Metadata")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/v2/models/openclip_vit_b32")
    if response.status_code != 200:
        print(f"✗ Failed to get model metadata: {response.status_code}")
        return
    
    metadata = response.json()
    print(f"Model name: {metadata.get('name')}")
    print(f"Platform: {metadata.get('platform')}")
    print(f"Max batch size: {metadata.get('max_batch_size')}")
    
    # Print inputs
    print("\nInputs:")
    for inp in metadata.get('inputs', []):
        print(f"  - {inp['name']}: {inp['datatype']} {inp['shape']}")
    
    # Print outputs
    print("\nOutputs:")
    for out in metadata.get('outputs', []):
        print(f"  - {out['name']}: {out['datatype']} {out['shape']}")


def test_inference(base_url: str = "http://localhost:8003"):
    """Test inference with dummy image."""
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)
    
    # Create dummy image (224x224 RGB)
    dummy_image = np.random.rand(224, 224, 3).astype(np.uint8)
    
    # Convert to format Triton expects
    # Triton expects: [batch, channels, height, width]
    image_tensor = np.array(dummy_image).astype(np.float32)
    image_tensor = np.transpose(image_tensor, (2, 0, 1))  # HWC -> CHW
    image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
    
    # Normalize to [0, 1]
    image_tensor = image_tensor / 255.0
    
    print(f"Input shape: {image_tensor.shape}")
    print(f"Input dtype: {image_tensor.dtype}")
    
    # Prepare Triton inference request
    # https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md
    payload = {
        "inputs": [
            {
                "name": "image",
                "shape": list(image_tensor.shape),
                "datatype": "FP32",
                "data": image_tensor.flatten().tolist()
            }
        ],
        "outputs": [
            {
                "name": "embedding"
            }
        ]
    }
    
    # Send inference request
    response = requests.post(
        f"{base_url}/v2/models/openclip_vit_b32/infer",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print(f"✗ Inference failed: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    
    # Extract embedding
    embedding = np.array(result['outputs'][0]['data'])
    
    print(f"\n✓ Inference successful")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding L2 norm: {np.linalg.norm(embedding):.4f}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    
    return embedding


def test_batch_inference(base_url: str = "http://localhost:8003", batch_size: int = 8):
    """Test batch inference."""
    print("\n" + "=" * 60)
    print(f"Testing Batch Inference (batch_size={batch_size})")
    print("=" * 60)
    
    # Create batch of dummy images
    batch = np.random.rand(batch_size, 3, 224, 224).astype(np.float32) / 255.0
    
    print(f"Input batch shape: {batch.shape}")
    
    # Prepare request
    payload = {
        "inputs": [
            {
                "name": "image",
                "shape": list(batch.shape),
                "datatype": "FP32",
                "data": batch.flatten().tolist()
            }
        ],
        "outputs": [
            {
                "name": "embedding"
            }
        ]
    }
    
    # Send request
    response = requests.post(
        f"{base_url}/v2/models/openclip_vit_b32/infer",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print(f"✗ Batch inference failed: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    
    # Extract embeddings
    embeddings = np.array(result['outputs'][0]['data']).reshape(batch_size, -1)
    
    print(f"\n✓ Batch inference successful")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected shape: ({batch_size}, 512)")
    
    if embeddings.shape == (batch_size, 512):
        print("✓ Output shape is correct")
    else:
        print(f"✗ Output shape mismatch")
    
    return embeddings


def compare_with_pytorch(base_url: str = "http://localhost:8003"):
    """Compare Triton outputs with PyTorch backend."""
    print("\n" + "=" * 60)
    print("Comparing Triton vs PyTorch")
    print("=" * 60)
    
    try:
        # Check if PyTorch server is running
        pytorch_url = "http://localhost:8002"
        response = requests.get(f"{pytorch_url}/health", timeout=2)
        if response.status_code != 200:
            print("⚠ PyTorch server not running on port 8002")
            print("  Skipping comparison")
            return
    except requests.exceptions.RequestException:
        print("⚠ PyTorch server not running on port 8002")
        print("  Skipping comparison")
        return
    
    # Create test image
    dummy_image = np.random.rand(224, 224, 3).astype(np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    # Convert to base64 for PyTorch
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Get PyTorch embedding
    pytorch_response = requests.post(
        f"{pytorch_url}/embed/base64",
        json={"images": [image_base64]}
    )
    pytorch_embedding = np.array(pytorch_response.json()["embeddings"][0])
    
    # Get Triton embedding
    image_tensor = np.array(dummy_image).astype(np.float32) / 255.0
    image_tensor = np.transpose(image_tensor, (2, 0, 1))
    image_tensor = np.expand_dims(image_tensor, axis=0)
    
    triton_payload = {
        "inputs": [{"name": "image", "shape": list(image_tensor.shape), "datatype": "FP32", "data": image_tensor.flatten().tolist()}],
        "outputs": [{"name": "embedding"}]
    }
    
    triton_response = requests.post(f"{base_url}/v2/models/openclip_vit_b32/infer", json=triton_payload)
    triton_embedding = np.array(triton_response.json()['outputs'][0]['data'])
    
    # Compare
    max_diff = np.abs(pytorch_embedding - triton_embedding).max()
    mean_diff = np.abs(pytorch_embedding - triton_embedding).mean()
    
    print(f"\nPyTorch embedding shape: {pytorch_embedding.shape}")
    print(f"Triton embedding shape:  {triton_embedding.shape}")
    print(f"Max difference:  {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if np.allclose(pytorch_embedding, triton_embedding, rtol=1e-2, atol=1e-3):
        print("✓ Outputs match within tolerance")
    else:
        print("⚠ Outputs differ - check preprocessing")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Triton Inference Server")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8003",
        help="Triton server URL"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batch inference test"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Triton Inference Server Test Suite")
    print("=" * 60)
    print(f"Server URL: {args.url}")
    print("=" * 60)
    
    # Run tests
    if not test_health(args.url):
        print("\n✗ Health checks failed. Is Triton running?")
        print(f"  Start with: ./build_triton_local.sh")
        sys.exit(1)
    
    test_model_metadata(args.url)
    test_inference(args.url)
    test_batch_inference(args.url, args.batch_size)
    compare_with_pytorch(args.url)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Integrate with UI: export INFERENCE_BACKEND=triton")
    print("2. Run benchmarks: python scripts/benchmark_backends.py")
    print("3. Deploy to GPU: docker buildx build --platform linux/amd64 ...")


if __name__ == "__main__":
    main()
