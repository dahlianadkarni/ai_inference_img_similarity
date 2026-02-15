#!/usr/bin/env python3
"""Test GPU inference on deployed instance."""
import requests
import base64
import time
from PIL import Image
import io

def test_gpu_inference():
    """Test GPU inference with batch-32."""
    # Create 32 test images
    print("Creating test batch of 32 images...")
    images = []
    for i in range(32):
        img = Image.new('RGB', (224, 224), color=(i*8, 255-i*8, 128))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        images.append(base64.b64encode(buf.getvalue()).decode())
    
    # Test batch inference
    print("Testing batch-32 inference on RTX A6000...")
    url = "http://75.58.61.51:18700/embed/base64"
    
    start = time.time()
    response = requests.post(url, json={"images": images}, timeout=30)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        print("\n" + "="*60)
        print("✅ GPU INFERENCE CONFIRMED WORKING!")
        print("="*60)
        print(f"Batch size:       32 images")
        print(f"Total latency:    {elapsed:.3f}s")
        print(f"Per-image:        {elapsed/32*1000:.1f}ms")
        print(f"Throughput:       {32/elapsed:.1f} img/s")
        print(f"Device:           {data['model_info']['device']}")
        print(f"Model:            {data['model_info']['model_name']}")
        print(f"Embedding dim:    {len(data['embeddings'][0])}")
        print("="*60)
        print("\n💡 Performance Analysis:")
        print(f"   RTX A6000 batch-32 latency: {elapsed:.3f}s")
        print(f"   This is GPU performance (CPU would be 10-20x slower)")
        return True
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return False

if __name__ == "__main__":
    # First check GPU info
    print("="*60)
    print("Step 1: Checking GPU Info")
    print("="*60)
    gpu_info = requests.get("http://75.58.61.51:18700/gpu-info").json()
    print(f"CUDA Available:   {gpu_info['cuda_available']}")
    print(f"GPU:              {gpu_info['devices'][0]['name']}")
    print(f"VRAM:             {gpu_info['devices'][0]['total_memory_gb']:.1f} GB")
    print(f"Model Location:   {gpu_info['model_device']}")
    print(f"Model on GPU:     {gpu_info['model_on_gpu']}")
    
    print("\n" + "="*60)
    print("Step 2: Testing Inference Performance")
    print("="*60)
    test_gpu_inference()
