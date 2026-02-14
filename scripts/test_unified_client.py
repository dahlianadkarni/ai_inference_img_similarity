#!/usr/bin/env python3
"""
Test the unified InferenceClient with both PyTorch and Triton backends.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference_service.client import InferenceClient

def test_backend(backend: str, url: str):
    """Test a specific backend."""
    print(f"\n{'='*60}")
    print(f"Testing {backend.upper()} Backend")
    print(f"URL: {url}")
    print(f"{'='*60}\n")
    
    # Create client
    client = InferenceClient(service_url=url, backend=backend)
    
    # Health check
    if not client.health_check():
        print(f"✗ {backend} server is not available")
        return False
    
    print(f"✓ {backend} server is healthy")
    
    # Get model info
    try:
        info = client.get_model_info()
        print(f"✓ Model info: {info}")
    except Exception as e:
        print(f"✗ Failed to get model info: {e}")
        return False
    
    # Create test images
    test_images = [
        Image.new('RGB', (224, 224), color=(255, 0, 0)),  # Red
        Image.new('RGB', (224, 224), color=(0, 255, 0)),  # Green
        Image.new('RGB', (224, 224), color=(0, 0, 255)),  # Blue
    ]
    
    # Test inference
    try:
        embeddings = client.embed_images_base64(test_images)
        print(f"✓ Inference successful")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Expected shape: (3, 512)")
        
        if embeddings.shape == (3, 512):
            print(f"  ✓ Shape is correct")
        else:
            print(f"  ✗ Shape mismatch!")
        
        # Check embedding properties
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  L2 norms: {norms}")
        print(f"  Mean L2 norm: {norms.mean():.4f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("="*60)
    print("Unified Inference Client Test")
    print("="*60)
    
    results = {}
    
    # Test PyTorch backend (if running)
    try:
        results['pytorch'] = test_backend('pytorch', 'http://localhost:8002')
    except Exception as e:
        print(f"\n✗ PyTorch test failed: {e}")
        results['pytorch'] = False
    
    # Test Triton backend
    try:
        results['triton'] = test_backend('triton', 'http://localhost:8003')
    except Exception as e:
        print(f"\n✗ Triton test failed: {e}")
        results['triton'] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for backend, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{backend.upper():10s}: {status}")
    
    print(f"\n{'='*60}")
    print("Usage Examples")
    print(f"{'='*60}")
    print("\n# Use PyTorch backend (default):")
    print("export INFERENCE_SERVICE_URL=http://localhost:8002")
    print("python -m src.ui.main")
    
    print("\n# Use Triton backend:")
    print("export INFERENCE_BACKEND=triton")
    print("export INFERENCE_SERVICE_URL=http://localhost:8003")
    print("python -m src.ui.main")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
