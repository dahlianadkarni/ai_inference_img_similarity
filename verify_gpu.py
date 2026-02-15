#!/usr/bin/env python3
"""
Verify GPU is being used by the inference service.

Run this script to check:
1. CUDA availability
2. GPU device information  
3. Model location (CPU vs GPU)
4. Test inference with GPU monitoring

Usage:
    python verify_gpu.py http://142.112.39.215:50912
"""
import sys
import requests
import json

def check_gpu_info(base_url: str):
    """Check GPU information from the service."""
    print("="*60)
    print("GPU Verification Report")
    print("="*60)
    
    # Check if service is up
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"\n✓ Service is UP at {base_url}")
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Service is DOWN: {e}")
        return False
    
    # Check GPU info endpoint
    try:
        response = requests.get(f"{base_url}/gpu-info", timeout=5)
        if response.status_code == 404:
            print("\n⚠️  /gpu-info endpoint not available")
            print("   Deploy the updated server code to enable GPU verification")
            return False
        
        gpu_info = response.json()
        
        print("\n" + "="*60)
        print("GPU Configuration:")
        print("="*60)
        print(f"CUDA Available: {gpu_info.get('cuda_available')}")
        
        if gpu_info.get('cuda_available'):
            print(f"CUDA Version: {gpu_info.get('cuda_version')}")
            print(f"Device Count: {gpu_info.get('device_count')}")
            
            if gpu_info.get('devices'):
                print("\nGPU Devices:")
                for dev in gpu_info['devices']:
                    marker = "  ← ACTIVE" if dev.get('current_device') else ""
                    print(f"  [{dev['id']}] {dev['name']} ({dev['total_memory_gb']} GB){marker}")
            
            print(f"\nModel Device: {gpu_info.get('model_device', 'unknown')}")
            
            if gpu_info.get('model_on_gpu'):
                print("\n✅ SUCCESS: Model is running on GPU!")
            else:
                print("\n❌ WARNING: Model is NOT on GPU!")
                print("   Check CUDA_VISIBLE_DEVICES and docker --gpus flag")
        else:
            print("\n❌ CUDA NOT AVAILABLE")
            print("   Reasons:")
            print("   - Docker container not started with --gpus flag")
            print("   - NVIDIA drivers not installed on host")
            print("   - Wrong base image (need nvidia/cuda base)")
            
            if gpu_info.get('warning'):
                print(f"\n   Server warning: {gpu_info['warning']}")
        
        print("="*60)
        return gpu_info.get('model_on_gpu', False)
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Failed to get GPU info: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_gpu.py <service-url>")
        print("Example: python verify_gpu.py http://142.112.39.215:50912")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    success = check_gpu_info(base_url)
    
    if success:
        print("\n✅ GPU verification PASSED")
        sys.exit(0)
    else:
        print("\n❌ GPU verification FAILED")
        print("\nNext steps:")
        print("1. Check if your Vast.ai instance has GPU enabled")
        print("2. Verify Docker image is started with --gpus all")
        print("3. Check NVIDIA drivers: nvidia-smi")
        print("4. Review container logs for CUDA errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
