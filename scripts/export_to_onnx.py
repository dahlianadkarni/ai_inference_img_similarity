#!/usr/bin/env python3
"""
Export OpenCLIP model to ONNX format for Triton Inference Server.

This script:
1. Loads the OpenCLIP model (ViT-B-32)
2. Exports it to ONNX format
3. Verifies the ONNX model produces the same outputs as PyTorch
4. Saves to model_repository/ for Triton
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import open_clip

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def export_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    output_dir: Path = None,
    verify: bool = True
):
    """
    Export OpenCLIP model to ONNX.
    
    Args:
        model_name: Model architecture
        pretrained: Pretrained weights
        output_dir: Directory to save ONNX model
        verify: Whether to verify ONNX outputs match PyTorch
    """
    print(f"Loading OpenCLIP model: {model_name} ({pretrained})")
    
    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained
    )
    model.eval()
    
    # Get model device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Create dummy input (batch_size=1, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "model_repository" / "openclip_vit_b32" / "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "model.onnx"
    
    print(f"\nExporting to ONNX: {output_path}")
    
    # Export to ONNX
    torch.onnx.export(
        model.visual,  # Only export visual encoder (not text encoder)
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "embedding": {0: "batch_size"}
        }
    )
    
    print(f"✓ ONNX model saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verify ONNX model
    if verify:
        print("\nVerifying ONNX model...")
        verify_onnx_model(output_path, model, device, dummy_input)
    
    return output_path


def verify_onnx_model(onnx_path: Path, pytorch_model, device: str, dummy_input: torch.Tensor):
    """
    Verify ONNX model produces same outputs as PyTorch.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        device: Device (cpu or cuda)
        dummy_input: Test input tensor
    """
    import onnx
    import onnxruntime as ort
    
    # Check ONNX model is valid
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model.visual(dummy_input)
        pytorch_output = pytorch_output.cpu().numpy()
    
    # Get ONNX output
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    
    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"\nOutput comparison:")
    print(f"  PyTorch shape: {pytorch_output.shape}")
    print(f"  ONNX shape:    {onnx_output.shape}")
    print(f"  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    # Check if outputs are close
    if np.allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5):
        print("✓ ONNX outputs match PyTorch (within tolerance)")
        return True
    else:
        print("⚠ Warning: ONNX outputs differ from PyTorch")
        print("  This may be acceptable depending on use case")
        return False


def test_onnx_inference(onnx_path: Path):
    """
    Test ONNX model inference with a sample image.
    
    Args:
        onnx_path: Path to ONNX model
    """
    import onnxruntime as ort
    
    print("\n" + "="*60)
    print("Testing ONNX inference with sample image")
    print("="*60)
    
    # Create a dummy image
    dummy_image = np.random.rand(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    ort_session = ort.InferenceSession(str(onnx_path))
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    print(f"Input:  {input_name} {ort_session.get_inputs()[0].shape}")
    print(f"Output: {output_name} {ort_session.get_outputs()[0].shape}")
    
    result = ort_session.run([output_name], {input_name: dummy_image})
    embedding = result[0]
    
    print(f"\nInference result:")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding L2 norm: {np.linalg.norm(embedding):.4f}")
    print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print("✓ ONNX inference successful")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export OpenCLIP model to ONNX")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test inference after export"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("OpenCLIP to ONNX Export")
    print("="*60)
    
    # Export model
    onnx_path = export_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        output_dir=args.output_dir,
        verify=not args.no_verify
    )
    
    # Test inference
    if args.test:
        test_onnx_inference(onnx_path)
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Verify model repository structure:")
    print(f"   model_repository/openclip_vit_b32/1/model.onnx")
    print(f"2. Create config.pbtxt for Triton")
    print(f"3. Build Triton Docker image")
    print(f"4. Test locally: docker run -p 8003:8000 triton-image")


if __name__ == "__main__":
    main()
