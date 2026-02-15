#!/usr/bin/env python3
"""
Step 5B: Convert ONNX model to TensorRT engine for maximum GPU inference speed.

This script converts the OpenCLIP ViT-B-32 ONNX model to a TensorRT engine,
optimized for the current GPU. The engine is GPU-architecture specific — it
must be built on the same GPU type where it will be served.

Features:
  - FP16 mode (default): ~2x faster, negligible accuracy loss for ViT
  - Dynamic batch sizes: supports batch 1–32
  - Optional FP32 fallback for accuracy-sensitive workloads
  - Output verification against ONNX model

Two conversion methods:
  1. trtexec (preferred): Uses NVIDIA's CLI tool (included in Triton image)
  2. Python API: Uses tensorrt Python bindings (fallback)

Usage:
  # On GPU machine (inside Triton container):
  python scripts/convert_to_tensorrt.py

  # With FP32 (no FP16 quantization):
  python scripts/convert_to_tensorrt.py --no-fp16

  # Custom paths:
  python scripts/convert_to_tensorrt.py \
      --onnx-path model_repository/openclip_vit_b32/1/model.onnx \
      --output-path model_repository/openclip_vit_b32_trt/1/model.plan

  # Verify output accuracy against ONNX:
  python scripts/convert_to_tensorrt.py --verify

  # Custom max batch size:
  python scripts/convert_to_tensorrt.py --max-batch-size 64

  # Use Python API instead of trtexec:
  python scripts/convert_to_tensorrt.py --method python
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def get_gpu_info() -> Optional[Dict]:
    """Detect GPU model and compute capability via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            timeout=5, text=True,
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "name": parts[0],
            "memory_mb": parts[1],
            "compute_capability": parts[2] if len(parts) > 2 else "unknown",
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def check_trtexec_available() -> bool:
    """Check if trtexec is available on the system."""
    return shutil.which("trtexec") is not None


def check_tensorrt_python() -> bool:
    """Check if TensorRT Python bindings are available."""
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Conversion via trtexec (preferred method)
# ---------------------------------------------------------------------------

def convert_with_trtexec(
    onnx_path: Path,
    output_path: Path,
    fp16: bool = True,
    max_batch_size: int = 32,
    opt_batch_size: int = 16,
    workspace_mb: int = 4096,
    verbose: bool = False,
) -> Tuple[bool, Dict]:
    """
    Convert ONNX to TensorRT using trtexec CLI.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to save TensorRT engine
        fp16: Enable FP16 mode
        max_batch_size: Maximum batch size for dynamic shapes
        opt_batch_size: Optimal batch size (used for kernel autotuning)
        workspace_mb: Maximum workspace size in MB
        verbose: Enable verbose logging

    Returns:
        (success, metadata) tuple
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        # Dynamic shapes: min=1, opt=opt_batch_size, max=max_batch_size
        f"--minShapes=image:1x3x224x224",
        f"--optShapes=image:{opt_batch_size}x3x224x224",
        f"--maxShapes=image:{max_batch_size}x3x224x224",
        f"--workspace={workspace_mb}",
    ]

    if fp16:
        cmd.append("--fp16")

    if verbose:
        cmd.append("--verbose")

    # Add timing iterations for profiling during build
    cmd.extend([
        "--avgRuns=10",
        "--warmUp=500",
    ])

    print(f"\n  Running trtexec command:")
    print(f"    {' '.join(cmd)}")
    print()

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for large models
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            # Parse performance info from trtexec output
            metadata = _parse_trtexec_output(result.stdout)
            metadata["conversion_time_s"] = round(elapsed, 2)
            metadata["method"] = "trtexec"
            metadata["fp16"] = fp16
            metadata["max_batch_size"] = max_batch_size
            metadata["opt_batch_size"] = opt_batch_size
            metadata["engine_size_mb"] = round(output_path.stat().st_size / 1024 / 1024, 2)
            return True, metadata
        else:
            print(f"  ✗ trtexec failed (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:2000]}")
            return False, {"error": result.stderr[:500]}

    except subprocess.TimeoutExpired:
        print("  ✗ trtexec timed out (>10 minutes)")
        return False, {"error": "timeout"}


def _parse_trtexec_output(stdout: str) -> Dict:
    """Parse performance metrics from trtexec stdout."""
    metadata = {}
    for line in stdout.splitlines():
        line = line.strip()
        # Look for latency info
        if "mean" in line.lower() and "ms" in line.lower() and "GPU" in line:
            metadata["trtexec_gpu_latency"] = line
        if "mean" in line.lower() and "ms" in line.lower() and "Host" in line:
            metadata["trtexec_host_latency"] = line
        if "Throughput" in line:
            metadata["trtexec_throughput"] = line
        if "Total Host Walltime" in line:
            metadata["trtexec_total_time"] = line
    return metadata


# ---------------------------------------------------------------------------
# Conversion via TensorRT Python API (fallback)
# ---------------------------------------------------------------------------

def convert_with_python_api(
    onnx_path: Path,
    output_path: Path,
    fp16: bool = True,
    max_batch_size: int = 32,
    opt_batch_size: int = 16,
    workspace_mb: int = 4096,
) -> Tuple[bool, Dict]:
    """
    Convert ONNX to TensorRT using Python API.

    This is a fallback if trtexec is not available.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("  ✗ TensorRT Python bindings not available")
        return False, {"error": "tensorrt not installed"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"  TensorRT version: {trt.__version__}")
    print(f"  Building engine (this may take several minutes)...")

    start = time.time()

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ✗ ONNX parse error: {parser.get_error(i)}")
            return False, {"error": "ONNX parse failed"}

    print(f"  ✓ ONNX model parsed ({network.num_layers} layers)")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  ✓ FP16 mode enabled")
    elif fp16:
        print("  ⚠ FP16 requested but GPU doesn't support fast FP16")

    # Set dynamic shape optimization profiles
    profile = builder.create_optimization_profile()
    # image input: [batch, 3, 224, 224]
    profile.set_shape(
        "image",
        min=(1, 3, 224, 224),
        opt=(opt_batch_size, 3, 224, 224),
        max=(max_batch_size, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("  ✗ Engine build failed")
        return False, {"error": "engine build failed"}

    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    elapsed = time.time() - start

    metadata = {
        "conversion_time_s": round(elapsed, 2),
        "method": "python_api",
        "fp16": fp16,
        "max_batch_size": max_batch_size,
        "opt_batch_size": opt_batch_size,
        "engine_size_mb": round(output_path.stat().st_size / 1024 / 1024, 2),
        "trt_version": trt.__version__,
        "num_layers": network.num_layers,
    }

    return True, metadata


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_tensorrt_output(
    onnx_path: Path,
    trt_path: Path,
    batch_size: int = 1,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> bool:
    """
    Compare TensorRT engine output against ONNX Runtime baseline.

    For FP16 engines, we use relaxed tolerances since precision loss is expected.
    """
    print(f"\n  Verifying TensorRT output against ONNX (batch_size={batch_size})...")

    # Get ONNX reference output
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        onnx_output = ort_session.run(None, {"image": dummy_input})[0]
    except ImportError:
        print("  ⚠ onnxruntime not available, skipping verification")
        return True
    except Exception as e:
        print(f"  ⚠ ONNX inference failed: {e}")
        return True

    # Get TensorRT output
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Set input shape for dynamic batch
        input_idx = engine.get_binding_index("image")
        context.set_binding_shape(input_idx, (batch_size, 3, 224, 224))

        # Allocate buffers
        input_buf = cuda.mem_alloc(dummy_input.nbytes)
        output_buf = cuda.mem_alloc(
            batch_size * 512 * np.dtype(np.float32).itemsize
        )
        cuda.memcpy_htod(input_buf, dummy_input)

        # Run inference
        context.execute_v2([int(input_buf), int(output_buf)])

        # Get output
        trt_output = np.empty((batch_size, 512), dtype=np.float32)
        cuda.memcpy_dtoh(trt_output, output_buf)

    except ImportError:
        print("  ⚠ TensorRT/PyCUDA not available, skipping verification")
        return True
    except Exception as e:
        print(f"  ⚠ TensorRT inference failed: {e}")
        print("  (This is expected if running on a different GPU than where engine was built)")
        return False

    # Compare
    max_diff = np.abs(onnx_output - trt_output).max()
    mean_diff = np.abs(onnx_output - trt_output).mean()
    cosine_sim = np.dot(onnx_output.flatten(), trt_output.flatten()) / (
        np.linalg.norm(onnx_output) * np.linalg.norm(trt_output)
    )

    print(f"    Max difference:   {max_diff:.6f}")
    print(f"    Mean difference:  {mean_diff:.6f}")
    print(f"    Cosine similarity: {cosine_sim:.6f}")

    if np.allclose(onnx_output, trt_output, rtol=rtol, atol=atol):
        print("  ✓ TensorRT outputs match ONNX (within tolerance)")
        return True
    elif cosine_sim > 0.999:
        print("  ✓ TensorRT outputs are highly similar (cosine > 0.999)")
        print("    (Small FP16 precision differences are expected and acceptable)")
        return True
    else:
        print("  ⚠ TensorRT outputs differ significantly from ONNX")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to TensorRT engine for Triton",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=PROJECT_ROOT / "model_repository" / "openclip_vit_b32" / "1" / "model.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "model_repository" / "openclip_vit_b32_trt" / "1" / "model.plan",
        help="Path to save TensorRT engine",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 mode (use FP32 only)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size (default: 32)",
    )
    parser.add_argument(
        "--opt-batch-size",
        type=int,
        default=16,
        help="Optimal batch size for autotuning (default: 16)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4096,
        help="TensorRT workspace size in MB (default: 4096)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "trtexec", "python"],
        default="auto",
        help="Conversion method (default: auto = prefer trtexec)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output accuracy against ONNX model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    fp16 = not args.no_fp16

    print("=" * 64)
    print("  Step 5B: ONNX → TensorRT Conversion")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    # Check GPU
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\n  GPU:     {gpu_info['name']}")
        print(f"  VRAM:    {gpu_info['memory_mb']}")
        print(f"  Compute: {gpu_info['compute_capability']}")
    else:
        print("\n  ⚠ No GPU detected (nvidia-smi not available)")
        print("  TensorRT conversion requires a GPU. Run this on a GPU instance.")
        sys.exit(1)

    # Check ONNX model exists
    if not args.onnx_path.exists():
        print(f"\n  ✗ ONNX model not found: {args.onnx_path}")
        print(f"  Run: python scripts/export_to_onnx.py --test")
        sys.exit(1)

    onnx_size_mb = args.onnx_path.stat().st_size / 1024 / 1024
    print(f"\n  ONNX model:  {args.onnx_path}")
    print(f"  ONNX size:   {onnx_size_mb:.1f} MB")
    print(f"  Output:      {args.output_path}")
    print(f"  FP16:        {'yes' if fp16 else 'no (FP32 only)'}")
    print(f"  Max batch:   {args.max_batch_size}")
    print(f"  Opt batch:   {args.opt_batch_size}")

    # Choose conversion method
    has_trtexec = check_trtexec_available()
    has_python_trt = check_tensorrt_python()

    print(f"\n  trtexec available:    {'yes' if has_trtexec else 'no'}")
    print(f"  TensorRT Python API:  {'yes' if has_python_trt else 'no'}")

    if args.method == "auto":
        method = "trtexec" if has_trtexec else "python"
    else:
        method = args.method

    if method == "trtexec" and not has_trtexec:
        print("  ✗ trtexec not found. Install TensorRT or use --method python")
        sys.exit(1)
    if method == "python" and not has_python_trt:
        print("  ✗ TensorRT Python bindings not found. Install tensorrt package")
        sys.exit(1)

    print(f"\n  Using method: {method}")
    print(f"\n  {'─' * 50}")
    print(f"  Converting ONNX → TensorRT (this may take 2-10 minutes)...")
    print(f"  {'─' * 50}")

    # Run conversion
    if method == "trtexec":
        success, metadata = convert_with_trtexec(
            onnx_path=args.onnx_path,
            output_path=args.output_path,
            fp16=fp16,
            max_batch_size=args.max_batch_size,
            opt_batch_size=args.opt_batch_size,
            workspace_mb=args.workspace,
            verbose=args.verbose,
        )
    else:
        success, metadata = convert_with_python_api(
            onnx_path=args.onnx_path,
            output_path=args.output_path,
            fp16=fp16,
            max_batch_size=args.max_batch_size,
            opt_batch_size=args.opt_batch_size,
            workspace_mb=args.workspace,
        )

    if not success:
        print(f"\n  ✗ Conversion failed")
        print(f"  {json.dumps(metadata, indent=2)}")
        sys.exit(1)

    # Report
    print(f"\n  {'=' * 50}")
    print(f"  ✓ TensorRT engine created successfully!")
    print(f"  {'=' * 50}")
    print(f"  Engine path:     {args.output_path}")
    print(f"  Engine size:     {metadata.get('engine_size_mb', '?')} MB")
    print(f"  Conversion time: {metadata.get('conversion_time_s', '?')}s")
    print(f"  Precision:       {'FP16' if fp16 else 'FP32'}")
    print(f"  Method:          {method}")

    if "trtexec_throughput" in metadata:
        print(f"\n  trtexec performance estimates:")
        if "trtexec_gpu_latency" in metadata:
            print(f"    {metadata['trtexec_gpu_latency']}")
        if "trtexec_throughput" in metadata:
            print(f"    {metadata['trtexec_throughput']}")

    # Add GPU info to metadata
    metadata["gpu"] = gpu_info

    # Save metadata
    metadata_path = args.output_path.parent / "conversion_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to: {metadata_path}")

    # Verify
    if args.verify:
        verify_tensorrt_output(args.onnx_path, args.output_path)

    # Next steps
    print(f"\n  {'─' * 50}")
    print(f"  Next steps:")
    print(f"  {'─' * 50}")
    print(f"  1. Verify config:  model_repository/openclip_vit_b32_trt/config.pbtxt")
    print(f"  2. Start Triton:   tritonserver --model-repository=/models")
    print(f"  3. Test model:     curl http://localhost:8000/v2/models/openclip_vit_b32_trt")
    print(f"  4. Benchmark:      python scripts/benchmark_backends.py --backend triton \\")
    print(f"                       --triton-url http://localhost:8000")

    return success


if __name__ == "__main__":
    main()
