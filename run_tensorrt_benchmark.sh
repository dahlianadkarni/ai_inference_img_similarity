#!/bin/bash
# =============================================================================
# Run TensorRT benchmark against remote Vast.ai GPU instance (Step 5B)
#
# This benchmarks both ONNX and TensorRT models on the same Triton instance,
# plus optionally the PyTorch backend if available.
# =============================================================================

set -e

# Configuration — UPDATE THESE for your deployment
TRITON_HOST="${TRITON_HOST:-142.112.39.215}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-5399}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-5037}"
PYTORCH_HOST="${PYTORCH_HOST:-}"
PYTORCH_PORT="${PYTORCH_PORT:-50968}"

echo "========================================"
echo "Step 5B: TensorRT Benchmark"
echo "========================================"
echo "Triton:  http://$TRITON_HOST:$TRITON_HTTP_PORT"
echo "Metrics: http://$TRITON_HOST:$TRITON_METRICS_PORT/metrics"
if [ -n "$PYTORCH_HOST" ]; then
    echo "PyTorch: http://$PYTORCH_HOST:$PYTORCH_PORT"
fi
echo ""

# Step 1: Health check
echo "1. Checking Triton health..."
if curl -s -f "http://$TRITON_HOST:$TRITON_HTTP_PORT/v2/health/ready" > /dev/null; then
    echo "✓ Triton is ready"
else
    echo "✗ Triton health check failed"
    echo "  Ensure container is running and TRT engine build is complete"
    exit 1
fi

# Step 2: Check models
echo ""
echo "2. Checking model status..."

ONNX_READY=false
TRT_READY=false

if curl -s -f "http://$TRITON_HOST:$TRITON_HTTP_PORT/v2/models/openclip_vit_b32/ready" > /dev/null; then
    echo "✓ ONNX model (openclip_vit_b32) is loaded"
    ONNX_READY=true
else
    echo "✗ ONNX model not loaded"
fi

if curl -s -f "http://$TRITON_HOST:$TRITON_HTTP_PORT/v2/models/openclip_vit_b32_trt/ready" > /dev/null; then
    echo "✓ TensorRT model (openclip_vit_b32_trt) is loaded"
    TRT_READY=true
else
    echo "⚠ TensorRT model not loaded (engine may still be building)"
    echo "  Wait a few minutes and try again, or check container logs"
fi

if [ "$ONNX_READY" = false ] && [ "$TRT_READY" = false ]; then
    echo ""
    echo "✗ No models available. Check container status."
    exit 1
fi

# Step 3: Run benchmark
echo ""
echo "3. Running 3-way benchmark (this will take several minutes)..."
echo ""

source venv/bin/activate

# Build arguments
ARGS="--triton-url http://$TRITON_HOST:$TRITON_HTTP_PORT"
ARGS="$ARGS --triton-metrics-url http://$TRITON_HOST:$TRITON_METRICS_PORT/metrics"
ARGS="$ARGS --iterations 30"
ARGS="$ARGS --concurrency 16"
ARGS="$ARGS --concurrent-requests 200"
ARGS="$ARGS --batch-sizes 1,4,8,16,32"

if [ -n "$PYTORCH_HOST" ]; then
    ARGS="$ARGS --pytorch-url http://$PYTORCH_HOST:$PYTORCH_PORT"
else
    ARGS="$ARGS --no-pytorch"
fi

python scripts/benchmark_tensorrt.py $ARGS

echo ""
echo "========================================"
echo "✓ Benchmark complete!"
echo "========================================"
echo ""
echo "Results saved to: benchmark_results/"
ls -lhtr benchmark_results/ | tail -n 3
