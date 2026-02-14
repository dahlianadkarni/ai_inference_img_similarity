#!/bin/bash
# Run PyTorch benchmark against remote Vast.ai GPU instance
# Instance: 142.112.39.215

set -e

PYTORCH_HOST="142.112.39.215"
PYTORCH_PORT="50968"

echo "========================================"
echo "PyTorch Remote Benchmark"
echo "========================================"
echo "Instance: $PYTORCH_HOST"
echo "HTTP:     http://$PYTORCH_HOST:$PYTORCH_PORT"
echo ""

# Step 1: Health check
echo "1. Checking PyTorch service health..."
if curl -s -f "http://$PYTORCH_HOST:$PYTORCH_PORT/health" > /dev/null; then
    echo "✓ PyTorch service is ready"
else
    echo "✗ PyTorch health check failed"
    echo "  Check that your PyTorch container is running on Vast.ai"
    exit 1
fi

# Step 2: Check model info
echo ""
echo "2. Checking model info..."
if curl -s -f "http://$PYTORCH_HOST:$PYTORCH_PORT/model-info" > /dev/null; then
    echo "✓ Model endpoint accessible"
else
    echo "✗ Model info not available"
    exit 1
fi

# Step 3: Run benchmark
echo ""
echo "3. Running benchmark (this will take a few minutes)..."
echo ""

source venv/bin/activate

python scripts/benchmark_backends.py \
  --backend pytorch \
  --pytorch-url "http://$PYTORCH_HOST:$PYTORCH_PORT" \
  --iterations 50 \
  --batch-sizes 1,4,8,16,32 \
  --concurrency 16 \
  --concurrent-requests 200 \
  --output "benchmark_results/pytorch_rtx3070_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "========================================"
echo "✓ Benchmark complete!"
echo "========================================"
echo ""
echo "Results saved to: benchmark_results/"
ls -lh benchmark_results/ | tail -n 2
