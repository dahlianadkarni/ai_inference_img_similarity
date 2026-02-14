#!/bin/bash
# Run Triton benchmark against remote Vast.ai GPU instance
# Instance: 142.112.39.215

set -e

TRITON_HOST="142.112.39.215"
TRITON_HTTP_PORT="5257"
TRITON_METRICS_PORT="5354"

echo "========================================"
echo "Triton Remote Benchmark"
echo "========================================"
echo "Instance: $TRITON_HOST"
echo "HTTP:     http://$TRITON_HOST:$TRITON_HTTP_PORT"
echo "Metrics:  http://$TRITON_HOST:$TRITON_METRICS_PORT/metrics"
echo ""

# Step 1: Health check
echo "1. Checking Triton health..."
if curl -s -f "http://$TRITON_HOST:$TRITON_HTTP_PORT/v2/health/ready" > /dev/null; then
    echo "✓ Triton is ready"
else
    echo "✗ Triton health check failed"
    echo "  Check that your Triton container is running on Vast.ai"
    exit 1
fi

# Step 2: Check model
echo ""
echo "2. Checking model status..."
if curl -s -f "http://$TRITON_HOST:$TRITON_HTTP_PORT/v2/models/openclip_vit_b32/ready" > /dev/null; then
    echo "✓ Model openclip_vit_b32 is loaded"
else
    echo "✗ Model not ready"
    exit 1
fi

# Step 3: Run benchmark
echo ""
echo "3. Running benchmark (this will take a few minutes)..."
echo ""

source venv/bin/activate

python scripts/benchmark_backends.py \
  --backend triton \
  --triton-url "http://$TRITON_HOST:$TRITON_HTTP_PORT" \
  --triton-metrics-url "http://$TRITON_HOST:$TRITON_METRICS_PORT/metrics" \
  --iterations 50 \
  --batch-sizes 1,4,8,16,32 \
  --concurrency 16 \
  --concurrent-requests 200 \
  --output "benchmark_results/triton_rtx3070_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "========================================"
echo "✓ Benchmark complete!"
echo "========================================"
echo ""
echo "Results saved to: benchmark_results/"
ls -lh benchmark_results/ | tail -n 1
