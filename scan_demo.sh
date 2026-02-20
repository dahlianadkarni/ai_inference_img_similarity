#!/bin/bash
# Scan demo photos directly to demo files (no overlap with Photos data)

DEMO_DIR="$HOME/demo_photos"

echo "=================================="
echo "Scanning Demo Photos"
echo "=================================="
echo ""
echo "Demo directory: $DEMO_DIR"
echo "Output: scan_results_demo.json"
echo "Embeddings: embeddings_demo/"
echo "Cache: .cache_demo/"
echo ""

# Check if demo directory exists
if [ ! -d "$DEMO_DIR" ]; then
    echo "❌ Demo directory not found: $DEMO_DIR"
    echo ""
    echo "Run ./setup_demo.sh first to create demo photos"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Scan directly to demo files (separate cache!)
echo "Step 1: Scanning photos..."
python -m src.scanner.main \
    "$DEMO_DIR" \
    --output scan_results_demo.json \
    --duplicates-output scan_duplicates_demo.json \
    --cache-file .cache_demo/scan_cache.json \
    --md5-mode on-demand

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Scan failed"
    exit 1
fi

echo ""
echo "Step 2: Generating embeddings..."
python -m src.embedding.main \
    scan_results_demo.json \
    --output embeddings_demo \
    --similarity-threshold 0.85

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Embedding generation failed"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ Demo Scan Complete!"
echo "=================================="
echo ""
echo "Files created:"
echo "  📄 scan_results_demo.json"
echo "  📄 scan_duplicates_demo.json"
echo "  📁 embeddings_demo/"
echo "  📁 .cache_demo/"
echo ""
echo "Start demo server (recommended):"
echo "  python start_services.py --ui-demo"
echo "  → This starts the demo UI (http://127.0.0.1:8081) and the inference service (http://127.0.0.1:8002)."
echo ""
