#!/bin/bash
# Setup demo mode by copying current scan results to demo files
# 
# IMPORTANT: This script is SAFE - it only COPIES your Photos scan data
# to separate demo files. Your original scan_for_embeddings.json and
# embeddings/ folder are NEVER modified or overwritten.

echo "=================================="
echo "Setting up Demo Mode"
echo "=================================="
echo ""
echo "⚠️  SAFETY NOTE:"
echo "   This script only COPIES (cp) your scan data"
echo "   Your Photos scan files are NOT modified"
echo ""

# Check if scan files exist
if [ ! -f "scan_for_embeddings.json" ]; then
    echo "❌ Error: scan_for_embeddings.json not found"
    echo ""
    echo "Please run a scan first:"
    echo "1. Start main server: python -m src.ui.main"
    echo "2. Go to http://127.0.0.1:8000"
    echo "3. Scan your ~/demo_photos directory"
    echo "4. Generate embeddings"
    echo "5. Then run this script again"
    exit 1
fi

echo "✓ Found scan files"
echo ""

# Copy scan results to demo files
echo "Copying scan results to demo files..."
echo ""
echo "Source (READ ONLY):"
ls -lh scan_for_embeddings.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Destination (COPY):"
cp scan_for_embeddings.json scan_results_demo.json
ls -lh scan_results_demo.json | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "✓ Copied scan_for_embeddings.json → scan_results_demo.json"
echo "  (Original file unchanged)"

if [ -f "scan_duplicates.json" ]; then
    cp scan_duplicates.json scan_duplicates_demo.json
    echo "✓ Copied scan_duplicates.json → scan_duplicates_demo.json"
fi

# Copy embeddings folder
if [ -d "embeddings" ]; then
    echo ""
    echo "Copying embeddings folder..."
    rm -rf embeddings_demo
    cp -r embeddings embeddings_demo
    echo "✓ Copied embeddings/ → embeddings_demo/"
fi

echo ""
echo "=================================="
echo "✓ Demo mode setup complete!"
echo "=================================="
echo ""
echo "Now you can run BOTH servers simultaneously:"
echo ""
echo "Terminal 1 (Photos Library - port 8000):"
echo "  source venv/bin/activate"
echo "  python -m src.ui.main"
echo "  → http://127.0.0.1:8000"
echo ""
echo "Terminal 2 (Demo - port 8001 + inference service):"
echo "  source venv/bin/activate"
echo "  python start_services.py --ui-demo"
echo "  → Demo UI: http://127.0.0.1:8001"
echo "  → Inference service: http://127.0.0.1:8002"
echo ""
echo "Both will run (UI on 8001) with the inference service on 8002; they use separate data!"
echo ""
