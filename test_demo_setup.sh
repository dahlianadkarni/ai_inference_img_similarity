#!/bin/bash
# Test that demo mode setup works

echo "Testing Demo Mode Setup..."
echo ""

# Check if main scan exists
if [ ! -f "scan_for_embeddings.json" ]; then
    echo "⚠️  No main scan found (scan_for_embeddings.json)"
    echo "   This is OK if you haven't scanned your Photos yet."
    echo ""
else
    echo "✅ Main scan found: scan_for_embeddings.json"
    
    # Try to copy to demo
    cp scan_for_embeddings.json scan_results_demo.json 2>/dev/null
    if [ -f "scan_results_demo.json" ]; then
        echo "✅ Successfully created demo scan file"
        
        # Check size
        SIZE=$(wc -c < scan_results_demo.json)
        echo "   Size: $SIZE bytes"
        
        # Clean up test
        rm scan_results_demo.json
    fi
fi

echo ""
echo "Testing scripts..."
[ -x "setup_demo.sh" ] && echo "✅ setup_demo.sh is executable" || echo "❌ setup_demo.sh not executable"
[ -x "setup_demo_mode.sh" ] && echo "✅ setup_demo_mode.sh is executable" || echo "❌ setup_demo_mode.sh not executable"
[ -x "start_demo_server.py" ] && echo "✅ start_demo_server.py is executable" || echo "❌ start_demo_server.py not executable"

echo ""
echo "Testing Python imports..."
python3 -c "from src.ui.app_v3 import load_data; print('✅ load_data import works')" 2>/dev/null || echo "❌ Import failed"

echo ""
echo "Demo mode setup test complete!"
echo ""
echo "Next steps:"
echo "1. ./setup_demo.sh               # Create demo dataset"
echo "2. python -m src.ui.main         # Scan demo folder on :8080"
echo "3. ./setup_demo_mode.sh          # Copy to demo files"  
echo "4. python start_demo_server.py  # Start demo server on :8081"
