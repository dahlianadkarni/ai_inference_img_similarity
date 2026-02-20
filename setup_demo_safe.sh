#!/bin/bash
# Safe demo setup - preserves your Photos Library scan

echo "=================================="
echo "Safe Demo Setup"
echo "=================================="
echo ""
echo "This script will:"
echo "  1. Backup your Photos scan (if exists)"
echo "  2. Prompt you to scan demo photos"
echo "  3. Move demo scan to demo files"
echo "  4. Restore your Photos scan"
echo ""

# Check if Photos scan exists
if [ -f "scan_for_embeddings.json" ]; then
    echo "📸 Found existing Photos scan - will backup and restore"
    HAS_PHOTOS_SCAN=true
else
    echo "ℹ️  No existing scan found - demo scan will go directly to demo files"
    HAS_PHOTOS_SCAN=false
fi

echo ""
read -p "Press Enter to continue..."
echo ""

# Step 1: Backup if needed
if [ "$HAS_PHOTOS_SCAN" = true ]; then
    echo "Step 1: Backing up Photos scan..."
    cp scan_for_embeddings.json scan_for_embeddings_photos_backup.json
    echo "  ✓ scan_for_embeddings.json → scan_for_embeddings_photos_backup.json"
    
    if [ -f "scan_duplicates.json" ]; then
        cp scan_duplicates.json scan_duplicates_photos_backup.json
        echo "  ✓ scan_duplicates.json → scan_duplicates_photos_backup.json"
    fi
    
    if [ -d "embeddings" ]; then
        cp -r embeddings embeddings_photos_backup
        echo "  ✓ embeddings/ → embeddings_photos_backup/"
    fi
    echo ""
fi

# Step 2: Wait for user to scan demo photos
echo "Step 2: Scan your demo photos now"
echo ""
echo "  1. Make sure server is running: python -m src.ui.main"
echo "  2. Go to: http://127.0.0.1:8080"
echo "  3. Select: Source → Directory"
echo "  4. Path: $HOME/demo_photos"
echo "  5. Click: Start Scan"
echo "  6. Click: Generate Embeddings"
echo "  7. Wait for completion"
echo ""
read -p "When scan is complete, press Enter to continue..."
echo ""

# Step 3: Check if demo scan exists
if [ ! -f "scan_for_embeddings.json" ]; then
    echo "❌ Error: scan_for_embeddings.json not found"
    echo "   Did you complete the scan?"
    exit 1
fi

# Step 4: Move demo scan to demo files
echo "Step 3: Moving demo scan to demo files..."
mv scan_for_embeddings.json scan_results_demo.json
echo "  ✓ Moved to scan_results_demo.json"

if [ -f "scan_duplicates.json" ]; then
    mv scan_duplicates.json scan_duplicates_demo.json
    echo "  ✓ Moved to scan_duplicates_demo.json"
fi

if [ -d "embeddings" ]; then
    mv embeddings embeddings_demo
    echo "  ✓ Moved to embeddings_demo/"
fi
echo ""

# Step 5: Restore Photos scan
if [ "$HAS_PHOTOS_SCAN" = true ]; then
    echo "Step 4: Restoring your Photos scan..."
    mv scan_for_embeddings_photos_backup.json scan_for_embeddings.json
    echo "  ✓ Restored scan_for_embeddings.json"
    
    if [ -f "scan_duplicates_photos_backup.json" ]; then
        mv scan_duplicates_photos_backup.json scan_duplicates.json
        echo "  ✓ Restored scan_duplicates.json"
    fi
    
    if [ -d "embeddings_photos_backup" ]; then
        mv embeddings_photos_backup embeddings
        echo "  ✓ Restored embeddings/"
    fi
    echo ""
fi

echo "=================================="
echo "✅ Demo Setup Complete!"
echo "=================================="
echo ""
echo "Your files:"
if [ "$HAS_PHOTOS_SCAN" = true ]; then
    echo "  📸 Photos scan: scan_for_embeddings.json, embeddings/ (PRESERVED!)"
fi
echo "  🎬 Demo scan: scan_results_demo.json, embeddings_demo/"
echo ""
echo "Start demo server (recommended):"
echo "  python start_services.py --ui-demo"
echo "  → This starts the demo UI (http://127.0.0.1:8081) and the inference service (http://127.0.0.1:8002)."
