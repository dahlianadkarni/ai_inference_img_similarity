# Demo Mode Setup - Separate Server

This guide explains how to run a demo server (port 8001) independently from your main Photos Library server (port 8000).

## Why Separate Servers?

- **Main Server (port 8000)**: Your Photos Library data - don't want to rescan
- **Demo Server (port 8001)**: Demo dataset for presentations - runs independently
- **Both run simultaneously**: No need to switch or reload data!
- **✅ SAFE**: Demo setup only COPIES data, never modifies your Photos scan files

## How It Works

**The Flow (Safe - Never Overwrites Photos Scan):**
1. Backup Photos scan → `*_photos_backup.json` files
2. Scan demo photos → writes to `scan_for_embeddings.json` (temporary)
3. Move to demo files → `scan_results_demo.json` (permanent demo storage)
4. Restore Photos scan → back to `scan_for_embeddings.json`
5. Both scans preserved!

**File Separation:**
- Main server (port 8000): Uses `scan_for_embeddings.json` / `embeddings/` (Photos Library)
- Demo server (port 8001): Uses `scan_results_demo.json` / `embeddings_demo/` (Demo photos)

Your Photos scan is never lost!

## Quick Start

### Step 1: Create Demo Dataset

Choose one option:

**Option A: Use setup_demo.sh** (Quick - 20 images)
```bash
./setup_demo.sh
# Creates ~/demo_photos with sample images and duplicates
```

**Option B: Use your own folder**
```bash
# Just use any folder with images
mkdir -p ~/demo_photos
# Copy some photos there
```

### Step 2: Backup Your Photos Scan (If You Have One)

If you've already scanned your Photos Library and want to preserve it:

```bash
# Backup your current Photos scan (if it exists)
if [ -f "scan_for_embeddings.json" ]; then
    cp scan_for_embeddings.json scan_for_embeddings_photos_backup.json
    cp scan_duplicates.json scan_duplicates_photos_backup.json 2>/dev/null || true
    cp -r embeddings embeddings_photos_backup 2>/dev/null || true
    echo "✓ Backed up Photos scan to *_photos_backup files"
fi
```

### Step 3: Scan Demo Directory

```bash
# Start main server if not running
source venv/bin/activate
python -m src.ui.main
```

Then in browser (http://127.0.0.1:8000):
1. Select **Source: Directory**
2. Path: `/Users/YOUR_USERNAME/demo_photos`
3. Click **Start Scan**
4. Click **Generate Embeddings**
5. Wait for completion

### Step 4: Move Demo Scan to Demo Files & Restore Photos Scan

```bash
# Move demo scan to demo files
mv scan_for_embeddings.json scan_results_demo.json
mv scan_duplicates.json scan_duplicates_demo.json 2>/dev/null || true
mv embeddings embeddings_demo

# Restore your Photos scan (if you backed it up)
if [ -f "scan_for_embeddings_photos_backup.json" ]; then
    mv scan_for_embeddings_photos_backup.json scan_for_embeddings.json
    mv scan_duplicates_photos_backup.json scan_duplicates.json 2>/dev/null || true
    mv embeddings_photos_backup embeddings
    echo "✓ Restored Photos scan"
fi
```

Now your Photos scan is back in place, and demo scan is in demo files!

### Step 5: Start Demo Server

In a **new terminal**:

```bash
source venv/bin/activate
python start_demo_server.py
```

## Using Both Servers

Now you have:

### Main Server (Photos Library)
- **URL**: http://127.0.0.1:8000
- **Data**: Your iCloud Photos Library
- **Scan files**: `scan_for_embeddings.json`, `embeddings/`
- **Duplicates**: Added to Photos.app "Duplicates" album

### Demo Server (Demo Dataset)  
- **URL**: http://127.0.0.1:8001
- **Data**: Demo photos from `~/demo_photos`
- **Scan files**: `scan_results_demo.json`, `embeddings_demo/`
- **Duplicates**: Copied to `~/demo_photos/duplicates/` folder

## Complete Workflow Example

```bash
# Terminal 1: Main server (Photos Library)
source venv/bin/activate
python -m src.ui.main
# → http://127.0.0.1:8000

# Use this for your real Photos management

# Terminal 2: Create and scan demo data
./setup_demo.sh                    # Create demo dataset
# Then in browser at :8000, scan ~/demo_photos directory
# Generate embeddings, wait for completion

# Back in terminal:
./setup_demo_mode.sh               # Copy to demo files

# Terminal 2: Start demo server
python start_demo_server.py
# → http://127.0.0.1:8001

# Use this for presentations/demos
```

## Demo Workflow

When presenting/demoing on port 8001:

1. **Show the dataset**: `~/demo_photos` folder
2. **Show scan results**: Visit http://127.0.0.1:8001
3. **Navigate tabs**: AI Results, Perceptual, Exact duplicates
4. **Select duplicates**: Check boxes on similar images
5. **Add to duplicates**: Click "📁 Add Selected to Duplicates Album"
6. **Show duplicates folder**: Opens `~/demo_photos/duplicates/` in Finder
7. **Show space savings**: File sizes before/after

## Switching Between Modes

Just open different URLs in different browser tabs:

- **Managing real Photos**: http://127.0.0.1:8000
- **Running demo**: http://127.0.0.1:8001

No rescanning, no data loss, completely independent!

## Updating Demo Data

If you want to change the demo dataset:

```bash
# 1. Create new demo photos
rm -rf ~/demo_photos/*
./setup_demo.sh  # or copy new photos manually

# 2. Rescan on main server (:8000)
# Select Directory → ~/demo_photos
# Start scan → Generate embeddings

# 3. Copy to demo files
./setup_demo_mode.sh

# 4. Restart demo server (Ctrl+C in Terminal 2, then):
python start_demo_server.py
```

## Tips for Presentations

1. **Pre-setup everything**: Have both servers running before demo
2. **Bookmark both URLs**: Easy switching during presentation
3. **Show folder first**: `open ~/demo_photos` to show source files
4. **Highlight auto-detection**: Show AI grouping similar images
5. **Demo both modes**: Switch to :8000 to show Photos integration
6. **Show logs**: Terminal output shows what's happening

## Troubleshooting

### "Demo scan file not found"
Run `./setup_demo_mode.sh` after scanning your demo directory on the main server.

### "Address already in use" on port 8001  
Demo server is already running. Stop it (Ctrl+C) first.

### Demo data out of sync
Re-run: `./setup_demo_mode.sh` to copy latest scan results.

### Want to reset demo
```bash
rm -f scan_results_demo.json scan_duplicates_demo.json
rm -rf embeddings_demo ~/demo_photos/duplicates
```


-------
# One-time setup
./setup_demo.sh                    # Create demo photos
# (Scan ~/demo_photos on :8000)
./setup_demo_mode.sh               # Copy to demo files

# Daily use - run both servers
# Terminal 1:
python -m src.ui.main              # Port 8000 - Photos Library

# Terminal 2:
python start_demo_server.py        # Port 8001 - Demo data

File Structure:
Main (Port 8000): scan_for_embeddings.json, embeddings
Demo (Port 8001): scan_results_demo.json, embeddings_demo/