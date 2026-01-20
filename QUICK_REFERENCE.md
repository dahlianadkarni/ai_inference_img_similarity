# Quick Reference: Main vs Demo Server

## Two Independent Servers

| Feature | Main Server | Demo Server |
|---------|------------|-------------|
| **Port** | 8000 | 8001 |
| **URL** | http://127.0.0.1:8000 | http://127.0.0.1:8001 |
| **Data Source** | Photos Library (iCloud) | Directory (~/demo_photos) |
| **Scan Files** | scan_for_embeddings.json | scan_results_demo.json |
| **Embeddings** | embeddings/ | embeddings_demo/ |
| **Duplicates** | Photos "Duplicates" album | ~/demo_photos/duplicates/ |
| **Use Case** | Real photo management | Demos & presentations |

## Starting Servers

### Main Server (Photos Library)
```bash
source venv/bin/activate
python -m src.ui.main
```

### Demo Server (after setup)
```bash
source venv/bin/activate
python start_demo_server.py
```

## Setup Demo (One Time)

```bash
# 1. Create demo photos
./setup_demo.sh

# 2. Scan on main server (:8000)
# Browser: Directory → ~/demo_photos → Scan → Generate Embeddings

# 3. Copy to demo files
./setup_demo_mode.sh

# 4. Start demo server
python start_demo_server.py
```

## Benefits

✅ **No rescanning**: Keep Photos Library results intact  
✅ **Run simultaneously**: Both servers at once  
✅ **Independent state**: Changes don't affect each other  
✅ **Easy demos**: No setup during presentation  
✅ **Safe testing**: Demo doesn't touch real Photos data  

## File Structure

```
inference_1/
├── scan_for_embeddings.json    # Main (Photos)
├── scan_duplicates.json         # Main (Photos)
├── embeddings/                  # Main (Photos)
│
├── scan_results_demo.json       # Demo
├── scan_duplicates_demo.json    # Demo
├── embeddings_demo/             # Demo
│
└── ~/demo_photos/               # Demo source files
    └── duplicates/              # Demo duplicates folder
```

## Common Commands

```bash
# Start main server
python -m src.ui.main

# Start demo server
python start_demo_server.py

# Create demo dataset
./setup_demo.sh

# Copy scan to demo files
./setup_demo_mode.sh

# Check demo folder
open ~/demo_photos

# Reset demo
rm -rf embeddings_demo scan_*_demo.json ~/demo_photos/duplicates
```
