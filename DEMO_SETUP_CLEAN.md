# Demo Setup - Complete Separation

**NO MORE CONFUSION!** Demo and Photos Library are 100% separate.

## File Separation

| Component | Photos Library (Port 8000) | Demo (Port 8001) |
|-----------|---------------------------|------------------|
| **Scan Results** | `scan_for_embeddings.json` | `scan_results_demo.json` |
| **Duplicates** | `scan_duplicates.json` | `scan_duplicates_demo.json` |
| **Embeddings** | `embeddings/` | `embeddings_demo/` |
| **Cache** | `.cache/` | `.cache_demo/` |
| **Source** | Photos Library | `~/demo_photos` |
| **Duplicates Folder** | Photos "Duplicates" album | `~/demo_photos/duplicates/` |

## Setup (One Time)

### 1. Create Demo Photos

```bash
./setup_demo.sh
```

This creates `~/demo_photos` with sample images.

### 2. Scan Demo Photos

```bash
./scan_demo.sh
```

This command:
- Scans `~/demo_photos`
- Writes to `scan_results_demo.json` (NOT scan_for_embeddings.json)
- Uses `.cache_demo/` (NOT .cache/)
- Generates `embeddings_demo/` (NOT embeddings/)
- **Never touches your Photos data**

### 3. Start Demo Server

```bash
python start_demo_server.py
```

Opens at http://127.0.0.1:8001

## Daily Use

### Photos Library Server (Port 8000)
```bash
source venv/bin/activate
python -m src.ui.main
```
→ http://127.0.0.1:8080

### Demo Server (Port 8081)
```bash
source venv/bin/activate
python start_demo_server.py
```
→ http://127.0.0.1:8081

**Both can run simultaneously!**

## Complete Workflow

```bash
# ONE TIME SETUP
./setup_demo.sh          # Create demo photos
./scan_demo.sh           # Scan demo photos (writes to demo files)

# DAILY USE
# Terminal 1: Photos Library
python -m src.ui.main    # Port 8000

# Terminal 2: Demo
python start_demo_server.py  # Port 8001
```

## Benefits

✅ **No file moving** - Demo scan writes directly to demo files  
✅ **No cache conflicts** - Separate `.cache_demo/` folder  
✅ **No overlap** - Photos and demo never touch same files  
✅ **No confusion** - Clear file naming  
✅ **Safe** - Impossible to overwrite Photos data  

## Updating Demo

```bash
# Change demo photos
rm -rf ~/demo_photos/*
cp /path/to/new/photos ~/demo_photos/

# Rescan
./scan_demo.sh

# Restart demo server
# Ctrl+C in terminal running demo server, then:
python start_demo_server.py
```

## File Tree

```
inference_1/
├── .cache/                          # Photos cache
│   ├── scan_cache.json
│   └── photos_metadata.json
│
├── .cache_demo/                     # Demo cache (SEPARATE!)
│   └── scan_cache.json
│
├── scan_for_embeddings.json         # Photos scan
├── scan_duplicates.json             # Photos duplicates
├── embeddings/                      # Photos embeddings
│
├── scan_results_demo.json           # Demo scan
├── scan_duplicates_demo.json        # Demo duplicates
├── embeddings_demo/                 # Demo embeddings
│
├── scan_demo.sh                     # Scan demo directly
├── setup_demo.sh                    # Create demo dataset
└── start_demo_server.py             # Start demo server
```

## Troubleshooting

**Q: Demo server shows Photos Library data**  
A: Make sure you ran `./scan_demo.sh` first. Check that `scan_results_demo.json` exists.

**Q: Want to reset demo**  
```bash
rm -rf scan_results_demo.json scan_duplicates_demo.json embeddings_demo/ .cache_demo/
./scan_demo.sh
```

**Q: Demo server won't start**  
A: Check that port 8001 is free: `lsof -ti:8001`
