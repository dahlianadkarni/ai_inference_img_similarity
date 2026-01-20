## ✅ Solution: Separate Demo Server

**Problem**: Need to demo the app without rescanning Photos Library  
**Solution**: Run two independent servers on different ports

### Main Server (Port 8000) - Photos Library
- URL: http://127.0.0.1:8000
- Data: Your iCloud Photos
- Files: `scan_for_embeddings.json`, `embeddings/`
- Start: `python -m src.ui.main`

### Demo Server (Port 8001) - Demo Dataset  
- URL: http://127.0.0.1:8001
- Data: `~/demo_photos` directory
- Files: `scan_results_demo.json`, `embeddings_demo/`
- Start: `python start_demo_server.py`

## Setup (One Time)

```bash
# 1. Create demo dataset
./setup_demo.sh

# 2. Scan demo folder (on main server :8000)
#    Browser: Directory → ~/demo_photos → Scan → Generate Embeddings

# 3. Copy scan data to demo files
./setup_demo_mode.sh

# 4. Start demo server
python start_demo_server.py
```

## Daily Use

```bash
# Terminal 1: Photos Library (your real data)
python -m src.ui.main
# → http://127.0.0.1:8000

# Terminal 2: Demo (for presentations)
python start_demo_server.py  
# → http://127.0.0.1:8001
```

Both run simultaneously with completely independent data!

## See Also
- [DEMO_SETUP.md](DEMO_SETUP.md) - Detailed setup guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet
