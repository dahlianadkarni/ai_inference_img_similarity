#!/usr/bin/env python3
"""
Start demo server on port 8001 with demo dataset.

This runs independently from the main Photos Library server (port 8000).
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import uvicorn
from src.ui.app_v4 import app, load_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Demo file paths
PROJECT_ROOT = Path(__file__).parent
DEMO_SCAN = PROJECT_ROOT / "scan_results_demo.json"
DEMO_GROUPS = PROJECT_ROOT / "embeddings_demo" / "similar_groups.json"
DEMO_EMBEDDINGS = PROJECT_ROOT / "embeddings_demo"
DEMO_DUPLICATES = PROJECT_ROOT / "scan_duplicates_demo.json"

def main():
    """Start demo server."""
    # Check if demo files exist
    if not DEMO_SCAN.exists():
        logger.warning(f"Demo scan file not found: {DEMO_SCAN}")
        logger.info("Run a scan with 'Directory' source first, then:")
        logger.info(f"  1. Copy scan_for_embeddings.json to {DEMO_SCAN}")
        logger.info(f"  2. Copy scan_duplicates.json to {DEMO_DUPLICATES}")
        logger.info(f"  3. Copy embeddings/ folder to {DEMO_EMBEDDINGS}")
        logger.info("")
        logger.info("Or run: python create_demo_dataset.py && python setup_demo.sh")
        sys.exit(1)
    
    # Load demo data
    logger.info("="  * 60)
    logger.info("DEMO SERVER - Starting with demo dataset")
    logger.info("="  * 60)
    
    load_data(
        scan_results=DEMO_SCAN,
        similar_groups=DEMO_GROUPS,
        embeddings_dir=DEMO_EMBEDDINGS,
        path_mapping_file=None,
        is_demo=True
    )
    
    # Start server on different port
    port = 8001
    logger.info(f"")
    logger.info(f"✓ Demo server starting at http://127.0.0.1:{port}")
    logger.info(f"✓ Main server (Photos) runs at http://127.0.0.1:8000")
    logger.info(f"")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )

if __name__ == "__main__":
    main()
