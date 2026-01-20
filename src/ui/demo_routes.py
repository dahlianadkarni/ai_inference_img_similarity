"""
Demo mode routes - separate endpoints for directory-based scanning.

This allows the main app (/) to maintain Photos Library state
while /demo maintains separate demo dataset state.
"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
import logging

logger = logging.getLogger(__name__)

# Create router for demo endpoints
demo_router = APIRouter(prefix="/api/demo")


# Demo-specific file names
DEMO_FILES = {
    "scan_results": "scan_results_demo.json",
    "scan_duplicates": "scan_duplicates_demo.json",
    "similar_groups": "embeddings_demo/similar_groups.json",
    "embeddings_dir": "embeddings_demo",
}


@demo_router.post("/add-to-duplicates")
async def demo_add_to_duplicates(request: dict):
    """Demo mode: Copy file to duplicates folder."""
    from .app_v3 import copy_to_duplicates_folder
    
    file_path = request.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path required")
    
    success = await copy_to_duplicates_folder(file_path)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add photo to duplicates")
    return {"success": True}


@demo_router.post("/open-duplicates-folder")
async def demo_open_duplicates():
    """Demo mode: Open the duplicates folder in Finder."""
    from .app_v3 import DEMO_SCAN_DATA, PROJECT_ROOT
    
    if DEMO_SCAN_DATA:
        first_path = Path(DEMO_SCAN_DATA[0]['file_path'])
        duplicates_dir = first_path.parent / "duplicates"
        
        if duplicates_dir.exists():
            try:
                subprocess.run(['open', str(duplicates_dir)], check=True)
                return {"success": True}
            except Exception as e:
                logger.error(f"Failed to open duplicates folder: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=404, detail="Duplicates folder not found. Add some duplicates first!")
    else:
        raise HTTPException(status_code=400, detail="No demo scan data available")


@demo_router.get("/status")
async def demo_status():
    """Get demo processing status."""
    from .app_v3 import DEMO_PROCESSING_STATUS
    return DEMO_PROCESSING_STATUS


@demo_router.get("/scan-info")
async def demo_scan_info():
    """Get demo scan information."""
    from .app_v3 import DEMO_SCAN_DATA, PROJECT_ROOT
    
    scan_file = PROJECT_ROOT / DEMO_FILES["scan_results"]
    duplicates_file = PROJECT_ROOT / DEMO_FILES["scan_duplicates"]
    
    return {
        "scan_exists": scan_file.exists(),
        "duplicates_exist": duplicates_file.exists(),
        "scan_count": len(DEMO_SCAN_DATA) if DEMO_SCAN_DATA else 0,
    }


@demo_router.get("/ai-groups-categorized")
async def demo_ai_groups():
    """Get categorized AI groups for demo mode."""
    from .app_v3 import DEMO_SIMILAR_GROUPS
    
    # Same categorization logic as main app
    # ... implement similar to main endpoint
    return {
        "new_discoveries": [],
        "exact_overlap": [],
        "perceptual_overlap": [],
    }
