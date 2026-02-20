"""FastAPI application for duplicate photo review with tab-based workflow (v6).

Refactored from v5:
- HTML template extracted to templates/review.html
- get_paths() helper eliminates repeated SERVER_MODE branching for file paths
- get_processing_status() returns the mode-appropriate status dict
"""

import logging
from pathlib import Path
from typing import List, Optional
from types import SimpleNamespace
import json
import shutil
import subprocess
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

"""Command-line interface for review UI server."""

import argparse
import logging
from pathlib import Path
import sys
import json
import os

import numpy as np
from tqdm import tqdm
from PIL import Image

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from ..grouping import FeedbackLearner
from ..embedding.storage import EmbeddingStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Photo Duplicate Reviewer",
    description="Review and manage duplicate photo groups with AI feedback",
    version="0.2.0",
)

# Server mode - which dataset is active
SERVER_MODE: str = "main"  # "main" or "demo"

# Inference service URL (configurable via environment variable)
INFERENCE_SERVICE_URL: str = os.getenv("INFERENCE_SERVICE_URL", "http://127.0.0.1:8002")

# Global state - Main (Photos Library)
SIMILAR_GROUPS: List[dict] = []
SCAN_DATA: dict = {}
FEEDBACK_LEARNER: Optional[FeedbackLearner] = None
EMBEDDING_STORE: Optional[EmbeddingStore] = None
PROCESSING_STATUS: dict = {"running": False, "stage": "", "message": ""}
CURRENT_THRESHOLD: float = 0.85
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
PATH_MAPPING: dict = {}  # Maps local cached paths to original Photos Library paths
PHOTOS_METADATA: dict = {}  # Maps UUID to original filename from Photos app
DEMO_MODE: bool = False  # True if using directory scanning (not Photos Library)

# Global state - Demo (Directory scanning)
DEMO_SIMILAR_GROUPS: List[dict] = []
DEMO_SCAN_DATA: dict = {}
DEMO_FEEDBACK_LEARNER: Optional[FeedbackLearner] = None
DEMO_EMBEDDING_STORE: Optional[EmbeddingStore] = None
DEMO_PROCESSING_STATUS: dict = {"running": False, "stage": "", "message": ""}
DEMO_THRESHOLD: float = 0.85

# Template directory
TEMPLATES_DIR: Path = Path(__file__).parent / "templates"


def get_paths(mode: Optional[str] = None) -> SimpleNamespace:
    """Return all mode-dependent file paths for the given mode (or SERVER_MODE).

    Pass mode="demo" or mode="main" explicitly (e.g. from main.py before
    load_data sets SERVER_MODE), or omit to use the current SERVER_MODE.
    """
    if (mode or SERVER_MODE) == "demo":
        return SimpleNamespace(
            # Absolute paths (for reading/writing in Python)
            scan_results_display = PROJECT_ROOT / "scan_results_demo.json",
            scan_results         = PROJECT_ROOT / "scan_results_demo.json",
            scan_duplicates      = PROJECT_ROOT / "scan_duplicates_demo.json",
            scan_cache           = PROJECT_ROOT / ".cache_demo" / "scan_cache.json",
            embeddings_dir       = PROJECT_ROOT / "embeddings_demo",
            groups_file          = PROJECT_ROOT / "embeddings_demo" / "similar_groups.json",
            # Relative paths (for subprocess commands run from PROJECT_ROOT)
            scan_output_rel      = "scan_results_demo.json",
            dup_output_rel       = "scan_duplicates_demo.json",
            cache_file_rel       = ".cache_demo/scan_cache.json",
            output_dir_rel       = "embeddings_demo",
            scan_file_rel        = "scan_results_demo.json",
        )
    return SimpleNamespace(
        # Absolute paths
        scan_results_display = PROJECT_ROOT / "scan_results.json",
        scan_results         = PROJECT_ROOT / "scan_for_embeddings.json",
        scan_duplicates      = PROJECT_ROOT / "scan_duplicates.json",
        scan_cache           = PROJECT_ROOT / ".cache" / "scan_cache.json",
        embeddings_dir       = PROJECT_ROOT / "embeddings",
        groups_file          = PROJECT_ROOT / "embeddings" / "similar_groups.json",
        # Relative paths
        scan_output_rel      = "scan_for_embeddings.json",
        dup_output_rel       = "scan_duplicates.json",
        cache_file_rel       = ".cache/scan_cache.json",
        output_dir_rel       = "embeddings",
        scan_file_rel        = "scan_for_embeddings.json",
    )


def get_processing_status() -> dict:
    """Return the processing status dict for the current SERVER_MODE."""
    return DEMO_PROCESSING_STATUS if SERVER_MODE == "demo" else PROCESSING_STATUS


class ReviewAction(BaseModel):
    """Model for review action."""
    group_id: int
    action: str  # "keep_all", "not_similar", or list of indices to delete
    delete_indices: Optional[List[int]] = None


class ScanRequest(BaseModel):
    """Model for scan request."""
    source: str  # "photos-library" or "directory"
    photos_access: str = "applescript"  # "applescript" or "originals"
    path: Optional[str] = None
    limit: Optional[int] = 20
    use_cache: bool = True
    md5_mode: str = "on-demand"  # "on-demand", "always", "never"


class EmbeddingRequest(BaseModel):
    """Model for embedding generation request."""
    similarity_threshold: float = 0.85
    estimate: bool = False  # If True, only estimate time/resources
    estimate_sample: int = 30  # Number of images to sample for estimation
    inference_mode: str = "remote"  # "local", "remote", or "auto"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main review interface."""
    template_path = TEMPLATES_DIR / "review.html"
    if not template_path.exists():
        raise HTTPException(status_code=500, detail="Template not found: templates/review.html")
    return HTMLResponse(content=template_path.read_text())


@app.get("/api/groups")
async def get_groups():
    """Get all similarity groups."""
    if SERVER_MODE == "demo":
        return {
            "groups": DEMO_SIMILAR_GROUPS,
            "threshold": DEMO_THRESHOLD
        }
    return {
        "groups": SIMILAR_GROUPS,
        "threshold": CURRENT_THRESHOLD
    }


@app.get("/api/status")
async def get_status():
    """Get processing status."""
    return get_processing_status()


@app.get("/api/scan-info")
async def get_scan_info():
    """Get information about the completed scan."""
    p = get_paths()
    scan_results_path = p.scan_results_display
    scan_duplicates_path = p.scan_duplicates
    cache_path = p.scan_cache
    
    info = {
        "scan_completed": False,
        "total_images": 0,
        "cache_entries": 0,
        "cache_size_mb": 0,
        "exact_duplicate_groups": 0,
        "perceptual_duplicate_groups": 0,
        "scan_file_exists": scan_results_path.exists(),
    }
    
    # Load scan results
    if scan_results_path.exists():
        try:
            with open(scan_results_path) as f:
                scan_data = json.load(f)
                info["total_images"] = len(scan_data)
                info["scan_completed"] = True
        except Exception as e:
            logger.error(f"Failed to load scan results: {e}")
    
    # Load cache info
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache_data = json.load(f)
                info["cache_entries"] = len(cache_data.get("entries", {}))
            info["cache_size_mb"] = round(cache_path.stat().st_size / (1024 * 1024), 2)
        except json.JSONDecodeError as e:
            logger.warning(f"Cache file is corrupted (JSON error at line {e.lineno}), will be rebuilt on next scan")
            info["cache_size_mb"] = round(cache_path.stat().st_size / (1024 * 1024), 2)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    # Load duplicate info
    if scan_duplicates_path.exists():
        try:
            with open(scan_duplicates_path) as f:
                dup_data = json.load(f)
                info["exact_duplicate_groups"] = len(dup_data.get("exact_groups", []))
                info["perceptual_duplicate_groups"] = len(dup_data.get("perceptual_groups", []))
                info["md5_mode"] = dup_data.get("md5_mode", "unknown")
        except Exception as e:
            logger.error(f"Failed to load duplicates: {e}")
    
    return info


@app.get("/api/scan-duplicates")
async def get_scan_duplicates():
    """Return deterministic duplicate groups from the scan step (MD5/dHash)."""
    report_path = get_paths().scan_duplicates
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"{report_path.name} not found (run Scan first)")
    try:
        with open(report_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {report_path.name}: {e}")


@app.get("/api/scan-data")
async def get_scan_data():
    """Return full scan data with file sizes."""
    scan_path = get_paths().scan_results
    if not scan_path.exists():
        raise HTTPException(status_code=404, detail=f"{scan_path.name} not found")
    try:
        with open(scan_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scan data: {e}")


@app.get("/api/ai-groups-categorized")
async def get_ai_groups_categorized():
    """Categorize AI groups by overlap with exact/perceptual duplicates."""
    p = get_paths()
    groups_file = p.groups_file
    duplicates_file = p.scan_duplicates
    
    if not groups_file.exists():
        raise HTTPException(status_code=404, detail="AI groups not found (run Generate Embeddings first)")
    
    # Load AI groups
    try:
        with open(groups_file) as f:
            ai_groups = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load AI groups: {e}")
    
    # Load exact and perceptual duplicates
    exact_files = set()
    perceptual_files = set()
    
    if duplicates_file.exists():
        try:
            with open(duplicates_file) as f:
                dup_data = json.load(f)
                
                # Collect all files in exact duplicate groups
                for group in dup_data.get("exact_groups", []):
                    for file_path in group.get("files", []):
                        exact_files.add(file_path)
                
                # Collect all files in perceptual duplicate groups
                for group in dup_data.get("perceptual_groups", []):
                    for file_path in group.get("files", []):
                        perceptual_files.add(file_path)
        except Exception as e:
            logger.warning(f"Failed to load duplicates file: {e}")
    
    # Categorize AI groups
    overlaps_exact = []
    overlaps_perceptual = []
    new_discoveries = []
    
    for group in ai_groups:
        group_files = {file_info["path"] for file_info in group.get("files", [])}
        
        # Check for overlaps
        has_exact_overlap = bool(group_files & exact_files)
        has_perceptual_overlap = bool(group_files & perceptual_files)
        
        if has_exact_overlap:
            overlaps_exact.append(group)
        elif has_perceptual_overlap:
            overlaps_perceptual.append(group)
        else:
            new_discoveries.append(group)
    
    return {
        "overlaps_exact": overlaps_exact,
        "overlaps_perceptual": overlaps_perceptual,
        "new_discoveries": new_discoveries,
        "counts": {
            "overlaps_exact": len(overlaps_exact),
            "overlaps_perceptual": len(overlaps_perceptual),
            "new_discoveries": len(new_discoveries),
            "total": len(ai_groups)
        }
    }


@app.post("/api/scan")
async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """Start scanning for photos."""
    status = get_processing_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="Already processing")
    
    background_tasks.add_task(run_scan, request)
    return {"success": True, "message": "Scan started"}


@app.post("/api/clear-scan")
async def clear_scan():
    """Clear scan results, duplicates, and cache for current mode."""
    global SCAN_DATA, DEMO_SCAN_DATA

    p = get_paths()
    scan_file = p.scan_results
    dup_file = p.scan_duplicates
    cache_file = p.scan_cache

    for path in [scan_file, dup_file]:
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete {path.name}: {e}")

    if cache_file.exists():
        try:
            cache_file.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete cache: {e}")

    if SERVER_MODE == "demo":
        DEMO_SCAN_DATA = {}
    else:
        SCAN_DATA = {}

    return {"success": True, "message": "Scan results cleared"}


@app.post("/api/clear-embeddings")
async def clear_embeddings():
    """Clear all embeddings and groups."""
    global DEMO_SIMILAR_GROUPS, DEMO_EMBEDDING_STORE, SIMILAR_GROUPS, EMBEDDING_STORE
    
    DEMO_SIMILAR_GROUPS = []
    DEMO_EMBEDDING_STORE = None
    SIMILAR_GROUPS = []
    EMBEDDING_STORE = None
    
    # Delete embedding files if they exist
    embeddings_dir = get_paths().embeddings_dir
    for file in ["similar_groups.json", "similar_pairs.json", "embeddings.npy", "metadata.json"]:
        file_path = embeddings_dir / file
        if file_path.exists():
            file_path.unlink()
    
    return {"success": True, "message": "Embeddings cleared"}


@app.post("/api/embeddings")
async def generate_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Generate embeddings from scan results."""
    status = get_processing_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="Already processing")
    
    scan_file = get_paths().scan_results
    
    if not scan_file.exists():
        raise HTTPException(status_code=400, detail="No scan results found. Run scan first.")
    
    if request.estimate:
        # Run estimation instead of full processing
        return await estimate_embeddings(request)
    
    background_tasks.add_task(run_embeddings, request)
    return {"success": True, "message": "Embedding generation started"}


@app.get("/api/image")
async def get_image(path: str, thumb: bool = False):
    """Serve an image file, optionally as a thumbnail."""
    file_path = Path(path)
    
    # If path is relative, resolve it from project root
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {file_path}")
    
    # Check if we have permission to read the file
    try:
        # Try to stat the file to check permissions
        file_path.stat()
        return FileResponse(file_path)
    except PermissionError:
        # Return a helpful error for Photos Library permission issues
        if "Photos Library.photoslibrary" in str(file_path):
            raise HTTPException(
                status_code=403, 
                detail="Cannot access Photos Library. Please enable Full Disk Access in System Settings > Privacy & Security > Full Disk Access, or re-scan with AppleScript export."
            )
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error serving image {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review")
async def review_group(action: ReviewAction):
    """Review a group and take action."""
    if action.group_id >= len(SIMILAR_GROUPS):
        raise HTTPException(status_code=404, detail="Group not found")
    
    group = SIMILAR_GROUPS[action.group_id]
    group["reviewed"] = True
    group["action"] = action.action
    
    deleted_count = 0
    
    if action.action == "not_similar":
        # Add negative feedback to learner
        if FEEDBACK_LEARNER and EMBEDDING_STORE:
            embeddings, file_paths = EMBEDDING_STORE.get_all_embeddings()
            
            # Add all pairs in the group as negative examples
            for i, file1 in enumerate(group["files"]):
                for j, file2 in enumerate(group["files"]):
                    if i < j:
                        # Get embeddings for these files
                        emb1 = EMBEDDING_STORE.get_embedding(file1["path"])
                        emb2 = EMBEDDING_STORE.get_embedding(file2["path"])
                        
                        if emb1 is not None and emb2 is not None:
                            FEEDBACK_LEARNER.add_negative_feedback(emb1, emb2)
            
            # Save feedback
            feedback_path = Path("embeddings") / "feedback.pkl"
            FEEDBACK_LEARNER.save(str(feedback_path))
            logger.info(f"Saved feedback for group {action.group_id}")
    
    elif action.action == "delete_selected" and action.delete_indices:
        # Delete selected images
        for idx in action.delete_indices:
            if idx < len(group["files"]):
                file_path = group["files"][idx]["path"]
                success = await delete_from_photos(file_path)
                if success:
                    deleted_count += 1
    
    return {
        "success": True,
        "group_id": action.group_id,
        "action": action.action,
        "deleted_count": deleted_count,
    }


@app.post("/api/add-to-duplicates")
async def add_to_duplicates(request: dict):
    """Add a photo to the Duplicates album (Photos) or folder (demo mode)."""
    file_path = request.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path required")
    
    # Use demo mode (folder copy) or Photos album based on scan source
    if DEMO_MODE:
        success = await copy_to_duplicates_folder(file_path)
    else:
        success = await add_to_duplicates_album(file_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add photo to duplicates")
    return {"success": True}


@app.post("/api/open-duplicates-album")
async def open_duplicates_album():
    """Open the Duplicates album (Photos) or folder (demo mode)."""
    if DEMO_MODE:
        # In demo mode, open the duplicates folder in Finder
        if SCAN_DATA:
            first_path = Path(SCAN_DATA[0]['file_path'])
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
            raise HTTPException(status_code=400, detail="No scan data available")
    else:
        # Photos Library mode - open album in Photos.app
        script = '''
        tell application "Photos"
            activate
            try
                set duplicatesAlbum to album "Duplicates"
                spotlight duplicatesAlbum
            on error
                display dialog "Duplicates album not found. Add some duplicates first!" buttons {"OK"} default button "OK"
            end try
        end tell
        '''
        
        try:
            subprocess.run(['osascript', '-e', script], check=True, timeout=10)
            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to open Duplicates album: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def run_scan(request: ScanRequest):
    """Run scanner in background."""
    p = get_paths()
    status = get_processing_status()

    status["running"] = True
    status["stage"] = "Scanning"
    status["message"] = "Initializing photo scanner..."
    
    try:
        cmd = [
            "python", "-m", "src.scanner.main",
            "--output", p.scan_output_rel,
            "--duplicates-output", p.dup_output_rel,
            "--cache-file", p.cache_file_rel,
            "--md5-mode", request.md5_mode,
        ]
        
        if request.source == "photos-library":
            cmd.extend(["--photos-library"])
            if request.photos_access == "applescript":
                cmd.extend(["--use-applescript", "--keep-export"])
                status["message"] = "Scanning Photos Library (using AppleScript export)..."
            else:
                status["message"] = "Scanning Photos Library (direct access)..."
        else:
            # Resolve relative paths from project root
            scan_path = Path(request.path)
            if not scan_path.is_absolute():
                scan_path = PROJECT_ROOT / scan_path
            cmd.append(str(scan_path))
            status["message"] = f"Scanning directory: {scan_path.name}..."

        if not request.use_cache:
            cmd.append("--no-cache")
        
        if request.limit:
            cmd.extend(["--limit", str(request.limit)])
            status["message"] += f" (limit: {request.limit} images)"
        
        logger.info(f"Running scan: {' '.join(cmd)}")
        
        status["message"] = "Scanning in progress... This may take a few moments."
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            status["message"] = "Scan completed successfully"
            logger.info("Scan completed")
        else:
            stderr_text = stderr.decode()
            # Write full stderr to log file
            write_stderr_to_log(stderr_text, "scan")
            # Show clean error in UI
            error_msg = clean_stderr_message(stderr_text)
            status["message"] = f"Scan failed: {error_msg}"
            logger.error(f"Scan failed with return code {process.returncode}")
    
    except Exception as e:
        status["message"] = f"Scan error: {str(e)}"
        logger.error(f"Scan error: {e}")
    
    finally:
        status["running"] = False
        status["stage"] = ""


async def estimate_embeddings(request: EmbeddingRequest):
    """Estimate time and resources for embedding generation."""
    scan_file = get_paths().scan_results
    
    try:
        # Load scan data
        with open(scan_file) as f:
            scan_data = json.load(f)
        
        total_images = len(scan_data)
        sample_size = min(request.estimate_sample, total_images)
        
        if total_images == 0:
            return {
                "success": False,
                "error": "No images found in scan results"
            }
        
        logger.info(f"Estimating embedding generation for {total_images} images using {sample_size} samples")
        
        # Import here to avoid startup overhead
        import time
        import random
        from PIL import Image
        from ..embedding.embedder import ImageEmbedder, estimate_embedding_throughput
        
        # Sample random images
        sample_indices = random.sample(range(total_images), sample_size)
        sample_items = [scan_data[i] for i in sample_indices]
        
        # Load sample images
        sample_images = []
        load_failures = 0
        for item in sample_items:
            try:
                img = Image.open(item["file_path"])
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                sample_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {item['file_path']}: {e}")
                load_failures += 1
        
        if len(sample_images) == 0:
            return {
                "success": False,
                "error": "Failed to load any sample images"
            }
        
        # Initialize embedder
        logger.info("Initializing embedding model...")
        model_load_start = time.time()
        embedder = ImageEmbedder(model_name="ViT-B-32", pretrained="openai")
        model_load_time = time.time() - model_load_start
        
        # Estimate throughput
        logger.info(f"Processing {len(sample_images)} sample images...")
        stats = estimate_embedding_throughput(embedder, sample_images, batch_size=32)
        
        # Calculate estimates
        if stats["images_per_second"] > 0:
            estimated_seconds = total_images / stats["images_per_second"]
            estimated_minutes = estimated_seconds / 60
            
            # Estimate grouping time (roughly 10% of embedding time for large sets)
            grouping_seconds = max(1, total_images * total_images * 0.000001)  # O(n²) comparison
            total_seconds = estimated_seconds + grouping_seconds
            total_minutes = total_seconds / 60
        else:
            estimated_minutes = 0
            total_minutes = 0
        
        return {
            "success": True,
            "total_images": total_images,
            "sample_size": len(sample_images),
            "sample_failures": load_failures,
            "model_load_time": round(model_load_time, 2),
            "throughput": {
                "images_per_second": round(stats["images_per_second"], 2),
                "sample_time": round(stats["total_time"], 2),
            },
            "estimates": {
                "embedding_minutes": round(estimated_minutes, 2),
                "grouping_minutes": round(grouping_seconds / 60, 2),
                "total_minutes": round(total_minutes, 2),
                "total_hours": round(total_minutes / 60, 2),
            },
            "note": "Actual time may vary based on system load and image complexity"
        }
        
    except Exception as e:
        logger.error(f"Estimation error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def run_embeddings(request: EmbeddingRequest):
    """Run embedding generation in background."""
    global DEMO_SIMILAR_GROUPS, DEMO_EMBEDDING_STORE, DEMO_THRESHOLD
    global SIMILAR_GROUPS, EMBEDDING_STORE, CURRENT_THRESHOLD
    p = get_paths()
    status = get_processing_status()
    
    status["running"] = True
    status["stage"] = "Generating Embeddings"
    status["message"] = f"Generating embeddings ({request.inference_mode} mode)... This may take several minutes."
    
    try:
        # Use main_v2.py for remote/auto support
        cmd = [
            "python", "-m", "src.embedding.main_v2",
            p.scan_file_rel,
            "--output", p.output_dir_rel,
            "--similarity-threshold", str(request.similarity_threshold),
            "--mode", request.inference_mode,
        ]
        
        # Add service URL if using remote or auto mode
        if request.inference_mode in ("remote", "auto"):
            cmd.extend(["--service-url", INFERENCE_SERVICE_URL])
        
        logger.info(f"Running embeddings: {' '.join(cmd)}")
        
        status["message"] = f"Generating embeddings ({request.inference_mode} mode, threshold: {request.similarity_threshold})... This may take several minutes."
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Reload data
            groups_file = p.groups_file
            if groups_file.exists():
                with open(groups_file) as f:
                    groups_data = json.load(f)
                    # Reset reviewed state
                    for group in groups_data:
                        group["reviewed"] = False
                    
                    if SERVER_MODE == "demo":
                        DEMO_SIMILAR_GROUPS = groups_data
                    else:
                        SIMILAR_GROUPS = groups_data
            
            # Reload embeddings and update threshold
            if SERVER_MODE == "demo":
                DEMO_EMBEDDING_STORE = EmbeddingStore(p.embeddings_dir)
                DEMO_THRESHOLD = request.similarity_threshold
            else:
                EMBEDDING_STORE = EmbeddingStore(p.embeddings_dir)
                CURRENT_THRESHOLD = request.similarity_threshold
            
            status["message"] = f"Found {len(groups_data)} similarity groups"
            logger.info(f"Embeddings completed ({request.inference_mode} mode), found {len(groups_data)} groups")
        else:
            stderr_text = stderr.decode()
            # Write full stderr to log file
            write_stderr_to_log(stderr_text, "embeddings")
            # Show clean error in UI
            error_msg = clean_stderr_message(stderr_text)
            status["message"] = f"Embedding generation failed: {error_msg}"
            logger.error(f"Embedding generation failed with return code {process.returncode}")
    
    except Exception as e:
        status["message"] = f"Embedding error: {str(e)}"
        logger.error(f"Embedding error: {e}")
    
    finally:
        status["running"] = False
        status["stage"] = ""


def clean_stderr_message(stderr: str) -> str:
    """Extract meaningful error message from verbose stderr output."""
    if not stderr:
        return "Unknown error"
    
    lines = stderr.strip().split('\n')
    
    # Filter out progress bars and verbose logging
    meaningful_lines = []
    for line in lines:
        # Skip progress bars (Loading images: X%)
        if 'Loading images:' in line and '%|' in line:
            continue
        # Skip resource tracker warnings (these are just cleanup warnings)
        if 'resource_tracker:' in line:
            continue
        # Skip empty lines
        if not line.strip():
            continue
        # Keep ERROR, WARNING, and exception lines
        if any(keyword in line for keyword in ['ERROR', 'error:', 'Error:', 'Exception', 'Traceback', 'WARNING']):
            meaningful_lines.append(line)
    
    # If we found meaningful error lines, return the last few
    if meaningful_lines:
        return ' | '.join(meaningful_lines[-3:])  # Last 3 error lines
    
    # If no meaningful lines but we have output, return last non-progress line
    for line in reversed(lines):
        if line.strip() and 'Loading images:' not in line and '%|' not in line:
            return line.strip()
    
    return "Process failed - check terminal logs for details"


def write_stderr_to_log(stderr: str, operation: str):
    """Write full stderr output to a log file for debugging."""
    if not stderr:
        return
    
    try:
        from datetime import datetime
        
        # Create logs directory if it doesn't exist
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{operation}_{timestamp}.log"
        
        with open(log_file, 'w') as f:
            f.write(f"Operation: {operation}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(stderr)
        
        logger.info(f"Full stderr logged to: {log_file}")
    except Exception as e:
        logger.error(f"Failed to write stderr log: {e}")


async def add_to_duplicates_album(file_path: str) -> bool:
    """Add a photo to the 'Duplicates' album in Photos.app via AppleScript.
    
    This allows users to review and manually delete duplicates from Photos.app,
    which properly syncs deletions to iCloud and all devices.
    """
    # Try to get original path from mapping (if using cached copies)
    original_path = PATH_MAPPING.get(file_path, file_path)
    uuid_filename = Path(file_path).stem  # Get UUID without extension
    
    # Try to find the original filename from Photos metadata
    original_filename = None
    if PHOTOS_METADATA:
        # Look up by UUID in Photos metadata
        if uuid_filename in PHOTOS_METADATA:
            original_filename = PHOTOS_METADATA[uuid_filename]
    
    # If we don't have metadata, try the filename from the path
    if not original_filename:
        original_filename = Path(original_path).name
    
    logger.info(f"Adding to Duplicates album: {original_filename} (UUID: {uuid_filename})")
    
    # AppleScript to add photo to Duplicates album
    script = f'''
    tell application "Photos"
        -- Get or create Duplicates album
        try
            set duplicatesAlbum to album "Duplicates"
        on error
            set duplicatesAlbum to make new album named "Duplicates"
        end try
        
        -- Find the photo by filename
        set matchingItems to (media items whose filename is "{original_filename}")
        if (count of matchingItems) > 0 then
            set thePhoto to item 1 of matchingItems
            -- Add to album (won't duplicate if already there)
            add {{thePhoto}} to duplicatesAlbum
            return "success"
        else
            return "not_found"
        end if
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"✗ AppleScript error for {original_filename}: {result.stderr.strip()}")
            return False
        
        if "success" in result.stdout:
            logger.info(f"✓ Added to Duplicates album: {original_filename}")
            return True
        else:
            logger.warning(f"✗ Photo not found in Photos Library: {original_filename}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout adding {original_filename} after 30s")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to add photo {original_filename}: {e}")
        return False


async def copy_to_duplicates_folder(file_path: str) -> bool:
    """Demo mode: Copy file to a 'duplicates' subfolder.
    
    For use with directory-based scanning when not using Photos Library.
    Creates a visual demonstration of duplicate management.
    """
    try:
        source_file = Path(file_path)
        if not source_file.exists():
            logger.error(f"Source file not found: {file_path}")
            return False
        
        # Create duplicates folder in the same directory as the source
        duplicates_dir = source_file.parent / "duplicates"
        duplicates_dir.mkdir(exist_ok=True)
        
        # Copy file to duplicates folder
        dest_file = duplicates_dir / source_file.name
        
        # If file already exists in duplicates, don't copy again
        if dest_file.exists():
            logger.info(f"Already in duplicates folder: {source_file.name}")
            return True
        
        shutil.copy2(source_file, dest_file)
        logger.info(f"✓ Copied to duplicates folder: {source_file.name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to copy {file_path} to duplicates folder: {e}")
        return False


@app.post("/api/batch-delete")
async def batch_delete(group_type: str):
    """Batch delete duplicates from a specific type (keeping oldest)."""
    p = get_paths()
    scan_path = p.scan_results
    duplicates_path = p.scan_duplicates
    
    if not duplicates_path.exists():
        raise HTTPException(status_code=404, detail="No scan duplicates found")
    
    # Load scan data to get dates
    scan_data_map = {}
    if scan_path.exists():
        with open(scan_path) as f:
            scan_data = json.load(f)
            for item in scan_data:
                scan_data_map[item["file_path"]] = item
    
    with open(duplicates_path) as f:
        dup_data = json.load(f)
    
    deleted_count = 0
    failed_count = 0
    groups_processed = 0
    
    if group_type == "exact":
        groups = dup_data.get("exact_groups", [])
    elif group_type == "perceptual":
        groups = dup_data.get("perceptual_groups", [])
    else:
        raise HTTPException(status_code=400, detail="Invalid group_type")
    
    for group in groups:
        files = group.get("files", [])
        if len(files) < 2:
            continue
        
        # Get dates for all files in the group (prefer photo_date from EXIF)
        files_with_dates = []
        for file_path in files:
            item = scan_data_map.get(file_path, {})
            # Use photo_date (actual photo capture) if available, otherwise fall back to created_at (file creation)
            date = item.get("photo_date") or item.get("created_at", "9999-99-99")
            files_with_dates.append((file_path, date))
        
        # Sort by date (oldest first)
        files_with_dates.sort(key=lambda x: x[1])
        
        # Keep the oldest, delete the rest
        for file_path, _ in files_with_dates[1:]:
            success = await delete_from_photos(file_path)
            if success:
                deleted_count += 1
            else:
                failed_count += 1
        
        groups_processed += 1
    
    return {
        "success": deleted_count > 0 or failed_count == 0,
        "groups_processed": groups_processed,
        "deleted_count": deleted_count,
        "failed_count": failed_count
    }


def load_data(
    scan_results: Path,
    similar_groups: Path,
    embeddings_dir: Path,
    path_mapping_file: Optional[Path] = None,
    is_demo: bool = False,
):
    """Load scan results, groups, and initialize feedback learner."""
    global SERVER_MODE
    if is_demo:
        SERVER_MODE = "demo"
        global DEMO_SIMILAR_GROUPS, DEMO_SCAN_DATA, DEMO_FEEDBACK_LEARNER, DEMO_EMBEDDING_STORE
        groups_var = "DEMO_SIMILAR_GROUPS"
        scan_var = "DEMO_SCAN_DATA"
        learner_var = "DEMO_FEEDBACK_LEARNER"
        store_var = "DEMO_EMBEDDING_STORE"
    else:
        SERVER_MODE = "main"
        global SIMILAR_GROUPS, SCAN_DATA, FEEDBACK_LEARNER, EMBEDDING_STORE, PATH_MAPPING, PHOTOS_METADATA, DEMO_MODE
        groups_var = "SIMILAR_GROUPS"
        scan_var = "SCAN_DATA"
        learner_var = "FEEDBACK_LEARNER"
        store_var = "EMBEDDING_STORE"
    
    if similar_groups.exists():
        with open(similar_groups, "r") as f:
            groups_data = json.load(f)
        if is_demo:
            DEMO_SIMILAR_GROUPS = groups_data
        else:
            SIMILAR_GROUPS = groups_data
        logger.info(f"[{'DEMO' if is_demo else 'MAIN'}] Loaded {len(groups_data)} similar groups")
    
    if scan_results.exists():
        with open(scan_results, "r") as f:
            scan_data = json.load(f)
        if is_demo:
            DEMO_SCAN_DATA = scan_data
        else:
            SCAN_DATA = scan_data
        logger.info(f"[{'DEMO' if is_demo else 'MAIN'}] Loaded {len(scan_data)} scan results")
        
        # Detect demo mode for main app (not applicable for demo endpoint)
        if not is_demo and scan_data:
            first_path = scan_data[0].get('file_path', '')
            #DEMO_MODE = 'Photos Library.photoslibrary' not in first_path
            DEMO_MODE = is_demo
            mode_str = "DEMO MODE (directory)" if DEMO_MODE else "Photos Library mode"
            logger.info(f"Detected {mode_str}")
    
    # Load path mapping if available
    if path_mapping_file and path_mapping_file.exists():
        with open(path_mapping_file, "r") as f:
            PATH_MAPPING = json.load(f)
        logger.info(f"Loaded path mapping with {len(PATH_MAPPING)} entries")
    else:
        logger.info("No path mapping file found - delete will use filename-only matching")
    
    # Load photos metadata (UUID -> original filename mapping)
    photos_metadata_file = PROJECT_ROOT / ".cache" / "photos_metadata.json"
    if photos_metadata_file.exists():
        try:
            with open(photos_metadata_file, "r") as f:
                metadata_list = json.load(f)
            # Create UUID -> original filename mapping
            for item in metadata_list:
                # Extract UUID from Photos ID (format varies, but UUID is usually in there)
                photo_id = item.get('id', '')
                filename = item.get('filename', '')
                # Try to extract UUID from the ID
                # Common formats: "AABBCCDD-1122-3344-5566-778899AABBCC/L0/001" or just the UUID
                uuid_part = photo_id.split('/')[0] if '/' in photo_id else photo_id
                if uuid_part and filename:
                    PHOTOS_METADATA[uuid_part] = filename
            logger.info(f"Loaded Photos metadata with {len(PHOTOS_METADATA)} UUID->filename mappings")
        except Exception as e:
            logger.warning(f"Failed to load photos metadata: {e}")
    else:
        logger.warning(f"Photos metadata not found at {photos_metadata_file}")
        logger.warning("Delete operations may fail. Run fetch_photos_dates.py to generate metadata.")
    
    # Initialize feedback learner
    FEEDBACK_LEARNER = FeedbackLearner()
    feedback_path = embeddings_dir / "feedback.pkl"
    if feedback_path.exists():
        FEEDBACK_LEARNER.load(str(feedback_path))
    
    # Load embedding store
    EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
    logger.info(f"Loaded {len(EMBEDDING_STORE)} embeddings")
