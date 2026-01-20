"""FastAPI application for duplicate photo review with groups and feedback."""

import logging
from pathlib import Path
from typing import List, Optional
import json
import subprocess
import asyncio

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

# Global state
SIMILAR_GROUPS: List[dict] = []
SCAN_DATA: dict = {}
FEEDBACK_LEARNER: Optional[FeedbackLearner] = None
EMBEDDING_STORE: Optional[EmbeddingStore] = None
PROCESSING_STATUS: dict = {"running": False, "stage": "", "message": ""}
CURRENT_THRESHOLD: float = 0.85
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


class ReviewAction(BaseModel):
    """Model for review action."""
    group_id: int
    action: str  # "keep_all", "not_similar", or list of indices to delete
    delete_indices: Optional[List[int]] = None


class ScanRequest(BaseModel):
    """Model for scan request."""
    source: str  # "photos-library" or "directory"
    path: Optional[str] = None
    limit: Optional[int] = 20


class EmbeddingRequest(BaseModel):
    """Model for embedding generation request."""
    similarity_threshold: float = 0.85


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main review interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Photo Duplicate Reviewer</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #f5f5f7;
                color: #1d1d1f;
            }
            .header {
                background: white;
                padding: 1rem 2rem;
                border-bottom: 1px solid #d2d2d7;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            .header h1 {
                font-size: 1.5rem;
                font-weight: 600;
            }
            .control-panel {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            .control-panel h2 {
                font-size: 1.125rem;
                margin-bottom: 1rem;
                color: #1d1d1f;
            }
            .control-row {
                display: flex;
                gap: 1rem;
                align-items: center;
                margin-bottom: 0.75rem;
                flex-wrap: wrap;
            }
            .control-input {
                padding: 0.5rem 0.75rem;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                font-size: 0.875rem;
            }
            .processing-status {
                padding: 1rem;
                background: #f5f5f7;
                border-radius: 8px;
                margin-top: 1rem;
                display: none;
            }
            .processing-status.active {
                display: block;
            }
            .processing-status.running {
                background: #e3f2fd;
                border: 1px solid #2196f3;
            }
            .processing-status.success {
                background: #e8f5e9;
                border: 1px solid #4caf50;
            }
            .processing-status.error {
                background: #ffebee;
                border: 1px solid #f44336;
            }
            .stats {
                margin-top: 0.5rem;
                font-size: 0.875rem;
                color: #86868b;
            }
            .current-threshold {
                display: inline-block;
                background: #e3f2fd;
                color: #1976d2;
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
                font-size: 0.875rem;
                font-weight: 600;
                margin-left: 1rem;
            }
            .container {
                max-width: 1600px;
                margin: 2rem auto;
                padding: 0 2rem;
            }
            .group-card {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            .group-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
            }
            .similarity-badge {
                background: #007aff;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.875rem;
            }
            .images-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .image-item {
                position: relative;
                text-align: center;
                border: 3px solid transparent;
                border-radius: 8px;
                padding: 0.5rem;
                transition: all 0.2s;
            }
            .image-item.selected {
                border-color: #ff3b30;
                background: #fff5f5;
            }
            .image-item img {
                max-width: 100%;
                max-height: 400px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                cursor: pointer;
            }
            .image-checkbox {
                position: absolute;
                top: 1rem;
                right: 1rem;
                width: 24px;
                height: 24px;
                cursor: pointer;
            }
            .image-info {
                margin-top: 0.75rem;
                font-size: 0.875rem;
                color: #86868b;
                word-break: break-all;
            }
            .actions {
                display: flex;
                gap: 1rem;
                justify-content: center;
                flex-wrap: wrap;
            }
            .btn {
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }
            .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            .btn-keep {
                background: #34c759;
                color: white;
            }
            .btn-delete {
                background: #ff3b30;
                color: white;
            }
            .btn-not-similar {
                background: #ff9500;
                color: white;
            }
            .loading {
                text-align: center;
                padding: 4rem;
                color: #86868b;
            }
            .reviewed-badge {
                text-align: center;
                margin-top: 1rem;
                padding: 0.75rem;
                background: #f5f5f7;
                border-radius: 8px;
                color: #34c759;
                font-weight: 600;
            }
            .feedback-notice {
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                color: #856404;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📸 Photo Duplicate Reviewer</h1>
            <div class="stats">
                <span id="stats-total">Loading...</span>
                <span id="stats-feedback" style="margin-left: 1rem;"></span>
                <span id="stats-threshold" class="current-threshold"></span>
            </div>
        </div>

        <div class="container">
            <div class="control-panel">
                <h2>🔧 Control Panel</h2>
                
                <div class="control-row">
                    <strong>Step 1: Scan Photos</strong>
                    <select id="scan-source" class="control-input">
                        <option value="photos-library">Photos Library</option>
                        <option value="directory">Directory</option>
                    </select>
                    <input id="scan-path" type="text" class="control-input" placeholder="Directory path" style="display: none; flex: 1;">
                    <input id="scan-limit" type="number" class="control-input" placeholder="Limit (default: 20)" value="20" style="width: 150px;">
                    <button class="btn btn-keep" onclick="startScan()" id="btn-scan">▶️ Start Scan</button>
                </div>
                
                <div class="control-row">
                    <strong>Step 2: Generate Embeddings</strong>
                    <label style="font-size: 0.875rem; color: #86868b;">Similarity threshold:</label>
                    <input id="similarity-threshold" type="number" class="control-input" placeholder="Threshold" value="0.85" step="0.01" min="0" max="1" style="width: 100px;">
                    <span style="font-size: 0.75rem; color: #86868b;">(Higher = more strict, e.g., 0.99 = nearly identical)</span>
                    <button class="btn btn-keep" onclick="startEmbeddings()" id="btn-embeddings">🧠 Generate Embeddings</button>
                </div>
                
                <div class="control-row">
                    <strong>After Giving Feedback:</strong>
                    <button class="btn btn-not-similar" onclick="startEmbeddings()">🔄 Re-analyze with Feedback</button>
                </div>
                
                <div class="control-row">
                    <strong>Start Fresh:</strong>
                    <button class="btn btn-delete" onclick="clearState()">🗑️ Clear All Results</button>
                    <span style="font-size: 0.75rem; color: #86868b;">(Clears displayed groups, doesn't delete files)</span>
                </div>
                
                <div id="processing-status" class="processing-status"></div>
            </div>

            <div id="groups-container">
                <div class="loading">Loading similarity groups...</div>
            </div>
        </div>

        <script>
            let groups = [];
            let selectedImages = new Map(); // groupId -> Set of selected indices
            let statusCheckInterval = null;

            // Setup scan source selector
            document.addEventListener('DOMContentLoaded', () => {
                const scanSource = document.getElementById('scan-source');
                const scanPath = document.getElementById('scan-path');
                
                scanSource.addEventListener('change', (e) => {
                    if (e.target.value === 'directory') {
                        scanPath.style.display = 'block';
                        scanPath.required = true;
                    } else {
                        scanPath.style.display = 'none';
                        scanPath.required = false;
                    }
                });
            });

            async function startScan() {
                const source = document.getElementById('scan-source').value;
                const path = document.getElementById('scan-path').value;
                const limit = parseInt(document.getElementById('scan-limit').value) || 20;
                
                if (source === 'directory' && !path) {
                    alert('Please enter a directory path');
                    return;
                }
                
                const btnScan = document.getElementById('btn-scan');
                btnScan.disabled = true;
                
                try {
                    const response = await fetch('/api/scan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ source, path, limit })
                    });
                    
                    if (response.ok) {
                        startStatusPolling();
                    } else {
                        const error = await response.json();
                        showStatus('error', error.detail || 'Scan failed');
                        btnScan.disabled = false;
                    }
                } catch (error) {
                    showStatus('error', 'Failed to start scan: ' + error.message);
                    btnScan.disabled = false;
                }
            }

            async function startEmbeddings() {
                const threshold = parseFloat(document.getElementById('similarity-threshold').value) || 0.85;
                
                const btnEmbeddings = document.getElementById('btn-embeddings');
                btnEmbeddings.disabled = true;
                
                try {
                    const response = await fetch('/api/embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ similarity_threshold: threshold })
                    });
                    
                    if (response.ok) {
                        startStatusPolling();
                    } else {
                        const error = await response.json();
                        showStatus('error', error.detail || 'Embedding generation failed');
                        btnEmbeddings.disabled = false;
                    }
                } catch (error) {
                    showStatus('error', 'Failed to start embeddings: ' + error.message);
                    btnEmbeddings.disabled = false;
                }
            }

            function startStatusPolling() {
                if (statusCheckInterval) return;
                
                statusCheckInterval = setInterval(async () => {
                    try {
                        const response = await fetch('/api/status');
                        const status = await response.json();
                        
                        if (status.running) {
                            showStatus('running', `${status.stage}: ${status.message}`);
                        } else {
                            showStatus('success', status.message);
                            clearInterval(statusCheckInterval);
                            statusCheckInterval = null;
                            
                            // Re-enable buttons
                            document.getElementById('btn-scan').disabled = false;
                            document.getElementById('btn-embeddings').disabled = false;
                            
                            // Reload groups
                            loadGroups();
                        }
                    } catch (error) {
                        console.error('Status check failed:', error);
                    }
                }, 1000);
            }

            function showStatus(type, message) {
                const statusDiv = document.getElementById('processing-status');
                statusDiv.className = `processing-status active ${type}`;
                statusDiv.textContent = message;
            }

            async function clearState() {
                if (!confirm('Clear all displayed results? This will not delete any files, only reset the view.')) {
                    return;
                }
                
                try {
                    const response = await fetch('/api/clear', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    if (response.ok) {
                        groups = [];
                        selectedImages.clear();
                        document.getElementById('similarity-threshold').value = '0.85';
                        showStatus('success', 'State cleared - ready for new scan');
                        updateStats(0.85);
                        renderGroups();
                    } else {
                        alert('Failed to clear state');
                    }
                } catch (error) {
                    console.error('Clear failed:', error);
                    alert('Failed to clear state');
                }
            }

            async function loadGroups() {
                try {
                    const response = await fetch('/api/groups');
                    const data = await response.json();
                    groups = data.groups || data; // Handle both old and new format
                    const threshold = data.threshold || 0.85;
                    
                    updateStats(threshold);
                    renderGroups();
                } catch (error) {
                    console.error('Failed to load groups:', error);
                    document.getElementById('groups-container').innerHTML = 
                        '<div class="loading">Failed to load groups</div>';
                }
            }

            function updateStats(threshold) {
                const total = groups.length;
                const reviewed = groups.filter(g => g.reviewed).length;
                const feedbackCount = groups.filter(g => g.action === 'not_similar').length;
                
                document.getElementById('stats-total').textContent = 
                    `${reviewed} / ${total} groups reviewed`;
                
                document.getElementById('stats-threshold').textContent = 
                    `Current threshold: ${(threshold * 100).toFixed(0)}%`;
                
                if (feedbackCount > 0) {
                    document.getElementById('stats-feedback').textContent = 
                        `🎓 ${feedbackCount} feedback examples learned`;
                }
            }

            function toggleImageSelection(groupId, imageIndex) {
                if (!selectedImages.has(groupId)) {
                    selectedImages.set(groupId, new Set());
                }
                
                const selected = selectedImages.get(groupId);
                if (selected.has(imageIndex)) {
                    selected.delete(imageIndex);
                } else {
                    selected.add(imageIndex);
                }
                
                renderGroups();
            }

            function renderGroups() {
                const container = document.getElementById('groups-container');
                
                if (groups.length === 0) {
                    container.innerHTML = `
                        <div class="group-card">
                            <h2>✅ No duplicate groups found</h2>
                            <p>Run the scanner with more photos to find duplicates.</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = groups.map((group, index) => {
                    const selected = selectedImages.get(group.group_id) || new Set();
                    const hasSelection = selected.size > 0;
                    
                    return `
                        <div class="group-card" style="${group.reviewed ? 'opacity: 0.6;' : ''}">
                            <div class="group-header">
                                <h3>Group ${index + 1} of ${groups.length} • ${group.size} images</h3>
                                <span class="similarity-badge">${(group.avg_similarity * 100).toFixed(1)}% similar</span>
                            </div>
                            
                            ${!group.reviewed ? `
                                <div class="feedback-notice">
                                    💡 <strong>Tip:</strong> Click images to select which ones to delete, or use buttons below
                                </div>
                            ` : ''}
                            
                            <div class="images-grid">
                                ${group.files.map((file, fileIdx) => `
                                    <div class="image-item ${selected.has(fileIdx) ? 'selected' : ''}" 
                                         onclick="toggleImageSelection(${group.group_id}, ${fileIdx})">
                                        <input type="checkbox" 
                                               class="image-checkbox" 
                                               ${selected.has(fileIdx) ? 'checked' : ''}
                                               onclick="event.stopPropagation()">
                                        <img src="/api/image?path=${encodeURIComponent(file.path)}" 
                                             alt="Image ${fileIdx + 1}">
                                        <div class="image-info">${file.name}</div>
                                    </div>
                                `).join('')}
                            </div>
                            
                            <div class="actions">
                                <button class="btn btn-keep" 
                                        onclick="handleAction(${group.group_id}, 'keep_all')"
                                        ${group.reviewed ? 'disabled' : ''}>
                                    ✓ Keep All
                                </button>
                                <button class="btn btn-delete" 
                                        onclick="handleDeleteSelected(${group.group_id})"
                                        ${group.reviewed || !hasSelection ? 'disabled' : ''}>
                                    🗑️ Delete Selected (${selected.size})
                                </button>
                                <button class="btn btn-not-similar" 
                                        onclick="handleAction(${group.group_id}, 'not_similar')"
                                        ${group.reviewed ? 'disabled' : ''}>
                                    ✖️ Not Similar (Teach AI)
                                </button>
                            </div>
                            
                            ${group.reviewed ? `
                                <div class="reviewed-badge">
                                    ✓ Reviewed - ${group.action === 'not_similar' ? 'Marked as not similar' : 
                                       group.action === 'keep_all' ? 'Kept all images' : 
                                       `Deleted ${group.deleted_count || 0} image(s)`}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }).join('');
            }

            async function handleAction(groupId, action) {
                try {
                    const response = await fetch('/api/review', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            group_id: groupId, 
                            action: action 
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        // Update local state
                        const group = groups.find(g => g.group_id === groupId);
                        if (group) {
                            group.reviewed = true;
                            group.action = action;
                            if (result.deleted_count) {
                                group.deleted_count = result.deleted_count;
                            }
                        }
                        
                        // Clear selection
                        selectedImages.delete(groupId);
                        
                        updateStats();
                        renderGroups();
                        
                        if (action === 'not_similar') {
                            alert('✓ Feedback recorded! AI will learn these images are not similar.');
                        } else if (result.deleted_count) {
                            alert(`✓ ${result.deleted_count} photo(s) deleted from Photos Library`);
                        }
                    } else {
                        alert('Failed to process action');
                    }
                } catch (error) {
                    console.error('Action failed:', error);
                    alert('Failed to process action');
                }
            }

            async function handleDeleteSelected(groupId) {
                const selected = selectedImages.get(groupId);
                if (!selected || selected.size === 0) return;
                
                const count = selected.size;
                if (!confirm(`Delete ${count} selected photo(s) from Photos Library?`)) {
                    return;
                }
                
                try {
                    const response = await fetch('/api/review', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            group_id: groupId, 
                            action: 'delete_selected',
                            delete_indices: Array.from(selected)
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        const group = groups.find(g => g.group_id === groupId);
                        if (group) {
                            group.reviewed = true;
                            group.action = 'delete_selected';
                            group.deleted_count = result.deleted_count;
                        }
                        
                        selectedImages.delete(groupId);
                        updateStats();
                        renderGroups();
                        
                        alert(`✓ ${result.deleted_count} photo(s) deleted from Photos Library`);
                    } else {
                        alert('Failed to delete photos');
                    }
                } catch (error) {
                    console.error('Delete failed:', error);
                    alert('Failed to delete photos');
                }
            }

            loadGroups();
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/api/groups")
async def get_groups():
    """Get all similarity groups."""
    return {
        "groups": SIMILAR_GROUPS,
        "threshold": CURRENT_THRESHOLD
    }


@app.get("/api/status")
async def get_status():
    """Get processing status."""
    return PROCESSING_STATUS


@app.post("/api/scan")
async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """Start scanning for photos."""
    if PROCESSING_STATUS["running"]:
        raise HTTPException(status_code=409, detail="Already processing")
    
    background_tasks.add_task(run_scan, request)
    return {"success": True, "message": "Scan started"}


@app.post("/api/embeddings")
async def generate_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Generate embeddings from scan results."""
    if PROCESSING_STATUS["running"]:
        raise HTTPException(status_code=409, detail="Already processing")
    
    scan_file = PROJECT_ROOT / "scan_for_embeddings.json"
    if not scan_file.exists():
        raise HTTPException(status_code=400, detail="No scan results found. Run scan first.")
    
    background_tasks.add_task(run_embeddings, request)
    return {"success": True, "message": "Embedding generation started"}


@app.post("/api/clear")
async def clear_state():
    """Clear all state and results."""
    global SIMILAR_GROUPS, SCAN_DATA, CURRENT_THRESHOLD
    
    SIMILAR_GROUPS = []
    SCAN_DATA = {}
    CURRENT_THRESHOLD = 0.85
    
    logger.info("Cleared all state")
    return {"success": True, "message": "State cleared"}


@app.get("/api/image")
async def get_image(path: str):
    """Serve an image file."""
    file_path = Path(path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)


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


async def run_scan(request: ScanRequest):
    """Run scanner in background."""
    global PROCESSING_STATUS
    PROCESSING_STATUS["running"] = True
    PROCESSING_STATUS["stage"] = "scanning"
    PROCESSING_STATUS["message"] = "Scanning for photos..."
    
    try:
        cmd = [
            "python", "-m", "src.scanner.main",
            "--output", "scan_for_embeddings.json"
        ]
        
        if request.source == "photos-library":
            cmd.extend(["--photos-library", "--use-applescript", "--keep-export"])
        else:
            cmd.append(request.path)
        
        if request.limit:
            cmd.extend(["--limit", str(request.limit)])
        
        logger.info(f"Running scan: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            PROCESSING_STATUS["message"] = "Scan completed successfully"
            logger.info("Scan completed")
        else:
            PROCESSING_STATUS["message"] = f"Scan failed: {stderr.decode()}"
            logger.error(f"Scan failed: {stderr.decode()}")
    
    except Exception as e:
        PROCESSING_STATUS["message"] = f"Scan error: {str(e)}"
        logger.error(f"Scan error: {e}")
    
    finally:
        PROCESSING_STATUS["running"] = False
        PROCESSING_STATUS["stage"] = ""


async def run_embeddings(request: EmbeddingRequest):
    """Run embedding generation in background."""
    global PROCESSING_STATUS, SIMILAR_GROUPS, EMBEDDING_STORE, CURRENT_THRESHOLD
    PROCESSING_STATUS["running"] = True
    PROCESSING_STATUS["stage"] = "embeddings"
    PROCESSING_STATUS["message"] = "Generating embeddings..."
    
    try:
        cmd = [
            "python", "-m", "src.embedding.main",
            "scan_for_embeddings.json",
            "--output", "embeddings",
            "--similarity-threshold", str(request.similarity_threshold)
        ]
        
        logger.info(f"Running embeddings: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Reload data
            groups_file = PROJECT_ROOT / "embeddings" / "similar_groups.json"
            if groups_file.exists():
                with open(groups_file) as f:
                    SIMILAR_GROUPS = json.load(f)
                    # Reset reviewed state
                    for group in SIMILAR_GROUPS:
                        group["reviewed"] = False
            
            # Reload embeddings
            embeddings_dir = PROJECT_ROOT / "embeddings"
            EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
            
            # Update current threshold
            CURRENT_THRESHOLD = request.similarity_threshold
            
            PROCESSING_STATUS["message"] = f"Found {len(SIMILAR_GROUPS)} similarity groups"
            logger.info(f"Embeddings completed, found {len(SIMILAR_GROUPS)} groups")
        else:
            PROCESSING_STATUS["message"] = f"Embedding generation failed: {stderr.decode()}"
            logger.error(f"Embedding generation failed: {stderr.decode()}")
    
    except Exception as e:
        PROCESSING_STATUS["message"] = f"Embedding error: {str(e)}"
        logger.error(f"Embedding error: {e}")
    
    finally:
        PROCESSING_STATUS["running"] = False
        PROCESSING_STATUS["stage"] = ""


async def delete_from_photos(file_path: str) -> bool:
    """Delete a photo from Photos Library via AppleScript."""
    filename = Path(file_path).name
    
    script = f'''
    tell application "Photos"
        set itemList to media items
        repeat with theItem in itemList
            if filename of theItem is "{filename}" then
                delete theItem
                return "success"
            end if
        end repeat
        return "not_found"
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        if "success" in result.stdout:
            logger.info(f"Deleted from Photos: {filename}")
            return True
        else:
            logger.warning(f"Photo not found in Photos Library: {filename}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to delete photo: {e}")
        return False


def load_data(
    scan_results: Path,
    similar_groups: Path,
    embeddings_dir: Path,
):
    """Load scan results, groups, and initialize feedback learner."""
    global SIMILAR_GROUPS, SCAN_DATA, FEEDBACK_LEARNER, EMBEDDING_STORE
    
    if similar_groups.exists():
        with open(similar_groups, "r") as f:
            SIMILAR_GROUPS = json.load(f)
        logger.info(f"Loaded {len(SIMILAR_GROUPS)} similar groups")
    
    if scan_results.exists():
        with open(scan_results, "r") as f:
            SCAN_DATA = json.load(f)
        logger.info(f"Loaded {len(SCAN_DATA)} scan results")
    
    # Initialize feedback learner
    FEEDBACK_LEARNER = FeedbackLearner()
    feedback_path = embeddings_dir / "feedback.pkl"
    if feedback_path.exists():
        FEEDBACK_LEARNER.load(str(feedback_path))
    
    # Load embedding store
    EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
    logger.info(f"Loaded {len(EMBEDDING_STORE)} embeddings")
