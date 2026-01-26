"""FastAPI application for duplicate photo review with groups and feedback."""

import logging
from pathlib import Path
from typing import List, Optional
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
                font-weight: 500;
            }
            .processing-status.active {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            .processing-status.running {
                background: #e3f2fd;
                border: 1px solid #2196f3;
                color: #1565c0;
            }
            .processing-status.success {
                background: #e8f5e9;
                border: 1px solid #4caf50;
                color: #2e7d32;
            }
            .processing-status.error {
                background: #ffebee;
                border: 1px solid #f44336;
                color: #c62828;
            }
            .spinner {
                width: 20px;
                height: 20px;
                border: 3px solid rgba(0, 0, 0, 0.1);
                border-top-color: #2196f3;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
                flex-shrink: 0;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .status-icon {
                font-size: 1.25rem;
                flex-shrink: 0;
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
                max-height: 120px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                cursor: pointer;
                transition: all 0.3s;
                object-fit: cover;
            }
            .image-item img.expanded {
                max-height: 600px;
                object-fit: contain;
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
            .tabs {
                display: flex;
                gap: 0.5rem;
                margin-bottom: 1.5rem;
                border-bottom: 2px solid #d2d2d7;
            }
            .tab {
                padding: 0.75rem 1.5rem;
                border: none;
                background: transparent;
                cursor: pointer;
                font-size: 1rem;
                font-weight: 500;
                color: #86868b;
                border-bottom: 3px solid transparent;
                transition: all 0.2s;
            }
            .tab:hover {
                color: #1d1d1f;
            }
            .tab.active {
                color: #007aff;
                border-bottom-color: #007aff;
            }
            .tab:disabled {
                opacity: 0.4;
                cursor: not-allowed;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .type-section {
                margin-bottom: 2rem;
            }
            .type-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 12px 12px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0;
                cursor: pointer;
                user-select: none;
            }
            .type-header:hover {
                opacity: 0.95;
            }
            .type-header.collapsed {
                border-radius: 12px;
                margin-bottom: 1rem;
            }
            .type-content {
                max-height: 10000px;
                overflow: hidden;
                transition: max-height 0.3s ease;
            }
            .type-content.collapsed {
                max-height: 0;
            }
            .type-header.exact {
                background: linear-gradient(135deg, #34c759 0%, #30d158 100%);
            }
            .type-header.perceptual {
                background: linear-gradient(135deg, #ff9500 0%, #ffb347 100%);
            }
            .type-header.ai {
                background: linear-gradient(135deg, #007aff 0%, #5ac8fa 100%);
            }
            .groups-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .mini-group-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                border: 2px solid transparent;
                transition: all 0.2s;
                cursor: pointer;
            }
            .mini-group-card.selected {
                border-color: #007aff;
                background: #f0f8ff;
            }
            .mini-group-card:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.12);
            }
            .mini-group-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.75rem;
            }
            .mini-images-preview {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
                gap: 0.5rem;
                margin-bottom: 0.75rem;
            }
            .mini-images-preview img {
                width: 100%;
                height: 80px;
                object-fit: cover;
                border-radius: 4px;
            }
            .group-stats {
                font-size: 0.75rem;
                color: #86868b;
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            .selection-summary {
                position: sticky;
                top: 4rem;
                z-index: 50;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .selection-summary.exact {
                background: linear-gradient(135deg, #34c759 0%, #30d158 100%);
            }
            .selection-summary.perceptual {
                background: linear-gradient(135deg, #ff9500 0%, #ffb347 100%);
            }
            .selection-actions {
                display: flex;
                gap: 0.5rem;
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.95);
                overflow: auto;
            }
            .modal.active {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .modal-content {
                max-width: 90%;
                max-height: 90vh;
                object-fit: contain;
            }
            .modal-close {
                position: absolute;
                top: 20px;
                right: 40px;
                color: white;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
            }
            .modal-close:hover {
                color: #ccc;
            }
            .modal-info {
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                color: white;
                background: rgba(0,0,0,0.7);
                padding: 1rem;
                border-radius: 8px;
            }
            input[type="checkbox"] {
                accent-color: #8e8e93;
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
            <!-- Workflow Step Indicator -->
            <div style="display: flex; gap: 2rem; margin: 1.5rem 0; padding: 1rem; background: #f5f5f7; border-radius: 8px; align-items: center;">
                <div style="flex: 1;">
                    <button onclick="switchTab('step1')" id="step1-btn" 
                            style="display: flex; align-items: center; gap: 1rem; padding: 1rem 1.5rem; border: 2px solid #0071e3; background: white; border-radius: 8px; cursor: pointer; width: 100%; font-size: 1rem; font-weight: 600; color: #0071e3;">
                        <span style="display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: #0071e3; color: white; border-radius: 50%; font-weight: bold;">1</span>
                        <span>Scanner</span>
                    </button>
                </div>
                <div style="width: 2px; height: 40px; background: #d2d2d7;"></div>
                <div style="flex: 1;">
                    <button onclick="switchTab('step2')" id="step2-btn" 
                            style="display: flex; align-items: center; gap: 1rem; padding: 1rem 1.5rem; border: 2px solid #d2d2d7; background: white; border-radius: 8px; cursor: pointer; width: 100%; font-size: 1rem; font-weight: 600; color: #86868b;">
                        <span style="display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: #d2d2d7; color: white; border-radius: 50%; font-weight: bold;">2</span>
                        <span>AI Inference/Embedding</span>
                    </button>
                </div>
            </div>

            <!-- Step 1: Scanner Tab -->
            <div id="step1-tab" class="control-panel" style="display: block;">
                <h2>🔍 Step 1: Scanner</h2>
                
                <div class="control-row">
                    <label style="font-size: 0.875rem; color: #86868b; font-weight: 600;">Scan Source:</label>
                    <select id="scan-source" class="control-input">
                        <option value="photos-library">Photos Library</option>
                        <option value="directory">Directory</option>
                    </select>
                    <select id="photos-access" class="control-input">
                        <option value="applescript">AppleScript export (no Full Disk Access, slower)</option>
                        <option value="originals">Direct originals (requires Full Disk Access, faster)</option>
                    </select>
                    <input id="scan-path" type="text" class="control-input" placeholder="Directory path" style="display: none; flex: 1;">
                    <input id="scan-limit" type="number" class="control-input" placeholder="Limit (default: 20)" value="20" style="width: 150px;">
                    <label style="font-size: 0.875rem; color: #86868b; display: flex; align-items: center; gap: 0.5rem;">
                        <input id="use-cache" type="checkbox" checked>
                        Use cache
                    </label>
                    <select id="md5-mode" class="control-input" title="MD5 strategy">
                        <option value="on-demand">MD5: on-demand (recommended)</option>
                        <option value="always">MD5: always</option>
                        <option value="never">MD5: never</option>
                    </select>
                    <button class="btn btn-keep" onclick="startScan()" id="btn-scan">▶️ Start Scan</button>
                    <button class="btn" onclick="clearScan()" id="btn-clear-scan" style="background: #a2aaad; color: white;">🗑️ Clear Scan</button>
                </div>
                
                <div id="scan-info" style="display: none; background: #f5f5f7; padding: 0.75rem 1rem; border-radius: 6px; margin-left: 2rem; margin-bottom: 0.75rem; font-size: 0.875rem; color: #86868b;">
                    <strong style="color: #1d1d1f;">📊 Last Scan:</strong>
                    <span id="scan-total">-</span> images •
                    Cache: <span id="scan-cache">-</span> entries (<span id="scan-cache-size">-</span>) •
                    Duplicates: <span id="scan-exact-dupes">-</span> exact, <span id="scan-perceptual-dupes">-</span> perceptual •
                    MD5: <span id="scan-md5-mode">-</span>
                </div>
                
                <div id="processing-status" class="processing-status"></div>
            </div>

            <!-- Step 2: AI Inference/Embedding Tab -->
            <div id="step2-tab" class="control-panel" style="display: none;">
                <h2>🧠 Step 2: AI Inference/Embedding</h2>
                
                <div class="control-row">
                    <label style="font-size: 0.875rem; color: #86868b; font-weight: 600;">Similarity threshold:</label>
                    <input id="similarity-threshold" type="number" class="control-input" placeholder="Threshold" value="0.85" step="0.01" min="0" max="1" style="width: 100px;">
                    <span style="font-size: 0.75rem; color: #86868b;">(Higher = more strict, e.g., 0.99 = nearly identical)</span>
                    <button class="btn" onclick="estimateEmbeddings()" id="btn-estimate" style="background: #5856d6; color: white;">⏱️ Estimate Time</button>
                </div>
                
                <div class="control-row">
                    <label style="font-size: 0.875rem; color: #86868b; font-weight: 600;">Processing Mode:</label>
                    <div style="display: flex; gap: 1rem; margin-bottom: 0.5rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem;">
                            <input type="radio" name="inference-mode" value="auto">
                            Auto (try remote, fallback local)
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem;">
                            <input type="radio" name="inference-mode" value="remote" checked>
                            Remote (requires service)
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem;">
                            <input type="radio" name="inference-mode" value="local">
                            Local (on this Mac)
                        </label>
                    </div>
                    <button class="btn btn-keep" onclick="startEmbeddings()" id="btn-embeddings">🧠 Generate Embeddings</button>
                    <button class="btn" onclick="clearEmbeddings()" id="btn-clear-embeddings" style="background: #a2aaad; color: white;">🗑️ Clear Embeddings</button>
                </div>
                
                <div id="estimate-results" style="display: none; margin-top: 0.5rem; padding: 1rem; background: #e3f2fd; border-radius: 8px; border: 1px solid #1976d2;">
                    <div style="font-size: 0.875rem; color: #1565c0;"></div>
                </div>
                
                <div class="control-row">
                    <strong>Re-analyze with Feedback:</strong>
                    <button class="btn btn-not-similar" onclick="startEmbeddings()">🔄 Re-analyze with Feedback</button>
                </div>
                
                <div id="processing-status-embed" class="processing-status"></div>
            </div>

            <div id="groups-container"></div>
            
            <!-- Image modal -->
            <div id="imageModal" class="modal" onclick="closeModal()">
                <span class="modal-close" onclick="closeModal()">&times;</span>
                <button id="modalPrev" onclick="event.stopPropagation(); modalPrevImage();" 
                        style="position: absolute; left: 20px; top: 50%; transform: translateY(-50%); 
                               background: rgba(0,0,0,0.7); color: white; border: none; 
                               padding: 1rem 1.5rem; font-size: 2rem; cursor: pointer; border-radius: 8px;
                               font-weight: bold;">
                    ‹
                </button>
                <button id="modalNext" onclick="event.stopPropagation(); modalNextImage();" 
                        style="position: absolute; right: 20px; top: 50%; transform: translateY(-50%); 
                               background: rgba(0,0,0,0.7); color: white; border: none; 
                               padding: 1rem 1.5rem; font-size: 2rem; cursor: pointer; border-radius: 8px;
                               font-weight: bold;">
                    ›
                </button>
                <img class="modal-content" id="modalImage" onclick="event.stopPropagation()">
                <div class="modal-info" id="modalInfo"></div>
            </div>
        </div>

        <script>
            let groups = [];
            let aiGroupsOverlapExact = []; // AI groups that overlap with exact duplicates
            let aiGroupsOverlapPerceptual = []; // AI groups that overlap with perceptual duplicates
            let aiGroupsNewDiscoveries = []; // AI groups with no overlap
            let scanDuplicates = null;
            let scanDataMap = {}; // file_path -> scan data with size
            let selectedImages = new Map(); // groupId -> Set of selected indices (for AI groups)
            let selectedExactImages = new Map(); // groupIdx -> Set of image indices
            let selectedPerceptualImages = new Map(); // groupIdx -> Set of image indices
            let activeExactGroups = new Set(); // Set of group indices that are active (checked)
            let activePerceptualGroups = new Set(); // Set of group indices that are active (checked)
            let activeAIGroups = new Set(); // Set of AI group IDs that are active (checked)
            let collapsedSections = { 
                exact: false, 
                perceptual: false, 
                ai: false,
                overlaps_with_exact_matches: true,  // Collapsed by default since usually fewer items
                overlaps_with_perceptual_matches: true,  // Collapsed by default since usually fewer items
                new_ai_discoveries: false  // Expanded by default - most interesting
            }; // type -> collapsed state
            let statusCheckInterval = null;
            let currentTab = 'scan'; // 'scan' or 'ai'
            let modalCurrentGroup = null; // { type, groupIdx, files }
            let modalCurrentIndex = 0;
            
            function openModal(imagePath, fileName, groupType, groupIdx) {
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                const modalInfo = document.getElementById('modalInfo');
                
                // Get the files for this group
                let files = [];
                if (groupType === 'exact' && scanDuplicates?.exact_groups) {
                    files = scanDuplicates.exact_groups[groupIdx]?.files || [];
                } else if (groupType === 'perceptual' && scanDuplicates?.perceptual_groups) {
                    files = scanDuplicates.perceptual_groups[groupIdx]?.files || [];
                } else if (groupType === 'ai') {
                    const group = groups.find(g => g.group_id === groupIdx);
                    files = group ? group.files.map(f => f.path) : [];
                }
                
                // Sort files by date (oldest first) for exact/perceptual, keep as-is for AI
                let sortedFiles;
                if (groupType === 'ai') {
                    sortedFiles = files.map((path, idx) => ({
                        path,
                        originalIdx: idx,
                        date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                    }));
                } else {
                    sortedFiles = files.map((path, idx) => ({
                        path,
                        originalIdx: idx,
                        date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                    })).sort((a, b) => a.date.localeCompare(b.date));
                }
                
                // Find the current image index in the sorted array
                const currentIdx = sortedFiles.findIndex(f => f.path === imagePath);
                
                modalCurrentGroup = { 
                    type: groupType, 
                    groupIdx: groupIdx, 
                    files: sortedFiles.map(f => f.path),
                    originalIndices: sortedFiles.map(f => f.originalIdx)
                };
                modalCurrentIndex = currentIdx >= 0 ? currentIdx : 0;
                
                updateModalImage();
                modal.classList.add('active');
            }
            
            function updateModalImage() {
                if (!modalCurrentGroup || modalCurrentGroup.files.length === 0) return;
                
                const modalImg = document.getElementById('modalImage');
                const modalInfo = document.getElementById('modalInfo');
                const path = modalCurrentGroup.files[modalCurrentIndex];
                const fileName = path.split('/').pop();
                
                modalImg.src = '/api/image?path=' + encodeURIComponent(path);
                
                const item = scanDataMap[path];
                let info = fileName + '<br>';
                info += '<strong>Image ' + (modalCurrentIndex + 1) + ' of ' + modalCurrentGroup.files.length + '</strong>';
                if (item) {
                    const date = item.photo_date || item.created_at || 'Unknown';
                    const size = formatBytes(item.file_size || 0);
                    info += '<br>Date: ' + date + '<br>Size: ' + size;
                    if (item.width && item.height) {
                        info += '<br>Dimensions: ' + item.width + 'x' + item.height;
                    }
                }
                
                // Add selection controls for both exact and perceptual groups
                if (modalCurrentGroup.type === 'perceptual' || modalCurrentGroup.type === 'exact' || modalCurrentGroup.type === 'ai') {
                    const selectedMap = modalCurrentGroup.type === 'perceptual' ? selectedPerceptualImages : 
                                       modalCurrentGroup.type === 'exact' ? selectedExactImages : 
                                       selectedImages;
                    const selected = selectedMap.get(modalCurrentGroup.groupIdx) || new Set();
                    const originalIdx = modalCurrentGroup.originalIndices ? modalCurrentGroup.originalIndices[modalCurrentIndex] : modalCurrentIndex;
                    const isSelected = selected.has(originalIdx);
                    info += '<br><br>';
                    info += '<button onclick="event.stopPropagation(); toggleModalSelection()" style="padding: 0.5rem 1rem; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.875rem; ';
                    if (isSelected) {
                        info += 'background: #ff9500; color: white;">🗑️ Selected for Deletion - Click to Keep</button>';
                    } else {
                        info += 'background: #34c759; color: white;">✓ Will Keep - Click to Delete</button>';
                    }
                }
                
                modalInfo.innerHTML = info;
                
                // Update navigation button states
                const prevBtn = document.getElementById('modalPrev');
                const nextBtn = document.getElementById('modalNext');
                if (prevBtn) prevBtn.disabled = modalCurrentIndex === 0;
                if (nextBtn) nextBtn.disabled = modalCurrentIndex === modalCurrentGroup.files.length - 1;
            }
            
            function toggleModalSelection() {
                if (!modalCurrentGroup || (modalCurrentGroup.type !== 'perceptual' && modalCurrentGroup.type !== 'exact' && modalCurrentGroup.type !== 'ai')) return;
                
                // Toggle the selection using the original index
                const groupIdx = modalCurrentGroup.groupIdx;
                const imageIdx = modalCurrentGroup.originalIndices ? modalCurrentGroup.originalIndices[modalCurrentIndex] : modalCurrentIndex;
                
                if (modalCurrentGroup.type === 'perceptual') {
                    togglePerceptualImage(groupIdx, imageIdx);
                } else if (modalCurrentGroup.type === 'exact') {
                    toggleExactImage(groupIdx, imageIdx);
                } else if (modalCurrentGroup.type === 'ai') {
                    toggleAIImage(groupIdx, imageIdx);
                }
                
                // Just update the modal display, don't re-render entire page
                updateModalImage();
            }
            
            function modalPrevImage() {
                if (!modalCurrentGroup || modalCurrentIndex <= 0) return;
                modalCurrentIndex--;
                updateModalImage();
            }
            
            function modalNextImage() {
                if (!modalCurrentGroup || modalCurrentIndex >= modalCurrentGroup.files.length - 1) return;
                modalCurrentIndex++;
                updateModalImage();
            }
            
            function closeModal() {
                const modal = document.getElementById('imageModal');
                modal.classList.remove('active');
                modalCurrentGroup = null;
                modalCurrentIndex = 0;
                
                // Update the page to reflect any selection changes made in the modal
                render();
            }
            
            // Keyboard navigation
            document.addEventListener('keydown', (e) => {
                const modal = document.getElementById('imageModal');
                if (!modal.classList.contains('active')) return;
                
                if (e.key === 'ArrowLeft') {
                    modalPrevImage();
                } else if (e.key === 'ArrowRight') {
                    modalNextImage();
                } else if (e.key === 'Escape') {
                    closeModal();
                }
            });
            
            function togglePerceptualImage(groupIdx, imageIdx) {
                if (!selectedPerceptualImages.has(groupIdx)) {
                    selectedPerceptualImages.set(groupIdx, new Set());
                }
                const selected = selectedPerceptualImages.get(groupIdx);
                if (selected.has(imageIdx)) {
                    selected.delete(imageIdx);
                } else {
                    selected.add(imageIdx);
                }
                render();
            }
            
            function toggleExactImage(groupIdx, imageIdx) {
                if (!selectedExactImages.has(groupIdx)) {
                    selectedExactImages.set(groupIdx, new Set());
                }
                const selected = selectedExactImages.get(groupIdx);
                if (selected.has(imageIdx)) {
                    selected.delete(imageIdx);
                } else {
                    selected.add(imageIdx);
                }
                render();
            }
            
            function toggleGroupActive(type, groupIdx) {
                if (type === 'ai') {
                    // AI groups use groupId instead of index
                    const groupId = groupIdx;
                    const group = groups.find(g => g.group_id === groupId);
                    
                    if (activeAIGroups.has(groupId)) {
                        // Unchecking: Deactivate group and clear selection
                        activeAIGroups.delete(groupId);
                        selectedImages.delete(groupId);
                    } else {
                        // Checking: Activate group and restore previous selection or use default
                        activeAIGroups.add(groupId);
                        if (!selectedImages.has(groupId) && group) {
                            // No previous selection, apply default (all but first)
                            const selected = new Set(group.files.map((_, idx) => idx).filter(idx => idx !== 0));
                            selectedImages.set(groupId, selected);
                        }
                    }
                } else {
                    const activeGroups = type === 'exact' ? activeExactGroups : activePerceptualGroups;
                    const selectedImagesMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                    const groupsList = type === 'exact' ? scanDuplicates.exact_groups : scanDuplicates.perceptual_groups;
                    const group = groupsList[groupIdx];
                    
                    if (activeGroups.has(groupIdx)) {
                        // Unchecking: Deactivate group and clear selection (all keep)
                        activeGroups.delete(groupIdx);
                        selectedImagesMap.delete(groupIdx);
                    } else {
                        // Checking: Activate group and restore previous selection or use default
                        activeGroups.add(groupIdx);
                        if (!selectedImagesMap.has(groupIdx) && group) {
                            // No previous selection, apply default (all but oldest)
                            const oldestIdx = getOldestImageIndex(group.files);
                            const selected = new Set(group.files.map((_, idx) => idx).filter(idx => idx !== oldestIdx));
                            selectedImagesMap.set(groupIdx, selected);
                        }
                    }
                }
                render();
            }
            
            function getOldestImageIndex(files) {
                const filesWithDates = files.map((path, idx) => ({
                    idx,
                    date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                }));
                filesWithDates.sort((a, b) => a.date.localeCompare(b.date));
                return filesWithDates[0].idx;
            }
            
            function getSortedFilesWithInfo(files) {
                const filesWithInfo = files.map((path, idx) => ({
                    path,
                    originalIdx: idx,
                    date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                }));
                filesWithInfo.sort((a, b) => a.date.localeCompare(b.date));
                return filesWithInfo;
            }
            
            function selectAllPerceptualDefault() {
                if (!scanDuplicates?.perceptual_groups) return;
                
                activePerceptualGroups.clear();
                selectedPerceptualImages.clear();
                scanDuplicates.perceptual_groups.forEach((group, groupIdx) => {
                    const oldestIdx = getOldestImageIndex(group.files);
                    const selected = new Set(group.files.map((_, idx) => idx).filter(idx => idx !== oldestIdx));
                    if (selected.size > 0) {
                        selectedPerceptualImages.set(groupIdx, selected);
                        activePerceptualGroups.add(groupIdx);
                    }
                });
                render();
            }
            
            function deselectAllPerceptual() {
                selectedPerceptualImages.clear();
                activePerceptualGroups.clear();
                render();
            }
            
            function selectAllExactDefault() {
                if (!scanDuplicates?.exact_groups) return;
                
                activeExactGroups.clear();
                selectedExactImages.clear();
                scanDuplicates.exact_groups.forEach((group, groupIdx) => {
                    const oldestIdx = getOldestImageIndex(group.files);
                    const selected = new Set(group.files.map((_, idx) => idx).filter(idx => idx !== oldestIdx));
                    if (selected.size > 0) {
                        selectedExactImages.set(groupIdx, selected);
                        activeExactGroups.add(groupIdx);
                    }
                });
                render();
            }
            
            function deselectAllExact() {
                selectedExactImages.clear();
                activeExactGroups.clear();
                render();
            }
            
            function selectAllAIDefault() {
                if (!groups || groups.length === 0) return;
                
                activeAIGroups.clear();
                selectedImages.clear();
                groups.forEach(group => {
                    // For AI groups, select all but the first image
                    const selected = new Set(group.files.map((_, idx) => idx).filter(idx => idx !== 0));
                    if (selected.size > 0) {
                        selectedImages.set(group.group_id, selected);
                        activeAIGroups.add(group.group_id);
                    }
                });
                render();
            }
            
            function deselectAllAI() {
                selectedImages.clear();
                activeAIGroups.clear();
                render();
            }
            
            function toggleAIImage(groupId, imageIdx) {
                if (!selectedImages.has(groupId)) {
                    selectedImages.set(groupId, new Set());
                }
                const selected = selectedImages.get(groupId);
                if (selected.has(imageIdx)) {
                    selected.delete(imageIdx);
                } else {
                    selected.add(imageIdx);
                }
                render();
            }
            
            // Progress overlay utilities
            function showProgressOverlay(message) {
                // Remove existing overlay if any
                hideProgressOverlay();
                
                // Create translucent background
                const overlay = document.createElement('div');
                overlay.id = 'progress-overlay';
                overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center';
                
                // Create message box
                const messageBox = document.createElement('div');
                messageBox.id = 'progress-message';
                messageBox.style.cssText = 'background:rgba(0,0,0,0.9);color:white;padding:30px 50px;border-radius:12px;font-size:18px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.5)';
                messageBox.innerHTML = message;
                
                overlay.appendChild(messageBox);
                document.body.appendChild(overlay);
                return overlay;
            }
            
            function hideProgressOverlay() {
                const overlay = document.getElementById('progress-overlay');
                if (overlay) document.body.removeChild(overlay);
            }
            
            function updateProgressMessage(message) {
                const messageBox = document.getElementById('progress-message');
                if (messageBox) messageBox.innerHTML = message;
            }
            
            // Generic batch delete function for all group types
            async function batchDeleteSelected(config) {
                const filesToDelete = [];
                let totalSize = 0;
                
                // Collect files from active groups
                config.activeGroups.forEach(groupKey => {
                    const selected = config.selectedMap.get(groupKey);
                    const group = config.getGroup(groupKey);
                    if (!group || !selected) return;
                    
                    selected.forEach(imgIdx => {
                        const path = config.getPath(group, imgIdx);
                        filesToDelete.push(path);
                        totalSize += scanDataMap[path]?.file_size || 0;
                    });
                });
                
                if (filesToDelete.length === 0) return;
                
                if (!confirm(`Add ${filesToDelete.length} photo(s) to Duplicates album?\n\nPotential space to save: ${formatBytes(totalSize)}\n\nYou can then delete them manually from Photos.app\n(which properly syncs to iCloud)`)) {
                    return;
                }
                
                showProgressOverlay(`<div style="text-align:center">Adding photos to Duplicates album...<br><span id="progress-count">0/${filesToDelete.length}</span></div>`);
                
                try {
                    let addedCount = 0;
                    let failedCount = 0;
                    for (let i = 0; i < filesToDelete.length; i++) {
                        updateProgressMessage(`<div style="text-align:center">Adding photos to Duplicates album...<br><span id="progress-count">${i + 1}/${filesToDelete.length}</span></div>`);
                        const success = await addToDuplicatesAlbum(filesToDelete[i]);
                        if (success) addedCount++;
                        else failedCount++;
                    }
                    
                    hideProgressOverlay();
                    
                    if (failedCount > 0) {
                        alert(`⚠️ Added ${addedCount} photos, ${failedCount} failed\n\nCheck console for details.`);
                    } else {
                        const openAlbum = confirm(`✓ Added ${addedCount} photo(s) to duplicates\n\nPotential savings: ${formatBytes(totalSize)}\n\nClick OK to open and view the duplicates`);
                        if (openAlbum) {
                            await openDuplicatesAlbum();
                        }
                    }
                    config.selectedMap.clear();
                    config.activeGroups.clear();
                    loadGroups();
                } catch (error) {
                    hideProgressOverlay();
                    alert('Failed: ' + error.message);
                }
            }
            
            // Specific batch delete functions using the generic implementation
            async function batchDeleteAISelected() {
                await batchDeleteSelected({
                    activeGroups: activeAIGroups,
                    selectedMap: selectedImages,
                    getGroup: (groupId) => groups.find(g => g.group_id === groupId),
                    getPath: (group, imgIdx) => group.files[imgIdx].path
                });
            }
            
            async function batchDeletePerceptualSelected() {
                await batchDeleteSelected({
                    activeGroups: activePerceptualGroups,
                    selectedMap: selectedPerceptualImages,
                    getGroup: (groupIdx) => scanDuplicates.perceptual_groups[groupIdx],
                    getPath: (group, imgIdx) => group.files[imgIdx]
                });
            }
            
            async function batchDeleteExactSelected() {
                await batchDeleteSelected({
                    activeGroups: activeExactGroups,
                    selectedMap: selectedExactImages,
                    getGroup: (groupIdx) => scanDuplicates.exact_groups[groupIdx],
                    getPath: (group, imgIdx) => group.files[imgIdx]
                });
            }
            
            // Load scan info
            async function loadScanInfo() {
                try {
                    const response = await fetch('/api/scan-info');
                    const info = await response.json();
                    
                    if (info.scan_completed) {
                        document.getElementById('scan-info').style.display = 'block';
                        document.getElementById('scan-total').textContent = info.total_images;
                        document.getElementById('scan-cache').textContent = info.cache_entries;
                        document.getElementById('scan-cache-size').textContent = info.cache_size_mb + ' MB';
                        document.getElementById('scan-exact-dupes').textContent = info.exact_duplicate_groups;
                        document.getElementById('scan-perceptual-dupes').textContent = info.perceptual_duplicate_groups;
                        document.getElementById('scan-md5-mode').textContent = info.md5_mode || 'unknown';
                    }
                } catch (error) {
                    console.error('Failed to load scan info:', error);
                }
            }

            // Setup scan source selector
            document.addEventListener('DOMContentLoaded', async () => {
                loadScanInfo(); // Load scan info on page load
                
                // Initialize tab: show Step 2 if embeddings exist, otherwise Step 1
                try {
                    console.log('Initializing - checking for embeddings...');
                    const response = await fetch('/api/ai-groups-categorized');
                    if (response.ok) {
                        console.log('Embeddings found, loading all data...');
                        await loadGroups(); // Load all data first
                        console.log('Data loaded, switching to step2. scanDuplicates:', scanDuplicates);
                        switchTab('step2'); // Then switch to step 2, which will render AI results
                    } else {
                        console.log('No embeddings, loading scan data...');
                        await loadGroups(); // Load all data first (scan duplicates, etc.)
                        console.log('Data loaded, switching to step1. scanDuplicates:', scanDuplicates);
                        switchTab('step1'); // Then switch to step 1, which will render scanner results
                    }
                } catch (err) {
                    console.log('Error loading groups:', err);
                    await loadGroups(); // Load all data first
                    console.log('Data loaded, switching to step1. scanDuplicates:', scanDuplicates);
                    switchTab('step1'); // Then switch to step 1
                }
                
                const scanSource = document.getElementById('scan-source');
                const scanPath = document.getElementById('scan-path');
                const photosAccess = document.getElementById('photos-access');
                
                scanSource.addEventListener('change', (e) => {
                    if (e.target.value === 'directory') {
                        scanPath.style.display = 'block';
                        scanPath.required = true;
                        photosAccess.style.display = 'none';
                    } else {
                        scanPath.style.display = 'none';
                        scanPath.required = false;
                        photosAccess.style.display = 'block';
                    }
                });
            });

            async function startScan() {
                const source = document.getElementById('scan-source').value;
                const photos_access = document.getElementById('photos-access').value;
                const path = document.getElementById('scan-path').value;
                const limit = parseInt(document.getElementById('scan-limit').value) || 20;
                const use_cache = document.getElementById('use-cache').checked;
                const md5_mode = document.getElementById('md5-mode').value;
                
                if (source === 'directory' && !path) {
                    alert('Please enter a directory path');
                    return;
                }
                
                const btnScan = document.getElementById('btn-scan');
                btnScan.disabled = true;
                
                // Show immediate feedback with overlay
                showProgressOverlay('Initializing photo scan...');
                showStatus('running', 'Initializing photo scan...');
                
                try {
                    const response = await fetch('/api/scan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ source, photos_access, path, limit, use_cache, md5_mode })
                    });
                    
                    if (response.ok) {
                        startStatusPolling();
                    } else {
                        const error = await response.json();
                        hideProgressOverlay();
                        showStatus('error', error.detail || 'Scan failed');
                        btnScan.disabled = false;
                    }
                } catch (error) {
                    hideProgressOverlay();
                    showStatus('error', 'Failed to start scan: ' + error.message);
                    btnScan.disabled = false;
                }
            }

            function switchTab(tabName) {
                // Track which tab is active
                currentTab = (tabName === 'step1') ? 'scan' : 'ai';
                
                // Hide both tabs
                document.getElementById('step1-tab').style.display = 'none';
                document.getElementById('step2-tab').style.display = 'none';
                
                // Reset button styles (get the span element correctly)
                const step1Btn = document.getElementById('step1-btn');
                const step2Btn = document.getElementById('step2-btn');
                const step1Circle = step1Btn.querySelector('span');
                const step2Circle = step2Btn.querySelector('span');
                
                step1Btn.style.borderColor = '#d2d2d7';
                step1Btn.style.color = '#86868b';
                if (step1Circle) step1Circle.style.background = '#d2d2d7';
                
                step2Btn.style.borderColor = '#d2d2d7';
                step2Btn.style.color = '#86868b';
                if (step2Circle) step2Circle.style.background = '#d2d2d7';
                
                // Show selected tab and render appropriate results
                if (tabName === 'step1') {
                    document.getElementById('step1-tab').style.display = 'block';
                    step1Btn.style.borderColor = '#0071e3';
                    step1Btn.style.color = '#0071e3';
                    if (step1Circle) step1Circle.style.background = '#0071e3';
                    renderScannerResults();
                } else if (tabName === 'step2') {
                    document.getElementById('step2-tab').style.display = 'block';
                    step2Btn.style.borderColor = '#0071e3';
                    step2Btn.style.color = '#0071e3';
                    if (step2Circle) step2Circle.style.background = '#0071e3';
                    renderAIResults();
                }
            }

            function renderScannerResults() {
                // Render only scanner/scan duplicate results
                console.log('renderScannerResults called. scanDuplicates:', scanDuplicates);
                groups = [];
                updateStats(0.85);
                render();
            }

            function renderAIResults() {
                // Render only AI embedding results
                console.log('renderAIResults called. AI groups:', aiGroupsOverlapExact.length, aiGroupsOverlapPerceptual.length, aiGroupsNewDiscoveries.length);
                groups = [...aiGroupsOverlapExact, ...aiGroupsOverlapPerceptual, ...aiGroupsNewDiscoveries];
                updateStats(0.85);
                render();
            }

            async function clearScan() {
                if (!confirm('Clear scan results? This will remove all duplicate detection data.')) return;
                
                try {
                    const response = await fetch('/api/clear-scan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        alert('Error: ' + (error.detail || 'Failed to clear scan'));
                        return;
                    }
                    
                    // Clear the UI
                    document.getElementById('scan-info').style.display = 'none';
                    document.getElementById('groups-container').innerHTML = '<div class="loading">⏳ No scan results</div>';
                    alert('Scan cleared');
                } catch (err) {
                    console.error('Error clearing scan:', err);
                    alert('Error: ' + err.message);
                }
            }

            async function clearEmbeddings() {
                if (!confirm('Clear all embedding results and groups?')) return;
                
                try {
                    const response = await fetch('/api/clear-embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        alert('Error: ' + (error.detail || 'Failed to clear embeddings'));
                        return;
                    }
                    
                    // Reload the scan results to show them again
                    await loadGroups();
                    alert('Embeddings cleared');
                } catch (err) {
                    console.error('Error clearing embeddings:', err);
                    alert('Error: ' + err.message);
                }
            }

            async function startEmbeddings() {
                const threshold = parseFloat(document.getElementById('similarity-threshold').value) || 0.85;
                const inferenceMode = document.querySelector('input[name="inference-mode"]:checked').value || 'remote';
                
                const btnEmbeddings = document.getElementById('btn-embeddings');
                btnEmbeddings.disabled = true;
                
                // Show immediate feedback
                showStatus('running', `Initializing embedding generation (${inferenceMode} mode)...`);
                
                try {
                    const response = await fetch('/api/embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            similarity_threshold: threshold, 
                            estimate: false,
                            inference_mode: inferenceMode,
                            service_url: null  // Use backend default from environment
                        })
                    });
                    
                    if (response.ok) {
                        // Hide estimate results when starting actual generation
                        document.getElementById('estimate-results').style.display = 'none';
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

            async function estimateEmbeddings() {
                const threshold = parseFloat(document.getElementById('similarity-threshold').value) || 0.85;
                const inferenceMode = document.querySelector('input[name="inference-mode"]:checked').value || 'remote';
                const btnEstimate = document.getElementById('btn-estimate');
                const resultsDiv = document.getElementById('estimate-results');
                
                btnEstimate.disabled = true;
                btnEstimate.textContent = '⏱️ Estimating...';
                resultsDiv.style.display = 'none';
                
                // Show processing status
                showStatus('running', 'Analyzing sample images to estimate processing time...');
                
                try {
                    const response = await fetch('/api/embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            similarity_threshold: threshold, 
                            estimate: true,
                            estimate_sample: 30,
                            inference_mode: inferenceMode
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        const est = data.estimates;
                        const throughput = data.throughput;
                        
                        let html = `
                            <strong>📊 Estimation Results (based on ${data.sample_size} sample images):</strong><br>
                            <div style="margin-top: 0.5rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem;">
                                <div>
                                    <strong>Total Images:</strong> ${data.total_images.toLocaleString()}<br>
                                    <strong>Processing Speed:</strong> ${throughput.images_per_second} img/sec
                                </div>
                                <div>
                                    <strong>Estimated Time:</strong><br>
                                    • Embeddings: ~${est.embedding_minutes} min<br>
                                    • Grouping: ~${est.grouping_minutes} min<br>
                                    • <strong>Total: ~${est.total_minutes} min (${est.total_hours} hrs)</strong>
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">
                                ${data.note}
                            </div>
                        `;
                        
                        resultsDiv.innerHTML = html;
                        resultsDiv.style.display = 'block';
                        
                        showStatus('success', `Estimation complete: ~${est.total_minutes} minutes for ${data.total_images} images`);
                    } else {
                        showStatus('error', data.error || 'Estimation failed');
                    }
                    
                } catch (error) {
                    showStatus('error', 'Failed to estimate: ' + error.message);
                } finally {
                    btnEstimate.disabled = false;
                    btnEstimate.textContent = '⏱️ Estimate Time';
                    // Hide status after a delay if successful
                    if (document.getElementById('processing-status').classList.contains('success')) {
                        setTimeout(() => {
                            const statusDiv = document.getElementById('processing-status');
                            statusDiv.classList.remove('active');
                        }, 3000);
                    }
                }
            }

            function startStatusPolling() {
                if (statusCheckInterval) return;
                
                const startingStage = document.getElementById('processing-status')?.textContent?.includes('Scanning') ? 'scan' : 'embeddings';
                
                statusCheckInterval = setInterval(async () => {
                    try {
                        const response = await fetch('/api/status');
                        const status = await response.json();
                        
                        if (status.running) {
                            updateProgressMessage(`${status.stage}: ${status.message}`);
                            showStatus('running', `${status.stage}: ${status.message}`);
                        } else {
                            hideProgressOverlay();
                            showStatus('success', status.message);
                            clearInterval(statusCheckInterval);
                            statusCheckInterval = null;
                            
                            // Re-enable buttons
                            document.getElementById('btn-scan').disabled = false;
                            document.getElementById('btn-embeddings').disabled = false;
                            
                            // Reload groups and switch to appropriate tab
                            await loadGroups();
                            
                            // Auto-switch to the relevant tab
                            if (startingStage === 'scan' && scanDuplicates) {
                                currentTab = 'scan';
                            } else if (startingStage === 'embeddings' && groups && groups.length > 0) {
                                currentTab = 'ai';
                            }
                            render();
                        }
                    } catch (error) {
                        console.error('Status check failed:', error);
                    }
                }, 1000);
            }

            function showStatus(type, message) {
                const statusDiv = document.getElementById('processing-status');
                statusDiv.className = `processing-status active ${type}`;
                
                // Clear existing content
                statusDiv.innerHTML = '';
                
                // Add appropriate icon/spinner
                if (type === 'running') {
                    const spinner = document.createElement('div');
                    spinner.className = 'spinner';
                    statusDiv.appendChild(spinner);
                } else if (type === 'success') {
                    const icon = document.createElement('span');
                    icon.className = 'status-icon';
                    icon.textContent = '✅';
                    statusDiv.appendChild(icon);
                } else if (type === 'error') {
                    const icon = document.createElement('span');
                    icon.className = 'status-icon';
                    icon.textContent = '❌';
                    statusDiv.appendChild(icon);
                }
                
                // Add message
                const messageSpan = document.createElement('span');
                messageSpan.textContent = message;
                statusDiv.appendChild(messageSpan);
            }

            async function loadGroups() {
                try {
                    console.log('Loading groups...');
                    
                    // Load categorized AI groups
                    const response = await fetch('/api/ai-groups-categorized');
                    
                    if (!response.ok) {
                        // No embeddings yet - show empty state
                        console.log('No embeddings loaded yet');
                        aiGroupsOverlapExact = [];
                        aiGroupsOverlapPerceptual = [];
                        aiGroupsNewDiscoveries = [];
                        groups = [];
                    } else {
                        const data = await response.json();
                        
                        // Store categorized groups
                        aiGroupsOverlapExact = data.overlaps_exact || [];
                        aiGroupsOverlapPerceptual = data.overlaps_perceptual || [];
                        aiGroupsNewDiscoveries = data.new_discoveries || [];
                        
                        // Keep old 'groups' variable for compatibility with existing code
                        groups = [...aiGroupsOverlapExact, ...aiGroupsOverlapPerceptual, ...aiGroupsNewDiscoveries];
                        
                        console.log(`AI groups: ${data.counts.overlaps_exact} overlap exact, ${data.counts.overlaps_perceptual} overlap perceptual, ${data.counts.new_discoveries} new discoveries`);
                    }

                    // Load scan duplicates (best-effort)
                    try {
                        const dupResp = await fetch('/api/scan-duplicates');
                        if (dupResp.ok) {
                            scanDuplicates = await dupResp.json();
                            console.log('Loaded scan duplicates:', scanDuplicates);
                            
                            // Filter out perceptual duplicates that are already in exact matches
                            if (scanDuplicates.exact_groups && scanDuplicates.perceptual_groups) {
                                const exactFiles = new Set();
                                scanDuplicates.exact_groups.forEach(group => {
                                    group.files.forEach(file => exactFiles.add(file));
                                });
                                
                                // Filter perceptual groups to exclude exact matches
                                scanDuplicates.perceptual_groups = scanDuplicates.perceptual_groups
                                    .map(group => ({
                                        ...group,
                                        files: group.files.filter(file => !exactFiles.has(file))
                                    }))
                                    .filter(group => group.files.length > 1); // Keep only groups with 2+ files
                            }
                        } else {
                            console.log('No scan duplicates found (run scan first)');
                            scanDuplicates = null;
                        }
                    } catch (e) {
                        console.error('Error loading scan duplicates:', e);
                        scanDuplicates = null;
                    }
                    
                    // Load scan data for file sizes
                    try {
                        const scanResp = await fetch('/api/scan-data');
                        if (scanResp.ok) {
                            const scanArray = await scanResp.json();
                            scanDataMap = {};
                            scanArray.forEach(item => {
                                scanDataMap[item.file_path] = item;
                            });
                        }
                    } catch (e) {
                        console.error('Failed to load scan data:', e);
                    }

                    // Auto-select exact duplicates by default, but not perceptual
                    if (scanDuplicates?.exact_groups && scanDuplicates.exact_groups.length > 0) {
                        selectAllExactDefault();
                    }

                    console.log('About to call render()');
                    updateStats(0.85);
                    render();
                    console.log('Render completed');
                } catch (error) {
                    console.error('Failed to load groups:', error);
                    console.error('Error stack:', error.stack);
                    document.getElementById('groups-container').innerHTML = 
                        '<div class="loading">Failed to load groups: ' + error.message + '</div>';
                }
            }

            function updateStats(threshold) {
                const total = groups.length;
                const reviewed = groups.filter(g => g.reviewed).length;
                const feedbackCount = groups.filter(g => g.action === 'not_similar').length;

                const scanExact = scanDuplicates?.exact_groups?.length || 0;
                const scanPerceptual = scanDuplicates?.perceptual_groups?.length || 0;
                
                let statsText = '';
                if (scanExact > 0 || scanPerceptual > 0) {
                    statsText = `Scanner: ${scanExact} exact + ${scanPerceptual} perceptual groups`;
                }
                if (total > 0) {
                    if (statsText) statsText += ' • ';
                    statsText += `AI: ${total} groups (${reviewed} reviewed)`;
                }
                if (!statsText) {
                    statsText = 'No results yet - run scanner first';
                }
                
                document.getElementById('stats-total').textContent = statsText;
                document.getElementById('stats-threshold').textContent = 
                    threshold ? `Threshold: ${(threshold * 100).toFixed(0)}%` : '';

                if (feedbackCount > 0) {
                    document.getElementById('stats-feedback').textContent = `${feedbackCount} feedback items`;
                } else {
                    document.getElementById('stats-feedback').textContent = '';
                }
            }



            function toggleImageExpand(event) {
                const img = event.target;
                img.classList.toggle('expanded');
            }
            
            function formatBytes(bytes) {
                if (bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            function toggleSection(type) {
                collapsedSections[type] = !collapsedSections[type];
                render();
            }
            
            function calculateTotalSavings(groupType) {
                let totalFiles = 0;
                let totalBytes = 0;
                
                const groupsList = groupType === 'exact' ? 
                    scanDuplicates?.exact_groups : scanDuplicates?.perceptual_groups;
                    
                if (!groupsList) return { files: 0, bytes: 0 };
                
                groupsList.forEach((group, idx) => {
                    const files = group.files;
                    // Keep oldest, delete the rest
                    const filesWithDates = files.map(path => ({
                        path,
                        // Use photo_date (EXIF capture date) if available, fall back to created_at (file creation)
                        date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99',
                        size: scanDataMap[path]?.file_size || 0
                    }));
                    filesWithDates.sort((a, b) => a.date.localeCompare(b.date));
                    
                    // Sum up all files except the oldest
                    for (let i = 1; i < filesWithDates.length; i++) {
                        totalFiles++;
                        totalBytes += filesWithDates[i].size;
                    }
                });
                
                return { files: totalFiles, bytes: totalBytes };
            }
            
            function getGroupSize(files) {
                return files.reduce((sum, file) => {
                    const path = typeof file === 'string' ? file : file.path;
                    return sum + (scanDataMap[path]?.file_size || 0);
                }, 0);
            }

            function switchTab(tab) {
                currentTab = tab;
                render();
            }

            /**
             * Unified section rendering function for all duplicate group types
             * @param {Object} config - Section configuration
             * @param {string} config.sectionKey - Unique key for section (e.g., 'exact', 'perceptual', 'new_ai_discoveries')
             * @param {string} config.title - Section title
             * @param {string} config.emoji - Emoji for section header
             * @param {string} config.color - Background color/gradient for header
             * @param {Array} config.groups - Array of groups to render
             * @param {string} config.type - Group type: 'exact' | 'perceptual' | 'ai'
             * @param {string} config.tipMessage - Tip message HTML
             * @param {string} config.tipBackground - Background color for tip box
             * @param {string} config.tipTextColor - Text color for tip box
             * @param {string} config.buttonColor - Color for action buttons
             * @param {string} config.savingsDisplayId - ID for savings display span
             * @param {Function} config.selectAllFn - Function name to call for "Select All"
             * @param {Function} config.deselectAllFn - Function name to call for "Clear"
             * @param {Function} config.deleteFn - Function name to call for "Delete Selected"
             * @param {Function} config.renderGroupFn - Function to render individual groups
             */
            function renderUnifiedSection(config) {
                const {
                    sectionKey,
                    title,
                    emoji,
                    color,
                    groups,
                    type,
                    tipMessage,
                    tipBackground,
                    tipTextColor,
                    buttonColor,
                    savingsDisplayId,
                    selectAllFn,
                    deselectAllFn,
                    deleteFn,
                    renderGroupFn
                } = config;

                if (!groups || groups.length === 0) return '';

                const isCollapsed = collapsedSections[sectionKey] || false;

                // Calculate potential savings
                let totalPotentialFiles = 0;
                let totalPotentialSize = 0;

                if (type === 'ai') {
                    // For AI groups, count all but first in each group
                    groups.forEach(group => {
                        if (group.files && group.files.length > 1) {
                            for (let i = 1; i < group.files.length; i++) {
                                totalPotentialFiles++;
                                const filePath = typeof group.files[i] === 'string' ? group.files[i] : group.files[i].path;
                                totalPotentialSize += scanDataMap[filePath]?.file_size || 0;
                            }
                        }
                    });
                } else {
                    // For exact/perceptual groups, keep oldest, count rest
                    groups.forEach(group => {
                        const filesWithDates = group.files.map(path => ({
                            path,
                            date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99',
                            size: scanDataMap[path]?.file_size || 0
                        }));
                        filesWithDates.sort((a, b) => a.date.localeCompare(b.date));
                        // Sum all except oldest
                        for (let i = 1; i < filesWithDates.length; i++) {
                            totalPotentialFiles++;
                            totalPotentialSize += filesWithDates[i].size;
                        }
                    });
                }

                const fileLabel = type === 'ai' ? 'potentially similar files' : 'files can be deleted';

                let html = `
                    <div class="type-section" style="margin-bottom: 1.5rem;">
                        <div class="type-header ${isCollapsed ? 'collapsed' : ''}" style="background: ${color};" onclick="toggleSection('${sectionKey}')">
                            <div style="flex: 1;">
                                <div style="display: flex; align-items: center; gap: 0.75rem;">
                                    <span style="font-size: 1.5rem;">${isCollapsed ? '▶' : '▼'}</span>
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.25rem;">${emoji} ${title}</h2>
                                        <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.9;">
                                            ${groups.length} groups • ${totalPotentialFiles} ${fileLabel}
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.125rem; font-weight: 600;">💾 ${formatBytes(totalPotentialSize)}</div>
                                <div style="font-size: 0.75rem; opacity: 0.9;">Potential savings</div>
                            </div>
                        </div>
                        <div class="type-content ${isCollapsed ? 'collapsed' : ''}">
                            <div style="background: white; padding: 1rem; border-radius: 0 0 12px 12px;">
                                <div style="margin-bottom: 1rem; padding: 0.75rem; background: ${tipBackground}; border-radius: 6px; color: ${tipTextColor}; font-size: 0.875rem;">
                                    ${tipMessage}
                                </div>
                                <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; align-items: center;">
                                    <button class="btn btn-keep" onclick="event.stopPropagation(); ${selectAllFn}()" style="background: ${buttonColor}; color: white;">
                                        ✓ Select All
                                    </button>
                                    <button class="btn btn-keep" onclick="event.stopPropagation(); ${deselectAllFn}()" style="background: #8e8e93; color: white;">
                                        ✗ Clear
                                    </button>
                                    <button class="btn btn-delete" onclick="event.stopPropagation(); ${deleteFn}()" style="background: ${buttonColor}; color: white; margin-left: auto;">
                                        � Add Selected to Duplicates Album
                                    </button>
                                    <span id="${savingsDisplayId}" style="font-size: 0.875rem; color: ${buttonColor}; font-weight: 600; white-space: nowrap;"></span>
                                </div>
                                <div class="groups-grid">
                `;

                // Render groups using provided function
                groups.forEach((group, idx) => {
                    html += renderGroupFn(group, idx);
                });

                html += '</div></div></div></div>';

                return html;
            }

            function render() {
                const container = document.getElementById('groups-container');
                
                const hasScanData = scanDuplicates && 
                    ((scanDuplicates.exact_groups && scanDuplicates.exact_groups.length > 0) ||
                     (scanDuplicates.perceptual_groups && scanDuplicates.perceptual_groups.length > 0));
                const hasAIData = groups && groups.length > 0;
                
                // Don't render old-style tabs - we use workflow step buttons now
                let html = '';
                
                // Scanner tab content
                if (currentTab === 'scan') {
                    if (!hasScanData) {
                        html += `
                            <div class="group-card">
                                <h2>📊 Scanner Results</h2>
                                <p>Run the scanner first to find exact and perceptual duplicates.</p>
                            </div>
                        `;
                    } else {
                        // Exact matches section
                        if (scanDuplicates.exact_groups && scanDuplicates.exact_groups.length > 0) {
                            html += renderUnifiedSection({
                                sectionKey: 'exact',
                                title: 'Exact Duplicates (MD5 Hash)',
                                emoji: '🟢',
                                color: 'linear-gradient(135deg, #34c759 0%, #30d158 100%)',
                                groups: scanDuplicates.exact_groups,
                                type: 'exact',
                                tipMessage: '💡 <strong>Tip:</strong> Check groups to include in batch delete. Click thumbnails to toggle individual images. Default keeps oldest photo.',
                                tipBackground: '#e8f5e9',
                                tipTextColor: '#2e7d32',
                                buttonColor: '#34c759',
                                savingsDisplayId: 'exact-savings-display',
                                selectAllFn: 'selectAllExactDefault',
                                deselectAllFn: 'deselectAllExact',
                                deleteFn: 'batchDeleteExactSelected',
                                renderGroupFn: (group, idx) => renderCompactGroup(group, idx, 'exact', '#34c759')
                            });
                        }
                        
                        // Perceptual matches section
                        if (scanDuplicates.perceptual_groups && scanDuplicates.perceptual_groups.length > 0) {
                            html += renderUnifiedSection({
                                sectionKey: 'perceptual',
                                title: 'Perceptual Duplicates (Review Required)',
                                emoji: '🟠',
                                color: 'linear-gradient(135deg, #ff9500 0%, #ff9f0a 100%)',
                                groups: scanDuplicates.perceptual_groups,
                                type: 'perceptual',
                                tipMessage: '💡 <strong>Tip:</strong> Check groups to include in batch delete. Click thumbnails to toggle individual images. Default keeps oldest photo.',
                                tipBackground: '#fff3cd',
                                tipTextColor: '#856404',
                                buttonColor: '#ff9500',
                                savingsDisplayId: 'perceptual-savings-display',
                                selectAllFn: 'selectAllPerceptualDefault',
                                deselectAllFn: 'deselectAllPerceptual',
                                deleteFn: 'batchDeletePerceptualSelected',
                                renderGroupFn: (group, idx) => renderPerceptualGroup(group, idx, '#ff9500')
                            });
                        }
                        
                        // Info about what goes to AI
                        html += `
                            <div class="group-card" style="background: #f0f8ff; border-left: 4px solid #007aff;">
                                <h3 style="margin: 0 0 0.5rem 0; color: #007aff;">ℹ️ About AI Embeddings</h3>
                                <p style="margin: 0; color: #1d1d1f;">
                                    When you run "Generate Embeddings", the AI analyzes <strong>all scanned photos</strong> 
                                    (not just duplicates) to find similar images based on visual content. This can discover 
                                    similar photos that aren't exact or perceptual matches - like different angles of the same scene, 
                                    similar compositions, or semantically related images.
                                </p>
                            </div>
                        `;
                    }
                }
                }
                
                // AI tab content
                if (currentTab === 'ai') {
                    html += '<div class="tab-content active">';
                    
                    if (!hasAIData) {
                        html += `
                            <div class="group-card">
                                <h2>🧠 AI Similar Groups</h2>
                                <p>Run "Generate Embeddings" after scanning to find AI-detected similar images.</p>
                            </div>
                        `;
                    } else {
                        // Render three AI sections using renderUnifiedSection
                        
                        html += renderUnifiedSection({
                            sectionKey: 'new_ai_discoveries',
                            title: 'New AI Discoveries',
                            emoji: '🔴',
                            color: 'linear-gradient(135deg, #c41e3a 0%, #8b0000 100%)',
                            groups: aiGroupsNewDiscoveries,
                            type: 'ai',
                            tipMessage: '💡 These AI groups contain images <strong>not found in exact or perceptual duplicates</strong>. These are truly new discoveries based on visual content analysis - like different angles of the same scene or semantically similar images.',
                            tipBackground: '#f5f5f7',
                            tipTextColor: '#1d1d1f',
                            buttonColor: '#c41e3a',
                            savingsDisplayId: 'ai-new-savings-display',
                            selectAllFn: 'selectAllAIDefault',
                            deselectAllFn: 'deselectAllAI',
                            deleteFn: 'batchDeleteAISelected',
                            renderGroupFn: (group) => renderAIGroup(group, group.group_id)
                        });

                        html += renderUnifiedSection({
                            sectionKey: 'overlaps_with_exact_matches',
                            title: 'Overlaps with Exact Matches',
                            emoji: '🟢',
                            color: 'linear-gradient(135deg, #34c759 0%, #30d158 100%)',
                            groups: aiGroupsOverlapExact,
                            type: 'ai',
                            tipMessage: '💡 These AI groups contain images that are also in <strong>Exact Duplicate</strong> groups. The AI found visual similarity among images that are already exact byte-for-byte matches.',
                            tipBackground: '#f5f5f7',
                            tipTextColor: '#1d1d1f',
                            buttonColor: '#34c759',
                            savingsDisplayId: 'ai-exact-savings-display',
                            selectAllFn: 'selectAllAIDefault',
                            deselectAllFn: 'deselectAllAI',
                            deleteFn: 'batchDeleteAISelected',
                            renderGroupFn: (group) => renderAIGroup(group, group.group_id)
                        });
                        
                        html += renderUnifiedSection({
                            sectionKey: 'overlaps_with_perceptual_matches',
                            title: 'Overlaps with Perceptual Matches',
                            emoji: '🟠',
                            color: 'linear-gradient(135deg, #ff9500 0%, #ff9f0a 100%)',
                            groups: aiGroupsOverlapPerceptual,
                            type: 'ai',
                            tipMessage: '💡 These AI groups contain images that are also in <strong>Perceptual Duplicate</strong> groups. The AI found visual similarity among images that were already flagged as perceptually similar.',
                            tipBackground: '#f5f5f7',
                            tipTextColor: '#1d1d1f',
                            buttonColor: '#ff9500',
                            savingsDisplayId: 'ai-perceptual-savings-display',
                            selectAllFn: 'selectAllAIDefault',
                            deselectAllFn: 'deselectAllAI',
                            deleteFn: 'batchDeleteAISelected',
                            renderGroupFn: (group) => renderAIGroup(group, group.group_id)
                        });
                        

                        
                        // Summary card
                        html += `
                            <div class="group-card" style="background: #f0f8ff; border-left: 4px solid #007aff;">
                                <h3 style="margin: 0 0 0.5rem 0; color: #007aff;">📊 AI Analysis Summary</h3>
                                <p style="margin: 0 0 0.5rem 0; color: #1d1d1f;">
                                    <strong>${aiGroupsOverlapExact.length}</strong> groups overlap with exact matches<br>
                                    <strong>${aiGroupsOverlapPerceptual.length}</strong> groups overlap with perceptual matches<br>
                                    <strong>${aiGroupsNewDiscoveries.length}</strong> groups are new discoveries
                                </p>
                                <p style="margin: 0; font-size: 0.875rem; color: #86868b;">
                                    Total: ${groups.length} AI-detected similarity groups
                                </p>
                            </div>
                        `;
                    }
                    
                    html += '</div>';
                }
                
                // Fallback if no data
                if (!hasScanData && !hasAIData) {
                    html = `
                        <div class="group-card">
                            <h2>👋 Welcome!</h2>
                            <p><strong>Step 1:</strong> Click "Start Scan" to scan for photos</p>
                            <p><strong>Step 2:</strong> Click "Generate Embeddings" to find AI-similar images</p>
                            <p>Results will appear here organized by type.</p>
                        </div>
                    `;
                }
                
                container.innerHTML = html;
                
                // Update disk savings displays
                updateSavingsDisplays();
            }
            
            function updateSavingsDisplays() {
                // Calculate exact duplicates savings
                let exactFiles = 0;
                let exactBytes = 0;
                activeExactGroups.forEach(groupIdx => {
                    const selected = selectedExactImages.get(groupIdx);
                    const group = scanDuplicates?.exact_groups?.[groupIdx];
                    if (group && selected) {
                        selected.forEach(imgIdx => {
                            exactFiles++;
                            exactBytes += scanDataMap[group.files[imgIdx]]?.file_size || 0;
                        });
                    }
                });
                
                // Calculate perceptual duplicates savings
                let perceptualFiles = 0;
                let perceptualBytes = 0;
                activePerceptualGroups.forEach(groupIdx => {
                    const selected = selectedPerceptualImages.get(groupIdx);
                    const group = scanDuplicates?.perceptual_groups?.[groupIdx];
                    if (group && selected) {
                        selected.forEach(imgIdx => {
                            perceptualFiles++;
                            perceptualBytes += scanDataMap[group.files[imgIdx]]?.file_size || 0;
                        });
                    }
                });
                
                // Calculate AI groups savings (all groups combined)
                let aiFiles = 0;
                let aiBytes = 0;
                activeAIGroups.forEach(groupId => {
                    const selected = selectedImages.get(groupId);
                    const group = groups.find(g => g.group_id === groupId);
                    if (group && selected) {
                        selected.forEach(imgIdx => {
                            aiFiles++;
                            aiBytes += scanDataMap[group.files[imgIdx].path]?.file_size || 0;
                        });
                    }
                });
                
                // Update displays
                const exactDisplay = document.getElementById('exact-savings-display');
                if (exactDisplay) {
                    exactDisplay.textContent = `(${exactFiles} files • ${formatBytes(exactBytes)})`;
                }
                
                const perceptualDisplay = document.getElementById('perceptual-savings-display');
                if (perceptualDisplay) {
                    perceptualDisplay.textContent = `(${perceptualFiles} files • ${formatBytes(perceptualBytes)})`;
                }
                
                // Update all three AI section displays with the same values
                const aiNewDisplay = document.getElementById('ai-new-savings-display');
                if (aiNewDisplay) {
                    aiNewDisplay.textContent = `(${aiFiles} files • ${formatBytes(aiBytes)})`;
                }
                
                const aiExactDisplay = document.getElementById('ai-exact-savings-display');
                if (aiExactDisplay) {
                    aiExactDisplay.textContent = `(${aiFiles} files • ${formatBytes(aiBytes)})`;
                }
                
                const aiPerceptualDisplay = document.getElementById('ai-perceptual-savings-display');
                if (aiPerceptualDisplay) {
                    aiPerceptualDisplay.textContent = `(${aiFiles} files • ${formatBytes(aiBytes)})`;
                }
            }

            function getSortedFilesWithInfo(files) {
                const filesWithInfo = files.map((path, idx) => ({
                    path,
                    originalIdx: idx,
                    date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                }));
                filesWithInfo.sort((a, b) => a.date.localeCompare(b.date));
                return filesWithInfo;
            }
            
            // Unified rendering function for both exact and perceptual duplicate groups
            // ========== UNIFIED SECTION RENDERING SYSTEM ==========
            
            // Unified function to render any duplicate group (exact, perceptual, or AI)
            function renderUnifiedGroup(config) {
                const { group, index, groupId, type, themeColor, showSimilarity } = config;
                const files = type === 'ai' ? group.files.map(f => f.path) : group.files;
                const id = groupId !== undefined ? groupId : index;
                
                // Get type-specific state
                let isActive, selected, toggleImageFunc;
                if (type === 'ai') {
                    isActive = activeAIGroups.has(id);
                    selected = isActive ? (selectedImages.get(id) || new Set()) : new Set();
                    toggleImageFunc = `toggleAIImage(${id}, INDEX)`;
                } else if (type === 'exact') {
                    isActive = activeExactGroups.has(index);
                    selected = isActive ? (selectedExactImages.get(index) || new Set()) : new Set();
                    toggleImageFunc = `toggleExactImage(${index}, INDEX)`;
                } else { // perceptual
                    isActive = activePerceptualGroups.has(index);
                    selected = isActive ? (selectedPerceptualImages.get(index) || new Set()) : new Set();
                    toggleImageFunc = `togglePerceptualImage(${index}, INDEX)`;
                }
                
                const totalSize = getGroupSize(files);
                
                // For exact/perceptual, sort by date and mark oldest
                let filesWithInfo;
                let oldestIdx = null;
                if (type !== 'ai') {
                    filesWithInfo = files.map((path, idx) => ({
                        path,
                        originalIdx: idx,
                        date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                    }));
                    filesWithInfo.sort((a, b) => a.date.localeCompare(b.date));
                    oldestIdx = filesWithInfo[0].originalIdx;
                }
                
                // Calculate thumbnail size
                const thumbSize = files.length === 2 ? '80px' : '70px';
                
                // Calculate selection stats
                const selectedSize = Array.from(selected).reduce((sum, idx) => 
                    sum + (scanDataMap[files[idx]]?.file_size || 0), 0);
                const hasSelection = selected.size > 0;
                
                let html = '<div class="mini-group-card">';
                
                // Group header with checkbox
                html += '<div class="mini-group-header" style="margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">';
                html += '<input type="checkbox" ' + (isActive ? 'checked' : '') + ' ';
                html += 'onclick="toggleGroupActive(\\'' + type + '\\', ' + id + ')" ';
                html += 'style="width: 18px; height: 18px; cursor: pointer;" ';
                html += 'title="' + (isActive ? 'Uncheck to exclude from batch delete' : 'Check to include in batch delete') + '">';
                html += '<span style="font-weight: 600; color: #1d1d1f;">#' + (index + 1) + '</span>';
                html += '<span class="similarity-badge" style="background: ' + themeColor + '; font-size: 0.625rem; padding: 0.25rem 0.5rem;">';
                html += files.length;
                html += '</span>';
                if (showSimilarity && group.avg_similarity) {
                    html += '<span style="font-size: 0.625rem; color: #86868b; margin-left: auto;">' + (group.avg_similarity * 100).toFixed(1) + '% sim</span>';
                }
                html += '</div>';
                
                // Thumbnail grid
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(' + thumbSize + ', 1fr)); gap: 0.5rem; margin-bottom: 0.75rem;">';
                
                const itemsToRender = type === 'ai' ? group.files : filesWithInfo;
                itemsToRender.forEach((item, arrayIdx) => {
                    const path = type === 'ai' ? item.path : item.path;
                    const fileIdx = type === 'ai' ? arrayIdx : item.originalIdx;
                    const isSelected = selected.has(fileIdx);
                    const isOldest = oldestIdx !== null && fileIdx === oldestIdx;
                    const willBeKept = !isSelected;
                    
                    html += '<div style="position: relative;">';
                    
                    // Image with border
                    html += '<img src="/api/image?path=' + encodeURIComponent(path) + '" ';
                    html += 'alt="thumb" ';
                    html += 'style="width: 100%; height: ' + thumbSize + '; object-fit: cover; border-radius: 4px; border: ' + (isSelected ? '2px solid #ff9500' : '2px solid #34c759') + '; position: relative; z-index: 1;">';
                    
                    // Clickable overlay
                    html += '<div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; cursor: pointer; z-index: 2;" ';
                    html += 'onclick="' + toggleImageFunc.replace('INDEX', fileIdx) + ';" ';
                    html += 'title="' + (willBeKept ? 'Click to select for deletion' : 'Click to keep') + '"></div>';
                    
                    // Badge
                    if (willBeKept) {
                        html += '<div style="position: absolute; top: 2px; left: 2px; background: #34c759; color: white; font-size: 0.625rem; padding: 2px 4px; border-radius: 3px; z-index: 3; font-weight: 600; pointer-events: none;">';
                        html += isOldest ? 'KEEP ★' : 'KEEP';
                        html += '</div>';
                    } else {
                        html += '<div style="position: absolute; top: 2px; left: 2px; background: #ff9500; color: white; font-size: 0.625rem; padding: 2px 4px; border-radius: 3px; z-index: 3; font-weight: 600; pointer-events: none;">';
                        html += 'DEL ✗';
                        html += '</div>';
                    }
                    
                    // View button
                    html += '<div style="position: absolute; bottom: 2px; right: 2px; background: rgba(0,0,0,0.7); color: white; font-size: 0.625rem; padding: 2px 6px; border-radius: 3px; z-index: 3; cursor: pointer;" ';
                    html += 'onclick="event.stopPropagation(); openModal(\\'' + path.replace(/'/g, "\\\\'") + '\\', \\'' + path.split('/').pop().replace(/'/g, "\\\\'") + '\\', \\'' + type + '\\', ' + id + ');" ';
                    html += 'title="View full size">🔍</div>';
                    
                    html += '</div>';
                });
                html += '</div>';
                
                // Stats footer
                html += '<div class="group-stats" style="margin-top: 0.5rem; display: flex; gap: 0.5rem; font-size: 0.75rem; color: #86868b;">';
                html += '<span>📦 ' + formatBytes(totalSize) + '</span>';
                if (hasSelection) {
                    html += '<span style="color: #ff3b30; font-weight: 600;">💾 Save ' + formatBytes(selectedSize) + '</span>';
                }
                html += '</div>';
                
                html += '</div>';
                return html;
            }
            
            // Legacy wrapper functions for backward compatibility
            function renderDuplicateGroup(group, index, type, themeColor) {
                return renderUnifiedGroup({ 
                    group, 
                    index, 
                    type, 
                    themeColor, 
                    showSimilarity: false 
                });
            }
            
            function renderPerceptualGroup(group, index, color) {
                return renderUnifiedGroup({ 
                    group, 
                    index, 
                    type: 'perceptual', 
                    themeColor: color, 
                    showSimilarity: false 
                });
            }
            
            function renderCompactGroup(group, index, type, color) {
                return renderUnifiedGroup({ 
                    group, 
                    index, 
                    type, 
                    themeColor: color, 
                    showSimilarity: false 
                });
            }
            
            function renderAIGroup(group, index) {
                return renderUnifiedGroup({ 
                    group, 
                    index, 
                    groupId: group.group_id,
                    type: 'ai', 
                    themeColor: '#007aff', 
                    showSimilarity: true 
                });
            }
            
            
            async function addToDuplicatesAlbum(filePath) {
                try {
                    const response = await fetch('/api/add-to-duplicates', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ file_path: filePath })
                    });
                    return response.ok;
                } catch (error) {
                    console.error('Failed to add to Duplicates album:', error);
                    return false;
                }
            }
            
            async function openDuplicatesAlbum() {
                try {
                    await fetch('/api/open-duplicates-album', { method: 'POST' });
                } catch (error) {
                    console.error('Failed to open album:', error);
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
                
                render();
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
                        
                        updateStats(0.85);
                        render();
                        
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
                        updateStats(0.85);
                        render();
                        
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
    if SERVER_MODE == "demo":
        return DEMO_PROCESSING_STATUS
    return PROCESSING_STATUS


@app.get("/api/scan-info")
async def get_scan_info():
    """Get information about the completed scan."""
    if SERVER_MODE == "demo":
        scan_results_path = PROJECT_ROOT / "scan_results_demo.json"
        scan_duplicates_path = PROJECT_ROOT / "scan_duplicates_demo.json"
        cache_path = PROJECT_ROOT / ".cache_demo" / "scan_cache.json"
    else:
        scan_results_path = PROJECT_ROOT / "scan_results.json"
        scan_duplicates_path = PROJECT_ROOT / "scan_duplicates.json"
        cache_path = PROJECT_ROOT / ".cache" / "scan_cache.json"
    
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
    if SERVER_MODE == "demo":
        report_path = PROJECT_ROOT / "scan_duplicates_demo.json"
    else:
        report_path = PROJECT_ROOT / "scan_duplicates.json"
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
    if SERVER_MODE == "demo":
        scan_path = PROJECT_ROOT / "scan_results_demo.json"
    else:
        scan_path = PROJECT_ROOT / "scan_for_embeddings.json"
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
    if SERVER_MODE == "demo":
        groups_file = PROJECT_ROOT / "embeddings_demo" / "similar_groups.json"
        duplicates_file = PROJECT_ROOT / "scan_duplicates_demo.json"
    else:
        groups_file = PROJECT_ROOT / "embeddings" / "similar_groups.json"
        duplicates_file = PROJECT_ROOT / "scan_duplicates.json"
    
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
    if PROCESSING_STATUS["running"]:
        raise HTTPException(status_code=409, detail="Already processing")
    
    background_tasks.add_task(run_scan, request)
    return {"success": True, "message": "Scan started"}


@app.post("/api/clear-embeddings")
async def clear_embeddings():
    """Clear all embeddings and groups."""
    global DEMO_SIMILAR_GROUPS, DEMO_EMBEDDING_STORE, SIMILAR_GROUPS, EMBEDDING_STORE
    
    DEMO_SIMILAR_GROUPS = []
    DEMO_EMBEDDING_STORE = None
    SIMILAR_GROUPS = []
    EMBEDDING_STORE = None
    
    # Delete embedding files if they exist
    embeddings_dir = PROJECT_ROOT / ("embeddings_demo" if SERVER_MODE == "demo" else "embeddings")
    for file in ["similar_groups.json", "similar_pairs.json", "embeddings.npy", "metadata.json"]:
        file_path = embeddings_dir / file
        if file_path.exists():
            file_path.unlink()
    
    return {"success": True, "message": "Embeddings cleared"}


@app.post("/api/clear-scan")
async def clear_scan():
    """Clear all scan results."""
    global DEMO_SCAN_DATA, SIMILAR_GROUPS, EMBEDDING_STORE, DEMO_SIMILAR_GROUPS, DEMO_EMBEDDING_STORE, SCAN_DATA
    
    # Clear scan data
    SCAN_DATA = {}
    DEMO_SCAN_DATA = {}
    
    # Also clear embeddings since they depend on scan results
    SIMILAR_GROUPS = []
    DEMO_SIMILAR_GROUPS = []
    EMBEDDING_STORE = None
    DEMO_EMBEDDING_STORE = None
    
    # Delete scan files if they exist
    for file in ["scan_for_embeddings.json", "scan_results.json", "scan_duplicates.json"]:
        file_path = PROJECT_ROOT / file
        if file_path.exists():
            file_path.unlink()
    
    for file in ["scan_results_demo.json", "scan_duplicates_demo.json"]:
        file_path = PROJECT_ROOT / file
        if file_path.exists():
            file_path.unlink()
    
    # Also clear embedding files
    for mode_dir in ["embeddings", "embeddings_demo"]:
        embeddings_dir = PROJECT_ROOT / mode_dir
        for file in ["similar_groups.json", "similar_pairs.json", "embeddings.npy", "metadata.json"]:
            file_path = embeddings_dir / file
            if file_path.exists():
                file_path.unlink()
    
    return {"success": True, "message": "Scan cleared"}


@app.post("/api/embeddings")
async def generate_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Generate embeddings from scan results."""
    status = DEMO_PROCESSING_STATUS if SERVER_MODE == "demo" else PROCESSING_STATUS
    if status["running"]:
        raise HTTPException(status_code=409, detail="Already processing")
    
    if SERVER_MODE == "demo":
        scan_file = PROJECT_ROOT / "scan_results_demo.json"
    else:
        scan_file = PROJECT_ROOT / "scan_for_embeddings.json"
    
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
    global PROCESSING_STATUS
    PROCESSING_STATUS["running"] = True
    PROCESSING_STATUS["stage"] = "Scanning"
    PROCESSING_STATUS["message"] = "Initializing photo scanner..."
    
    try:
        cmd = [
            "python", "-m", "src.scanner.main",
            "--output", "scan_for_embeddings.json",
            "--duplicates-output", "scan_duplicates.json",
            "--cache-file", ".cache/scan_cache.json",
            "--md5-mode", request.md5_mode,
        ]
        
        if request.source == "photos-library":
            cmd.extend(["--photos-library"])
            if request.photos_access == "applescript":
                cmd.extend(["--use-applescript", "--keep-export"])
                PROCESSING_STATUS["message"] = "Scanning Photos Library (using AppleScript export)..."
            else:
                PROCESSING_STATUS["message"] = "Scanning Photos Library (direct access)..."
        else:
            # Resolve relative paths from project root
            scan_path = Path(request.path)
            if not scan_path.is_absolute():
                scan_path = PROJECT_ROOT / scan_path
            cmd.append(str(scan_path))
            PROCESSING_STATUS["message"] = f"Scanning directory: {scan_path.name}..."

        if not request.use_cache:
            cmd.append("--no-cache")
        
        if request.limit:
            cmd.extend(["--limit", str(request.limit)])
            PROCESSING_STATUS["message"] += f" (limit: {request.limit} images)"
        
        logger.info(f"Running scan: {' '.join(cmd)}")
        
        PROCESSING_STATUS["message"] = "Scanning in progress... This may take a few moments."
        
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
            stderr_text = stderr.decode()
            # Write full stderr to log file
            write_stderr_to_log(stderr_text, "scan")
            # Show clean error in UI
            error_msg = clean_stderr_message(stderr_text)
            PROCESSING_STATUS["message"] = f"Scan failed: {error_msg}"
            logger.error(f"Scan failed with return code {process.returncode}")
    
    except Exception as e:
        PROCESSING_STATUS["message"] = f"Scan error: {str(e)}"
        logger.error(f"Scan error: {e}")
    
    finally:
        PROCESSING_STATUS["running"] = False
        PROCESSING_STATUS["stage"] = ""


async def estimate_embeddings(request: EmbeddingRequest):
    """Estimate time and resources for embedding generation."""
    if SERVER_MODE == "demo":
        scan_file = PROJECT_ROOT / "scan_results_demo.json"
    else:
        scan_file = PROJECT_ROOT / "scan_for_embeddings.json"
    
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
        
        logger.info(f"Estimating embedding generation for {total_images} images using {sample_size} samples ({request.inference_mode} mode)")
        
        # Import here to avoid startup overhead
        import time
        import random
        from PIL import Image
        from ..embedding.embedder import ImageEmbedder, estimate_embedding_throughput
        
        # Sample random images
        sample_indices = random.sample(range(total_images), sample_size)
        sample_items = [scan_data[i] for i in sample_indices]
        
        # Load sample images (keep as PIL Image objects in memory)
        sample_images = []
        load_failures = 0
        for item in sample_items:
            try:
                img = Image.open(item["file_path"])
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                # Load into memory immediately
                sample_images.append(img.copy())
            except Exception as e:
                logger.warning(f"Failed to load {item['file_path']}: {e}")
                load_failures += 1
        
        if len(sample_images) == 0:
            return {
                "success": False,
                "error": "Failed to load any sample images"
            }
        
        # Initialize embedder for estimation (all modes use local for timing reference)
        logger.info("Initializing embedding model...")
        model_load_start = time.time()
        embedder = ImageEmbedder(model_name="ViT-B-32", pretrained="openai")
        model_load_time = time.time() - model_load_start
        
        # Estimate throughput using local embedder
        logger.info(f"Processing {len(sample_images)} sample images...")
        stats = estimate_embedding_throughput(embedder, sample_images, batch_size=32)
        images_per_second = stats["images_per_second"]
        total_test_time = stats["total_time"]
        
        # Adjust estimates based on inference mode
        mode_note = ""
        if request.inference_mode == "remote":
            mode_note = " (remote service may differ)"
        
        # Calculate estimates
        if images_per_second > 0:
            estimated_seconds = total_images / images_per_second
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
            "inference_mode": request.inference_mode,
            "mode_note": mode_note,
            "throughput": {
                "images_per_second": round(images_per_second, 2),
                "sample_time": round(total_test_time, 2),
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
    if SERVER_MODE == "demo":
        global DEMO_PROCESSING_STATUS, DEMO_SIMILAR_GROUPS, DEMO_EMBEDDING_STORE, DEMO_THRESHOLD
        status = DEMO_PROCESSING_STATUS
        scan_file = "scan_results_demo.json"
        output_dir = "embeddings_demo"
    else:
        global PROCESSING_STATUS, SIMILAR_GROUPS, EMBEDDING_STORE, CURRENT_THRESHOLD
        status = PROCESSING_STATUS
        scan_file = "scan_for_embeddings.json"
        output_dir = "embeddings"
    
    status["running"] = True
    status["stage"] = "Generating Embeddings"
    status["message"] = f"Generating embeddings ({request.inference_mode} mode)... This may take several minutes."
    
    try:
        # Use main_v2.py for remote/auto support
        cmd = [
            "python", "-m", "src.embedding.main_v2",
            scan_file,
            "--output", output_dir,
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
            groups_file = PROJECT_ROOT / output_dir / "similar_groups.json"
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
            
            # Reload embeddings
            embeddings_dir = PROJECT_ROOT / output_dir
            if SERVER_MODE == "demo":
                DEMO_EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
            else:
                EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
            
            # Update current threshold
            if SERVER_MODE == "demo":
                DEMO_THRESHOLD = request.similarity_threshold
            else:
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
    scan_path = PROJECT_ROOT / "scan_for_embeddings.json"
    duplicates_path = PROJECT_ROOT / "scan_duplicates.json"
    
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
