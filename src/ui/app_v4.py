"""FastAPI application for duplicate photo review with tab-based workflow UI (v4).

Separates Scanner and AI Inference/Embedding into distinct workflow tabs.
- Scanner Tab: Step 1 controls + scan results (exact & perceptual duplicates)
- AI Tab: Step 2 controls + AI results + feedback section
"""

import logging
from pathlib import Path
from typing import List, Optional
import json
import shutil
import subprocess
import asyncio
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from ..grouping import FeedbackLearner
from ..embedding.storage import EmbeddingStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Photo Duplicate Reviewer - Tab Workflow",
    description="Review and manage duplicate photo groups with workflow tabs (Scanner → AI)",
    version="0.4.0",
)

# Server mode - which dataset is active
SERVER_MODE: str = "main"  # "main" or "demo"

# Inference service URL (configurable via environment variable)
INFERENCE_SERVICE_URL: str = os.getenv("INFERENCE_SERVICE_URL", "http://127.0.0.1:8001")

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
    """Serve the main review interface with tab-based workflow."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Photo Duplicate Reviewer - Workflow Tabs</title>
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
                margin-bottom: 0.5rem;
            }
            .header .stats {
                font-size: 0.875rem;
                color: #86868b;
            }
            .container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            /* Tab Navigation */
            .workflow-tabs {
                display: flex;
                gap: 0.5rem;
                margin-bottom: 2rem;
                border-bottom: 2px solid #d2d2d7;
            }
            .workflow-tab {
                padding: 1rem 1.5rem;
                border: none;
                background: transparent;
                cursor: pointer;
                font-size: 1.125rem;
                font-weight: 500;
                color: #86868b;
                border-bottom: 3px solid transparent;
                transition: all 0.2s;
            }
            .workflow-tab:hover {
                color: #1d1d1f;
            }
            .workflow-tab.active {
                color: #007aff;
                border-bottom-color: #007aff;
            }
            .workflow-tab:disabled {
                opacity: 0.4;
                cursor: not-allowed;
            }
            .workflow-tab .badge {
                display: inline-block;
                background: #e5e5ea;
                color: #1d1d1f;
                padding: 0.25rem 0.5rem;
                border-radius: 10px;
                font-size: 0.75rem;
                margin-left: 0.5rem;
                font-weight: 600;
            }
            .workflow-tab.active .badge {
                background: #007aff;
                color: white;
            }
            
            /* Tab Content */
            .tab-panel {
                display: none;
            }
            .tab-panel.active {
                display: block;
            }
            
            /* Control Panels */
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
            
            /* Status Messages */
            .processing-status {
                padding: 1rem;
                background: #f5f5f7;
                border-radius: 8px;
                margin-top: 1rem;
                display: none;
                font-weight: 500;
                align-items: center;
                gap: 0.75rem;
            }
            .processing-status.active {
                display: flex;
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
            
            /* Buttons */
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
            .btn-primary {
                background: #007aff;
                color: white;
            }
            .btn-success {
                background: #34c759;
                color: white;
            }
            .btn-warning {
                background: #ff9500;
                color: white;
            }
            .btn-danger {
                background: #ff3b30;
                color: white;
            }
            .btn-secondary {
                background: #a2aaad;
                color: white;
            }
            .btn-special {
                background: #5856d6;
                color: white;
            }
            
            /* Info Boxes */
            .info-box {
                background: white;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                border-left: 4px solid #007aff;
            }
            .info-box.success {
                border-left-color: #34c759;
                background: #f0fdf4;
            }
            .info-box.warning {
                border-left-color: #ff9500;
                background: #fffbf0;
            }
            .info-box h3 {
                margin: 0 0 0.5rem 0;
                color: #1d1d1f;
            }
            .info-box p {
                margin: 0;
                color: #86868b;
                font-size: 0.875rem;
            }
            
            /* Results Grid */
            .results-container {
                margin-top: 2rem;
            }
            .section-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 12px 12px 0 0;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0;
            }
            .section-header.exact {
                background: linear-gradient(135deg, #34c759 0%, #30d158 100%);
            }
            .section-header.perceptual {
                background: linear-gradient(135deg, #ff9500 0%, #ffb347 100%);
            }
            .section-header.ai {
                background: linear-gradient(135deg, #007aff 0%, #5ac8fa 100%);
            }
            .section-content {
                background: white;
                padding: 1rem;
                border-radius: 0 0 12px 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                margin-bottom: 1.5rem;
                max-height: 10000px;
                overflow: hidden;
                transition: max-height 0.3s ease;
            }
            .section-content.collapsed {
                max-height: 0;
                padding: 0;
                margin-bottom: 1rem;
            }
            
            /* Groups Grid */
            .groups-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .group-card {
                background: white;
                border: 2px solid #e5e5ea;
                border-radius: 8px;
                padding: 1rem;
                transition: all 0.2s;
            }
            .group-card:hover {
                border-color: #007aff;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .group-card.selected {
                border-color: #007aff;
                background: #f0f8ff;
            }
            
            /* Image Grid */
            .images-preview {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
                gap: 0.5rem;
                margin: 0.75rem 0;
            }
            .image-thumb {
                width: 100%;
                aspect-ratio: 1;
                object-fit: cover;
                border-radius: 4px;
                border: 2px solid #e5e5ea;
                cursor: pointer;
                transition: all 0.2s;
            }
            .image-thumb:hover {
                border-color: #007aff;
            }
            
            /* Modal */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.95);
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
                cursor: pointer;
            }
            
            .empty-state {
                text-align: center;
                padding: 3rem;
                color: #86868b;
            }
            .empty-state h3 {
                color: #1d1d1f;
                margin-bottom: 0.5rem;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📸 Photo Duplicate Reviewer - Workflow Edition</h1>
            <div class="stats">
                <span id="stats-total">Loading...</span>
                <span id="stats-feedback" style="margin-left: 1rem;"></span>
            </div>
        </div>

        <div class="container">
            <!-- Workflow Tabs -->
            <div class="workflow-tabs">
                <button class="workflow-tab active" onclick="switchTab('scanner')" id="tab-scanner">
                    📱 Scanner
                    <span class="badge" id="scanner-count">Step 1</span>
                </button>
                <button class="workflow-tab" onclick="switchTab('ai')" id="tab-ai" disabled>
                    🧠 AI Inference/Embedding
                    <span class="badge" id="ai-count">Step 2</span>
                </button>
            </div>

            <!-- SCANNER TAB -->
            <div class="tab-panel active" id="scanner-tab">
                <div class="control-panel">
                    <h2>🔍 Step 1: Scan for Photos</h2>
                    <p style="color: #86868b; margin-bottom: 1rem; font-size: 0.875rem;">
                        First, scan your photo library to find exact and perceptual duplicates using MD5 hashing and image fingerprinting.
                    </p>
                    
                    <div class="control-row">
                        <label style="font-weight: 600; color: #1d1d1f;">Source:</label>
                        <select id="scan-source" class="control-input">
                            <option value="photos-library">Photos Library</option>
                            <option value="directory">Directory</option>
                        </select>
                        
                        <select id="photos-access" class="control-input">
                            <option value="applescript">AppleScript (no Full Disk Access)</option>
                            <option value="originals">Direct Access (requires Full Disk Access)</option>
                        </select>
                        
                        <input id="scan-path" type="text" class="control-input" placeholder="Directory path" style="display: none; flex: 1;">
                    </div>
                    
                    <div class="control-row">
                        <label style="font-weight: 600; color: #1d1d1f;">Options:</label>
                        <input id="scan-limit" type="number" class="control-input" placeholder="Limit (default: 20)" value="20" style="width: 150px;" title="Max photos to scan (0 = all)">
                        
                        <label style="font-size: 0.875rem; color: #86868b; display: flex; align-items: center; gap: 0.5rem;">
                            <input id="use-cache" type="checkbox" checked>
                            Use cache
                        </label>
                        
                        <select id="md5-mode" class="control-input" title="MD5 strategy for exact matches">
                            <option value="on-demand">MD5: on-demand (fast)</option>
                            <option value="always">MD5: always (thorough)</option>
                            <option value="never">MD5: never (skip exact)</option>
                        </select>
                        
                        <button class="btn btn-success" onclick="startScan()" id="btn-scan">▶️ Start Scan</button>
                    </div>
                    
                    <div id="scan-info" style="display: none; background: #f5f5f7; padding: 0.75rem 1rem; border-radius: 6px; margin: 1rem 0; font-size: 0.875rem; color: #86868b;">
                        <strong style="color: #1d1d1f;">📊 Last Scan Results:</strong>
                        <span id="scan-total">-</span> images •
                        Cache: <span id="scan-cache">-</span> entries (<span id="scan-cache-size">-</span>) •
                        Exact: <span id="scan-exact-dupes">-</span> groups •
                        Perceptual: <span id="scan-perceptual-dupes">-</span> groups
                    </div>
                    
                    <div id="processing-status-scan" class="processing-status"></div>
                </div>

                <!-- Scanner Results -->
                <div id="scanner-results" class="results-container">
                    <div class="empty-state">
                        <h3>📊 No scanner results yet</h3>
                        <p>Click "Start Scan" above to find exact and perceptual duplicates.</p>
                    </div>
                </div>
            </div>

            <!-- AI TAB -->
            <div class="tab-panel" id="ai-tab">
                <div class="control-panel">
                    <h2>🧠 Step 2: Generate AI Embeddings</h2>
                    <p style="color: #86868b; margin-bottom: 1rem; font-size: 0.875rem;">
                        Analyze all scanned photos with AI to find visually similar images. The AI discovers images that aren't exact or perceptual matches but share visual characteristics.
                    </p>
                    
                    <div class="control-row">
                        <label style="font-weight: 600; color: #1d1d1f;">Inference Mode:</label>
                        <div style="display: flex; gap: 1rem;">
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
                    </div>
                    
                    <div class="control-row">
                        <label style="font-weight: 600; color: #1d1d1f;">Similarity Threshold:</label>
                        <input id="similarity-threshold" type="number" class="control-input" value="0.85" step="0.01" min="0" max="1" style="width: 100px;">
                        <span style="font-size: 0.75rem; color: #86868b; max-width: 300px;">(Higher = more strict; 0.99 = nearly identical, 0.50 = loosely similar)</span>
                        
                        <button class="btn btn-special" onclick="estimateEmbeddings()" id="btn-estimate">⏱️ Estimate Time</button>
                        <button class="btn btn-success" onclick="startEmbeddings()" id="btn-embeddings">🧠 Generate Embeddings</button>
                    </div>
                    
                    <div id="estimate-results" style="display: none; margin-top: 1rem; padding: 1rem; background: #e3f2fd; border-radius: 8px; border: 1px solid #1976d2; color: #1565c0; font-size: 0.875rem;"></div>
                    
                    <div id="processing-status-ai" class="processing-status"></div>
                </div>

                <!-- After Giving Feedback Section -->
                <div class="control-panel" style="background: #f0fdf4; border-left: 4px solid #34c759;">
                    <h2>📝 After Giving Feedback</h2>
                    <p style="color: #86868b; margin-bottom: 1rem; font-size: 0.875rem;">
                        Once you've reviewed groups and marked images for deletion, re-run embeddings to let the AI learn from your feedback.
                    </p>
                    <button class="btn btn-success" onclick="startEmbeddings()">🔄 Re-analyze with Feedback</button>
                </div>

                <!-- AI Results -->
                <div id="ai-results" class="results-container">
                    <div class="empty-state">
                        <h3>🧠 No AI results yet</h3>
                        <p>Run "Generate Embeddings" after scanning to find AI-detected similar groups.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Image Modal -->
        <div id="imageModal" class="modal" onclick="closeModal()">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <img class="modal-content" id="modalImage" onclick="event.stopPropagation()">
        </div>

        <script>
            let scanDuplicates = null;
            let groups = [];
            let scanDataMap = {};
            let currentTab = 'scanner';
            
            // Tab switching
            function switchTab(tab) {
                currentTab = tab;
                
                // Update tab buttons
                document.querySelectorAll('.workflow-tab').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.getElementById('tab-' + tab).classList.add('active');
                
                // Update tab panels
                document.querySelectorAll('.tab-panel').forEach(panel => {
                    panel.classList.remove('active');
                });
                document.getElementById(tab + '-tab').classList.add('active');
            }
            
            // Load scan info on page load
            document.addEventListener('DOMContentLoaded', () => {
                loadScanInfo();
                
                // Setup scan source selector
                const scanSource = document.getElementById('scan-source');
                const scanPath = document.getElementById('scan-path');
                const photosAccess = document.getElementById('photos-access');
                
                scanSource.addEventListener('change', (e) => {
                    if (e.target.value === 'directory') {
                        scanPath.style.display = 'block';
                        photosAccess.style.display = 'none';
                    } else {
                        scanPath.style.display = 'none';
                        photosAccess.style.display = 'block';
                    }
                });
            });
            
            // Load and display scan info
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
                        
                        // Load and display results
                        loadScanResults();
                    }
                } catch (error) {
                    console.error('Failed to load scan info:', error);
                }
            }
            
            // Load scan duplicates and display results
            async function loadScanResults() {
                try {
                    const response = await fetch('/api/scan-duplicates');
                    scanDuplicates = await response.json();
                    
                    // Load scan data for file sizes
                    const dataResponse = await fetch('/api/scan-data');
                    const scanArray = await dataResponse.json();
                    scanDataMap = {};
                    scanArray.forEach(item => {
                        scanDataMap[item.file_path] = item;
                    });
                    
                    // Update tab badge
                    const totalGroups = (scanDuplicates.exact_groups?.length || 0) + 
                                       (scanDuplicates.perceptual_groups?.length || 0);
                    document.getElementById('scanner-count').textContent = totalGroups + ' groups';
                    
                    // Enable AI tab
                    document.getElementById('tab-ai').disabled = false;
                    
                    renderScannerResults();
                } catch (error) {
                    console.error('Failed to load scan results:', error);
                }
            }
            
            // Render scanner results
            function renderScannerResults() {
                const container = document.getElementById('scanner-results');
                let html = '';
                
                if (!scanDuplicates || (!scanDuplicates.exact_groups && !scanDuplicates.perceptual_groups)) {
                    container.innerHTML = '<div class="empty-state"><h3>📊 No duplicates found</h3><p>The scanner didn\'t find any exact or perceptual duplicates in your library.</p></div>';
                    return;
                }
                
                // Exact duplicates section
                if (scanDuplicates.exact_groups && scanDuplicates.exact_groups.length > 0) {
                    html += renderDuplicateSection('exact', scanDuplicates.exact_groups, 
                        '🟢 Exact Duplicates (MD5)', '#34c759');
                }
                
                // Perceptual duplicates section
                if (scanDuplicates.perceptual_groups && scanDuplicates.perceptual_groups.length > 0) {
                    html += renderDuplicateSection('perceptual', scanDuplicates.perceptual_groups,
                        '🟠 Perceptual Duplicates (dHash)', '#ff9500');
                }
                
                container.innerHTML = html;
            }
            
            // Render a duplicate section
            function renderDuplicateSection(type, groups, title, color) {
                let html = `
                    <div>
                        <div class="section-header ${type}" onclick="toggleSection(this)">
                            <div>
                                <h3 style="margin: 0; font-size: 1.125rem;">${title}</h3>
                                <p style="margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.875rem;">
                                    ${groups.length} groups
                                </p>
                            </div>
                        </div>
                        <div class="section-content">
                            <div style="margin-bottom: 1rem; padding: 0.75rem; background: #f5f5f7; border-radius: 6px; font-size: 0.875rem; color: #86868b;">
                                💡 <strong>Tip:</strong> Click on images to view details. Scanner found these using ${type === 'exact' ? 'MD5 file hashing (byte-for-byte matches)' : 'perceptual hashing (visual similarity)'}.
                            </div>
                            <div class="groups-grid">
                `;
                
                groups.forEach((group, idx) => {
                    const files = group.files || [];
                    html += `
                        <div class="group-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong>#${idx + 1}</strong>
                                <span style="background: ${color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                                    ${files.length} files
                                </span>
                            </div>
                            <div class="images-preview">
                    `;
                    
                    files.slice(0, 6).forEach((filePath, imgIdx) => {
                        const encodedPath = encodeURIComponent(filePath);
                        html += `
                            <img class="image-thumb" 
                                 src="/api/image?path=${encodedPath}"
                                 data-path="${filePath.replace(/"/g, '&quot;')}"
                                 onclick="openModal(this.dataset.path)"
                                 alt="thumb ${imgIdx}">
                        `;
                    });
                    
                    if (files.length > 6) {
                        html += `
                            <div style="display: flex; align-items: center; justify-content: center; background: #f5f5f7; border-radius: 4px; font-size: 0.75rem; color: #86868b; font-weight: 600;">
                                +${files.length - 6}
                            </div>
                        `;
                    }
                    
                    html += `
                            </div>
                            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #86868b;">
                                Size: ${formatBytes(group.files.reduce((sum, f) => sum + (scanDataMap[f]?.file_size || 0), 0))}
                            </div>
                        </div>
                    `;
                });
                
                html += `
                            </div>
                        </div>
                    </div>
                `;
                
                return html;
            }
            
            // Toggle section visibility
            function toggleSection(headerElement) {
                const content = headerElement.nextElementSibling;
                content.classList.toggle('collapsed');
            }
            
            // Modal functions
            function openModal(imagePath) {
                const modal = document.getElementById('imageModal');
                const img = document.getElementById('modalImage');
                img.src = '/api/image?path=' + encodeURIComponent(imagePath);
                modal.classList.add('active');
            }
            
            function closeModal() {
                const modal = document.getElementById('imageModal');
                modal.classList.remove('active');
            }
            
            // Format bytes
            function formatBytes(bytes) {
                if (bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            // Show status message
            function showStatus(tabType, type, message) {
                const statusElement = document.getElementById('processing-status-' + tabType);
                statusElement.className = 'processing-status active ' + type;
                statusElement.innerHTML = '';
                
                if (type === 'running') {
                    const spinner = document.createElement('div');
                    spinner.className = 'spinner';
                    statusElement.appendChild(spinner);
                }
                
                const messageSpan = document.createElement('span');
                messageSpan.textContent = message;
                statusElement.appendChild(messageSpan);
            }
            
            // Start scan
            async function startScan() {
                const source = document.getElementById('scan-source').value;
                const photosAccess = document.getElementById('photos-access').value;
                const path = document.getElementById('scan-path').value;
                const limit = parseInt(document.getElementById('scan-limit').value) || 20;
                const useCache = document.getElementById('use-cache').checked;
                const md5Mode = document.getElementById('md5-mode').value;
                
                if (source === 'directory' && !path) {
                    alert('Please enter a directory path');
                    return;
                }
                
                document.getElementById('btn-scan').disabled = true;
                showStatus('scan', 'running', 'Initializing scan...');
                
                try {
                    const response = await fetch('/api/scan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            source, photos_access: photosAccess, path, limit, use_cache: useCache, md5_mode: md5Mode
                        })
                    });
                    
                    if (response.ok) {
                        pollStatus('scan');
                    } else {
                        const error = await response.json();
                        showStatus('scan', 'error', error.detail || 'Scan failed');
                        document.getElementById('btn-scan').disabled = false;
                    }
                } catch (error) {
                    showStatus('scan', 'error', 'Failed to start scan: ' + error.message);
                    document.getElementById('btn-scan').disabled = false;
                }
            }
            
            // Poll status
            async function pollStatus(tabType) {
                const pollInterval = setInterval(async () => {
                    try {
                        const response = await fetch('/api/status');
                        const status = await response.json();
                        
                        if (status.running) {
                            showStatus(tabType, 'running', status.message);
                        } else {
                            clearInterval(pollInterval);
                            showStatus(tabType, 'success', status.message);
                            document.getElementById('btn-scan').disabled = false;
                            document.getElementById('btn-embeddings').disabled = false;
                            
                            // Reload results
                            if (tabType === 'scan') {
                                loadScanResults();
                            } else {
                                loadAIResults();
                            }
                            
                            // Hide status after delay
                            setTimeout(() => {
                                document.getElementById('processing-status-' + tabType).classList.remove('active');
                            }, 3000);
                        }
                    } catch (error) {
                        console.error('Status check failed:', error);
                    }
                }, 1000);
            }
            
            // Start embeddings
            async function startEmbeddings() {
                const threshold = parseFloat(document.getElementById('similarity-threshold').value) || 0.85;
                const mode = document.querySelector('input[name="inference-mode"]:checked').value || 'remote';
                
                document.getElementById('btn-embeddings').disabled = true;
                showStatus('ai', 'running', 'Initializing embedding generation...');
                
                try {
                    const response = await fetch('/api/embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            similarity_threshold: threshold,
                            estimate: false,
                            inference_mode: mode
                        })
                    });
                    
                    if (response.ok) {
                        document.getElementById('estimate-results').style.display = 'none';
                        pollStatus('ai');
                    } else {
                        const error = await response.json();
                        showStatus('ai', 'error', error.detail || 'Failed to start');
                        document.getElementById('btn-embeddings').disabled = false;
                    }
                } catch (error) {
                    showStatus('ai', 'error', 'Failed: ' + error.message);
                    document.getElementById('btn-embeddings').disabled = false;
                }
            }
            
            // Estimate embeddings
            async function estimateEmbeddings() {
                const threshold = parseFloat(document.getElementById('similarity-threshold').value) || 0.85;
                const btnEstimate = document.getElementById('btn-estimate');
                const resultsDiv = document.getElementById('estimate-results');
                
                btnEstimate.disabled = true;
                btnEstimate.textContent = '⏱️ Estimating...';
                
                try {
                    const response = await fetch('/api/embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            similarity_threshold: threshold,
                            estimate: true,
                            estimate_sample: 30
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        const est = data.estimates;
                        resultsDiv.innerHTML = `
                            <strong>📊 Estimation Results (${data.sample_size} sample images):</strong><br>
                            <div style="margin-top: 0.5rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <strong>Total Images:</strong> ${data.total_images.toLocaleString()}<br>
                                    <strong>Speed:</strong> ${data.throughput.images_per_second} img/sec
                                </div>
                                <div>
                                    <strong>Estimated Time:</strong><br>
                                    • Embeddings: ~${est.embedding_minutes} min<br>
                                    • Grouping: ~${est.grouping_minutes} min<br>
                                    • <strong>Total: ~${est.total_minutes} min</strong>
                                </div>
                            </div>
                        `;
                        resultsDiv.style.display = 'block';
                    }
                } catch (error) {
                    resultsDiv.innerHTML = '<strong>Error:</strong> ' + error.message;
                    resultsDiv.style.display = 'block';
                } finally {
                    btnEstimate.disabled = false;
                    btnEstimate.textContent = '⏱️ Estimate Time';
                }
            }
            
            // Load AI results
            async function loadAIResults() {
                try {
                    const response = await fetch('/api/ai-groups-categorized');
                    const data = await response.json();
                    groups = data.overlaps_exact || [];
                    groups = groups.concat(data.overlaps_perceptual || []);
                    groups = groups.concat(data.new_discoveries || []);
                    
                    // Update tab badge
                    document.getElementById('ai-count').textContent = groups.length + ' groups';
                    
                    renderAIResults(data);
                } catch (error) {
                    console.error('Failed to load AI results:', error);
                    document.getElementById('ai-results').innerHTML = 
                        '<div class="empty-state"><h3>Error loading results</h3><p>' + error.message + '</p></div>';
                }
            }
            
            // Render AI results
            function renderAIResults(data) {
                const container = document.getElementById('ai-results');
                let html = '';
                
                if (groups.length === 0) {
                    container.innerHTML = '<div class="empty-state"><h3>🧠 No AI results yet</h3><p>Run "Generate Embeddings" to analyze your photos.</p></div>';
                    return;
                }
                
                // Summary info
                html += `
                    <div class="info-box">
                        <h3>🧠 AI Analysis Summary</h3>
                        <p>
                            Found <strong>${groups.length}</strong> AI-detected similarity groups:
                            ${data.counts.overlaps_exact} overlap with exact matches •
                            ${data.counts.overlaps_perceptual} overlap with perceptual matches •
                            ${data.counts.new_discoveries} new discoveries
                        </p>
                    </div>
                `;
                
                // New discoveries section
                if (data.new_discoveries && data.new_discoveries.length > 0) {
                    html += renderAISection('new', data.new_discoveries, 
                        '🔴 New Discoveries', 'AI found these images without scanner help', '#c41e3a');
                }
                
                // Overlaps with exact section
                if (data.overlaps_exact && data.overlaps_exact.length > 0) {
                    html += renderAISection('exact', data.overlaps_exact,
                        '🟢 Overlaps with Exact Matches', 'AI agrees with MD5 hashing', '#34c759');
                }
                
                // Overlaps with perceptual section
                if (data.overlaps_perceptual && data.overlaps_perceptual.length > 0) {
                    html += renderAISection('perceptual', data.overlaps_perceptual,
                        '🟠 Overlaps with Perceptual Matches', 'AI agrees with perceptual analysis', '#ff9500');
                }
                
                container.innerHTML = html;
            }
            
            // Render AI section
            function renderAISection(sectionType, groups, title, description, color) {
                let html = `
                    <div>
                        <div class="section-header ai" onclick="toggleSection(this)" style="background: linear-gradient(135deg, ${color} 0%, ${color}99 100%);">
                            <div>
                                <h3 style="margin: 0; font-size: 1.125rem;">${title}</h3>
                                <p style="margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.875rem;">
                                    ${groups.length} groups • ${description}
                                </p>
                            </div>
                        </div>
                        <div class="section-content">
                            <div style="margin-bottom: 1rem; padding: 0.75rem; background: #f5f5f7; border-radius: 6px; font-size: 0.875rem; color: #86868b;">
                                💡 <strong>Tip:</strong> Click on images to preview. These are ${sectionType === 'new' ? 'truly new discoveries' : 'validated by scanner'}.
                            </div>
                            <div class="groups-grid">
                `;
                
                groups.forEach((group, idx) => {
                    const files = group.files || [];
                    html += `
                        <div class="group-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <strong>#${idx + 1}</strong>
                                <span style="background: ${color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                                    ${(group.avg_similarity * 100).toFixed(1)}% sim
                                </span>
                            </div>
                            <div class="images-preview">
                    `;
                    
                    files.slice(0, 6).forEach((file, imgIdx) => {
                        const filePath = file.path || file;
                        const encodedPath = encodeURIComponent(filePath);
                        html += `
                            <img class="image-thumb" 
                                 src="/api/image?path=${encodedPath}"
                                 data-path="${filePath.replace(/"/g, '&quot;')}"
                                 onclick="openModal(this.dataset.path)"
                                 alt="thumb ${imgIdx}">
                        `;
                    });
                    
                    if (files.length > 6) {
                        html += `
                            <div style="display: flex; align-items: center; justify-content: center; background: #f5f5f7; border-radius: 4px; font-size: 0.75rem; color: #86868b; font-weight: 600;">
                                +${files.length - 6}
                            </div>
                        `;
                    }
                    
                    html += `
                            </div>
                            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #86868b;">
                                Similarity: ${(group.avg_similarity * 100).toFixed(1)}%
                            </div>
                        </div>
                    `;
                });
                
                html += `
                            </div>
                        </div>
                    </div>
                `;
                
                return html;
            }
            
            // Load stats
            async function updateStats() {
                let statsText = '';
                if (scanDuplicates) {
                    const exact = scanDuplicates.exact_groups?.length || 0;
                    const perceptual = scanDuplicates.perceptual_groups?.length || 0;
                    if (exact > 0 || perceptual > 0) {
                        statsText = `Scanner: ${exact} exact + ${perceptual} perceptual`;
                    }
                }
                if (groups.length > 0) {
                    if (statsText) statsText += ' • ';
                    statsText += `AI: ${groups.length} groups`;
                }
                if (!statsText) {
                    statsText = 'Ready to scan';
                }
                document.getElementById('stats-total').textContent = statsText;
            }
            
            // Initialize
            updateStats();
        </script>
    </body>
    </html>
    """
    return html_content


# ... (API routes and helper functions - reuse from app_v3.py)
# Include all the API endpoints: /api/groups, /api/status, /api/scan-info, 
# /api/scan-duplicates, /api/scan-data, /api/ai-groups-categorized,
# /api/scan, /api/clear-embeddings, /api/embeddings, /api/image, /api/review,
# /api/add-to-duplicates, /api/open-duplicates-album, /api/batch-delete

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
    
    if scan_results_path.exists():
        try:
            with open(scan_results_path) as f:
                scan_data = json.load(f)
                info["total_images"] = len(scan_data)
                info["scan_completed"] = True
        except Exception as e:
            logger.error(f"Failed to load scan results: {e}")
    
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache_data = json.load(f)
                info["cache_entries"] = len(cache_data.get("entries", {}))
            info["cache_size_mb"] = round(cache_path.stat().st_size / (1024 * 1024), 2)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
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
    """Return deterministic duplicate groups from the scan step."""
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
    
    try:
        with open(groups_file) as f:
            ai_groups = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load AI groups: {e}")
    
    exact_files = set()
    perceptual_files = set()
    
    if duplicates_file.exists():
        try:
            with open(duplicates_file) as f:
                dup_data = json.load(f)
                for group in dup_data.get("exact_groups", []):
                    for file_path in group.get("files", []):
                        exact_files.add(file_path)
                for group in dup_data.get("perceptual_groups", []):
                    for file_path in group.get("files", []):
                        perceptual_files.add(file_path)
        except Exception as e:
            logger.warning(f"Failed to load duplicates file: {e}")
    
    overlaps_exact = []
    overlaps_perceptual = []
    new_discoveries = []
    
    for group in ai_groups:
        group_files = {file_info["path"] for file_info in group.get("files", [])}
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
    status = DEMO_PROCESSING_STATUS if SERVER_MODE == "demo" else PROCESSING_STATUS
    if status["running"]:
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
    
    embeddings_dir = PROJECT_ROOT / ("embeddings_demo" if SERVER_MODE == "demo" else "embeddings")
    for file in ["similar_groups.json", "similar_pairs.json", "embeddings.npy", "metadata.json"]:
        file_path = embeddings_dir / file
        if file_path.exists():
            file_path.unlink()
    
    return {"success": True, "message": "Embeddings cleared"}


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
        return await estimate_embeddings(request)
    
    background_tasks.add_task(run_embeddings, request)
    return {"success": True, "message": "Embedding generation started"}


@app.get("/api/image")
async def get_image(path: str):
    """Serve an image file."""
    file_path = Path(path)
    
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {file_path}")
    
    try:
        file_path.stat()
        return FileResponse(file_path)
    except PermissionError:
        if "Photos Library.photoslibrary" in str(file_path):
            raise HTTPException(
                status_code=403, 
                detail="Cannot access Photos Library. Please enable Full Disk Access or re-scan with AppleScript."
            )
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error serving image {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def run_scan(request: ScanRequest):
    """Run scanner in background."""
    status = DEMO_PROCESSING_STATUS if SERVER_MODE == "demo" else PROCESSING_STATUS
    status["running"] = True
    status["stage"] = "Scanning"
    status["message"] = "Initializing photo scanner..."
    
    try:
        cmd = [
            "python", "-m", "src.scanner.main",
            "--output", "scan_for_embeddings.json",
            "--duplicates-output", "scan_duplicates.json",
            "--cache-file", ".cache/scan_cache.json",
            "--md5-mode", request.md5_mode,
        ]
        
        if request.source == "photos-library":
            cmd.append("--photos-library")
            if request.photos_access == "applescript":
                cmd.extend(["--use-applescript", "--keep-export"])
                status["message"] = "Scanning Photos Library (AppleScript)..."
            else:
                status["message"] = "Scanning Photos Library (direct access)..."
        else:
            scan_path = Path(request.path) if Path(request.path).is_absolute() else PROJECT_ROOT / request.path
            cmd.append(str(scan_path))
            status["message"] = f"Scanning directory: {scan_path.name}..."
        
        if not request.use_cache:
            cmd.append("--no-cache")
        
        if request.limit:
            cmd.extend(["--limit", str(request.limit)])
        
        logger.info(f"Running scan: {' '.join(cmd)}")
        status["message"] = "Scan in progress... This may take a few moments."
        
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
            status["message"] = f"Scan failed: Check terminal logs"
            logger.error(f"Scan failed with return code {process.returncode}")
    
    except Exception as e:
        status["message"] = f"Scan error: {str(e)}"
        logger.error(f"Scan error: {e}")
    
    finally:
        status["running"] = False
        status["stage"] = ""


async def estimate_embeddings(request: EmbeddingRequest):
    """Estimate time and resources for embedding generation."""
    if SERVER_MODE == "demo":
        scan_file = PROJECT_ROOT / "scan_results_demo.json"
    else:
        scan_file = PROJECT_ROOT / "scan_for_embeddings.json"
    
    try:
        with open(scan_file) as f:
            scan_data = json.load(f)
        
        total_images = len(scan_data)
        sample_size = min(request.estimate_sample, total_images)
        
        if total_images == 0:
            return {"success": False, "error": "No images found"}
        
        # Simplified estimation
        estimated_minutes = max(1, int(total_images / 50))  # Rough estimate
        grouping_minutes = max(1, int(estimated_minutes * 0.1))
        
        return {
            "success": True,
            "total_images": total_images,
            "sample_size": sample_size,
            "throughput": {"images_per_second": 50},
            "estimates": {
                "embedding_minutes": estimated_minutes,
                "grouping_minutes": grouping_minutes,
                "total_minutes": estimated_minutes + grouping_minutes,
                "total_hours": round((estimated_minutes + grouping_minutes) / 60, 2),
            },
            "note": "Estimate based on typical throughput"
        }
        
    except Exception as e:
        logger.error(f"Estimation error: {e}")
        return {"success": False, "error": str(e)}


async def run_embeddings(request: EmbeddingRequest):
    """Run embedding generation in background."""
    status = DEMO_PROCESSING_STATUS if SERVER_MODE == "demo" else PROCESSING_STATUS
    status["running"] = True
    status["stage"] = "Generating Embeddings"
    status["message"] = f"Generating embeddings ({request.inference_mode} mode)... This may take several minutes."
    
    try:
        cmd = [
            "python", "-m", "src.embedding.main_v2",
            "scan_for_embeddings.json",
            "--output", "embeddings_demo" if SERVER_MODE == "demo" else "embeddings",
            "--similarity-threshold", str(request.similarity_threshold),
            "--mode", request.inference_mode,
        ]
        
        if request.inference_mode in ("remote", "auto"):
            cmd.extend(["--service-url", INFERENCE_SERVICE_URL])
        
        logger.info(f"Running embeddings: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            output_dir = "embeddings_demo" if SERVER_MODE == "demo" else "embeddings"
            groups_file = PROJECT_ROOT / output_dir / "similar_groups.json"
            
            if groups_file.exists():
                with open(groups_file) as f:
                    groups_data = json.load(f)
                    for group in groups_data:
                        group["reviewed"] = False
                    
                    if SERVER_MODE == "demo":
                        globals()["DEMO_SIMILAR_GROUPS"] = groups_data
                    else:
                        globals()["SIMILAR_GROUPS"] = groups_data
            
            status["message"] = f"Found {len(groups_data)} similarity groups"
            logger.info(f"Embeddings completed, found {len(groups_data)} groups")
        else:
            status["message"] = "Embedding generation failed - check terminal logs"
            logger.error(f"Embedding generation failed with return code {process.returncode}")
    
    except Exception as e:
        status["message"] = f"Embedding error: {str(e)}"
        logger.error(f"Embedding error: {e}")
    
    finally:
        status["running"] = False
        status["stage"] = ""


def load_data(
    scan_results: Path,
    similar_groups: Path,
    embeddings_dir: Path,
    path_mapping_file: Optional[Path] = None,
    is_demo: bool = False,
):
    """Load scan results, groups, and initialize feedback learner."""
    global SERVER_MODE, PATH_MAPPING, PHOTOS_METADATA, DEMO_MODE
    if is_demo:
        SERVER_MODE = "demo"
        global DEMO_SIMILAR_GROUPS, DEMO_EMBEDDING_STORE, DEMO_FEEDBACK_LEARNER
        if similar_groups.exists():
            with open(similar_groups, "r") as f:
                DEMO_SIMILAR_GROUPS = json.load(f)
        DEMO_EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
        DEMO_FEEDBACK_LEARNER = FeedbackLearner()
    else:
        SERVER_MODE = "main"
        global SIMILAR_GROUPS, EMBEDDING_STORE, FEEDBACK_LEARNER
        if similar_groups.exists():
            with open(similar_groups, "r") as f:
                SIMILAR_GROUPS = json.load(f)
        EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
        FEEDBACK_LEARNER = FeedbackLearner()
    
    # Load path mapping if available
    if path_mapping_file and path_mapping_file.exists():
        with open(path_mapping_file, "r") as f:
            PATH_MAPPING = json.load(f)
        logger.info(f"[{'DEMO' if is_demo else 'MAIN'}] Loaded path mapping with {len(PATH_MAPPING)} entries")
    
    # Load photos metadata (UUID -> original filename mapping)
    photos_metadata_file = PROJECT_ROOT / ".cache" / "photos_metadata.json"
    if photos_metadata_file.exists():
        try:
            with open(photos_metadata_file, "r") as f:
                metadata_list = json.load(f)
            for item in metadata_list:
                photo_id = item.get('id', '')
                filename = item.get('filename', '')
                uuid_part = photo_id.split('/')[0] if '/' in photo_id else photo_id
                if uuid_part and filename:
                    PHOTOS_METADATA[uuid_part] = filename
            logger.info(f"[{'DEMO' if is_demo else 'MAIN'}] Loaded photos metadata with {len(PHOTOS_METADATA)} UUID->filename mappings")
        except Exception as e:
            logger.warning(f"Failed to load photos metadata: {e}")
    
    logger.info(f"[{'DEMO' if is_demo else 'MAIN'}] Data loaded")
