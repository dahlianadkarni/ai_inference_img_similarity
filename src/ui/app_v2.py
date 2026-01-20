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
PATH_MAPPING: dict = {}  # Maps local cached paths to original Photos Library paths


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
            <div class="control-panel">
                <h2>🔧 Control Panel</h2>
                
                <div class="control-row">
                    <strong>Step 1: Scan Photos</strong>
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
                </div>
                
                <div id="scan-info" style="display: none; background: #f5f5f7; padding: 0.75rem 1rem; border-radius: 6px; margin-left: 2rem; margin-bottom: 0.75rem; font-size: 0.875rem; color: #86868b;">
                    <strong style="color: #1d1d1f;">📊 Last Scan:</strong>
                    <span id="scan-total">-</span> images •
                    Cache: <span id="scan-cache">-</span> entries (<span id="scan-cache-size">-</span>) •
                    Duplicates: <span id="scan-exact-dupes">-</span> exact, <span id="scan-perceptual-dupes">-</span> perceptual •
                    MD5: <span id="scan-md5-mode">-</span>
                </div>
                
                <div class="control-row">
                    <strong>Step 2: Generate Embeddings</strong>
                    <label style="font-size: 0.875rem; color: #86868b;">Similarity threshold:</label>
                    <input id="similarity-threshold" type="number" class="control-input" placeholder="Threshold" value="0.85" step="0.01" min="0" max="1" style="width: 100px;">
                    <span style="font-size: 0.75rem; color: #86868b;">(Higher = more strict, e.g., 0.99 = nearly identical)</span>
                    <button class="btn" onclick="estimateEmbeddings()" id="btn-estimate" style="background: #5856d6; color: white;">⏱️ Estimate Time</button>
                    <button class="btn btn-keep" onclick="startEmbeddings()" id="btn-embeddings">🧠 Generate Embeddings</button>
                </div>
                
                <div id="estimate-results" style="display: none; margin-top: 0.5rem; padding: 1rem; background: #e3f2fd; border-radius: 8px; border: 1px solid #1976d2;">
                    <div style="font-size: 0.875rem; color: #1565c0;"></div>
                </div>
                
                <div class="control-row">
                    <strong>After Giving Feedback:</strong>
                    <button class="btn btn-not-similar" onclick="startEmbeddings()">🔄 Re-analyze with Feedback</button>
                </div>
                
                <div id="processing-status" class="processing-status"></div>
            </div>

            <div id="groups-container">
                <div class="loading">⏳ Initializing...</div>
            </div>
            
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
                overlaps_with_exact_matches: false,
                overlaps_with_perceptual_matches: false,
                new_ai_discoveries: false
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
            
            async function batchDeleteAISelected() {
                const filesToDelete = [];
                let totalSize = 0;
                
                activeAIGroups.forEach(groupId => {
                    const selected = selectedImages.get(groupId);
                    const group = groups.find(g => g.group_id === groupId);
                    if (!group || !selected) return;
                    
                    selected.forEach(imgIdx => {
                        const path = group.files[imgIdx].path;
                        filesToDelete.push(path);
                        totalSize += scanDataMap[path]?.file_size || 0;
                    });
                });
                
                if (filesToDelete.length === 0) return;
                
                if (!confirm(`Delete ${filesToDelete.length} selected photo(s)?\\n\\nSpace saved: ${formatBytes(totalSize)}`)) {
                    return;
                }
                
                try {
                    let deletedCount = 0;
                    for (const path of filesToDelete) {
                        const success = await deleteFromPhotos(path);
                        if (success) deletedCount++;
                    }
                    
                    alert(`✓ Deleted ${deletedCount} photos, saved ${formatBytes(totalSize)}`);
                    selectedImages.clear();
                    activeAIGroups.clear();
                    loadGroups();
                } catch (error) {
                    alert('Failed: ' + error.message);
                }
            }
            
            async function batchDeletePerceptualSelected() {
                const filesToDelete = [];
                let totalSize = 0;
                
                activePerceptualGroups.forEach(groupIdx => {
                    const selected = selectedPerceptualImages.get(groupIdx);
                    const group = scanDuplicates.perceptual_groups[groupIdx];
                    if (!group || !selected) return;
                    
                    selected.forEach(imgIdx => {
                        const path = group.files[imgIdx];
                        filesToDelete.push(path);
                        totalSize += scanDataMap[path]?.file_size || 0;
                    });
                });
                
                if (filesToDelete.length === 0) return;
                
                if (!confirm(`Delete ${filesToDelete.length} selected photo(s)?\\n\\nSpace saved: ${formatBytes(totalSize)}`)) {
                    return;
                }
                
                try {
                    let deletedCount = 0;
                    for (const path of filesToDelete) {
                        const success = await deleteFromPhotos(path);
                        if (success) deletedCount++;
                    }
                    
                    alert(`✓ Deleted ${deletedCount} photos, saved ${formatBytes(totalSize)}`);
                    selectedPerceptualImages.clear();
                    activePerceptualGroups.clear();
                    loadGroups();
                } catch (error) {
                    alert('Failed: ' + error.message);
                }
            }
            
            async function batchDeleteExactSelected() {
                const filesToDelete = [];
                let totalSize = 0;
                
                activeExactGroups.forEach(groupIdx => {
                    const selected = selectedExactImages.get(groupIdx);
                    const group = scanDuplicates.exact_groups[groupIdx];
                    if (!group || !selected) return;
                    
                    selected.forEach(imgIdx => {
                        const path = group.files[imgIdx];
                        filesToDelete.push(path);
                        totalSize += scanDataMap[path]?.file_size || 0;
                    });
                });
                
                if (filesToDelete.length === 0) return;
                
                if (!confirm(`Delete ${filesToDelete.length} selected photo(s)?\\n\\nSpace saved: ${formatBytes(totalSize)}`)) {
                    return;
                }
                
                try {
                    let deletedCount = 0;
                    for (const path of filesToDelete) {
                        const success = await deleteFromPhotos(path);
                        if (success) deletedCount++;
                    }
                    
                    alert(`✓ Deleted ${deletedCount} photos, saved ${formatBytes(totalSize)}`);
                    selectedExactImages.clear();
                    activeExactGroups.clear();
                    loadGroups();
                } catch (error) {
                    alert('Failed: ' + error.message);
                }
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
            document.addEventListener('DOMContentLoaded', () => {
                loadScanInfo(); // Load scan info on page load
                
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
                
                // Show immediate feedback
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
                
                // Show immediate feedback
                showStatus('running', 'Initializing embedding generation...');
                
                try {
                    const response = await fetch('/api/embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ similarity_threshold: threshold, estimate: false })
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
                            estimate_sample: 30
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
                            showStatus('running', `${status.stage}: ${status.message}`);
                        } else {
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
                    const data = await response.json();
                    
                    // Store categorized groups
                    aiGroupsOverlapExact = data.overlaps_exact || [];
                    aiGroupsOverlapPerceptual = data.overlaps_perceptual || [];
                    aiGroupsNewDiscoveries = data.new_discoveries || [];
                    
                    // Keep old 'groups' variable for compatibility with existing code
                    groups = [...aiGroupsOverlapExact, ...aiGroupsOverlapPerceptual, ...aiGroupsNewDiscoveries];
                    
                    console.log(`AI groups: ${data.counts.overlaps_exact} overlap exact, ${data.counts.overlaps_perceptual} overlap perceptual, ${data.counts.new_discoveries} new discoveries`);

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
                        }
                    } catch (e) {
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

            function render() {
                const container = document.getElementById('groups-container');
                
                const hasScanData = scanDuplicates && 
                    ((scanDuplicates.exact_groups && scanDuplicates.exact_groups.length > 0) ||
                     (scanDuplicates.perceptual_groups && scanDuplicates.perceptual_groups.length > 0));
                const hasAIData = groups && groups.length > 0;
                
                // Build tabs
                let tabsHtml = '';
                if (hasScanData || hasAIData) {
                    tabsHtml = `
                        <div class="tabs">
                            <button class="tab ${currentTab === 'scan' ? 'active' : ''}" 
                                    onclick="switchTab('scan')"
                                    ${!hasScanData ? 'disabled' : ''}>
                                📊 Scanner Results ${hasScanData ? `(${(scanDuplicates.exact_groups?.length || 0) + (scanDuplicates.perceptual_groups?.length || 0)})` : '(empty)'}
                            </button>
                            <button class="tab ${currentTab === 'ai' ? 'active' : ''}" 
                                    onclick="switchTab('ai')"
                                    ${!hasAIData ? 'disabled' : ''}>
                                🧠 AI Results ${hasAIData ? `(${groups.length})` : '(empty)'}
                            </button>
                        </div>
                    `;
                }
                
                let html = tabsHtml;
                
                // Scanner tab content
                if (currentTab === 'scan') {
                    html += '<div class="tab-content active">';
                    
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
                            const exactGroups = scanDuplicates.exact_groups;
                            const isCollapsed = collapsedSections.exact;
                            
                            // Calculate total potential savings (all but oldest in each group)
                            let totalPotentialFiles = 0;
                            let totalPotentialSize = 0;
                            exactGroups.forEach(group => {
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
                            
                            html += `
                                <div class="type-section">
                                    <div class="type-header exact ${isCollapsed ? 'collapsed' : ''}" onclick="toggleSection('exact')">
                                        <div style="flex: 1;">
                                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                                <span style="font-size: 1.5rem;">${isCollapsed ? '▶' : '▼'}</span>
                                                <div>
                                                    <h2 style="margin: 0; font-size: 1.25rem;">� Exact Duplicates (MD5 Hash)</h2>
                                                    <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.9;">
                                        ${exactGroups.length} groups • ${totalPotentialFiles} files can be deleted
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
                            <div style="margin-bottom: 1rem; padding: 0.75rem; background: #e8f5e9; border-radius: 6px; color: #2e7d32; font-size: 0.875rem;">
                                💡 <strong>Tip:</strong> Check groups to include in batch delete. Click thumbnails to toggle individual images. Default keeps oldest photo.
                            </div>
                            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; align-items: center;">
                                <button class="btn btn-keep" onclick="event.stopPropagation(); selectAllExactDefault()" style="background: #34c759; color: white;">
                                    ✓ Select All
                                </button>
                                <button class="btn btn-keep" onclick="event.stopPropagation(); deselectAllExact()" style="background: #8e8e93; color: white;">
                                    ✗ Clear
                                </button>
                                <button class="btn btn-delete" onclick="event.stopPropagation(); batchDeleteExactSelected()" style="background: #34c759; color: white; margin-left: auto;">
                                    🗑️ Delete Selected
                                </button>
                                <span id="exact-savings-display" style="font-size: 0.875rem; color: #34c759; font-weight: 600; white-space: nowrap;"></span>
                            </div>
                            <div class="groups-grid">
                            `;
                            exactGroups.forEach((group, idx) => {
                                html += renderCompactGroup(group, idx, 'exact', '#34c759');
                            });
                            html += '</div></div></div></div>';
                        }
                        
                        // Perceptual matches section
                        if (scanDuplicates.perceptual_groups && scanDuplicates.perceptual_groups.length > 0) {
                            const perceptualGroups = scanDuplicates.perceptual_groups;
                            const isCollapsedPerceptual = collapsedSections.perceptual;
                            
                            // Calculate total potential savings (all but oldest in each group)
                            let totalPotentialFiles = 0;
                            let totalPotentialSize = 0;
                            perceptualGroups.forEach(group => {
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
                            
                            // Count currently selected images across all groups
                            let totalSelectedImages = 0;
                            let totalSelectedSize = 0;
                            selectedPerceptualImages.forEach((selected, idx) => {
                                const group = perceptualGroups[idx];
                                if (group) {
                                    selected.forEach(imgIdx => {
                                        totalSelectedImages++;
                                        totalSelectedSize += scanDataMap[group.files[imgIdx]]?.file_size || 0;
                                    });
                                }
                            });
                            
                            html += `
                                <div class="type-section">
                                    <div class="type-header perceptual ${isCollapsedPerceptual ? 'collapsed' : ''}" onclick="toggleSection('perceptual')">
                                        <div style="flex: 1;">
                                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                                <span style="font-size: 1.5rem;">${isCollapsedPerceptual ? '▶' : '▼'}</span>
                                                <div>
                                                    <h2 style="margin: 0; font-size: 1.25rem;">🟠 Perceptual Duplicates (Review Required)</h2>
                                                    <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.9;">
                                                        ${perceptualGroups.length} groups • ${totalPotentialFiles} files can be deleted
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-size: 1.125rem; font-weight: 600;">💾 ${formatBytes(totalPotentialSize)}</div>
                                            <div style="font-size: 0.75rem; opacity: 0.9;">Potential savings</div>
                                        </div>
                                    </div>
                                    <div class="type-content ${isCollapsedPerceptual ? 'collapsed' : ''}">
                                        <div style="background: white; padding: 1rem; border-radius: 0 0 12px 12px;">
                                            <div style="margin-bottom: 1rem; padding: 0.75rem; background: #fff3cd; border-radius: 6px; color: #856404; font-size: 0.875rem;">
                                                💡 <strong>Tip:</strong> Check groups to include in batch delete. Click thumbnails to toggle individual images. Default keeps oldest photo.
                                            </div>
                                            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; align-items: center;">
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); selectAllPerceptualDefault()" style="background: #ff9500; color: white;">
                                                    ✓ Select All
                                                </button>
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); deselectAllPerceptual()" style="background: #8e8e93; color: white;">
                                                    ✗ Clear
                                                </button>
                                                <button class="btn btn-delete" onclick="event.stopPropagation(); batchDeletePerceptualSelected()" style="background: #ff9500; color: white; margin-left: auto;">
                                                    🗑️ Delete Selected
                                                </button>
                                                <span id="perceptual-savings-display" style="font-size: 0.875rem; color: #ff9500; font-weight: 600; white-space: nowrap;"></span>
                                            </div>
                                            <div class="groups-grid">
                            `;
                            perceptualGroups.forEach((group, idx) => {
                                html += renderPerceptualGroup(group, idx, '#ff9500');
                            });
                            html += '</div></div></div></div>';
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
                    
                    html += '</div>';
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
                        // Helper function to render AI section
                        const renderAISection = (title, groups, emoji, color, description, savingsDisplayId) => {
                            if (groups.length === 0) return '';
                            
                            const sectionKey = title.toLowerCase().replace(/\s+/g, '_');
                            const isCollapsed = collapsedSections[sectionKey] || false;
                            
                            // Calculate potential savings
                            let totalPotentialFiles = 0;
                            let totalPotentialSize = 0;
                            groups.forEach(group => {
                                if (group.files.length > 1) {
                                    for (let i = 1; i < group.files.length; i++) {
                                        totalPotentialFiles++;
                                        totalPotentialSize += scanDataMap[group.files[i].path]?.file_size || 0;
                                    }
                                }
                            });
                            
                            let sectionHtml = `
                                <div class="type-section" style="margin-bottom: 1.5rem;">
                                    <div class="type-header" style="background: ${color};" onclick="toggleSection('${sectionKey}')">
                                        <div style="flex: 1;">
                                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                                <span style="font-size: 1.5rem;">${isCollapsed ? '▶' : '▼'}</span>
                                                <div>
                                                    <h2 style="margin: 0; font-size: 1.25rem;">${emoji} ${title}</h2>
                                                    <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.9;">
                                                        ${groups.length} groups • ${totalPotentialFiles} potentially similar files
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
                                            <div style="margin-bottom: 1rem; padding: 0.75rem; background: #f5f5f7; border-radius: 6px; color: #1d1d1f; font-size: 0.875rem;">
                                                ${description}
                                            </div>
                                            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; align-items: center;">
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); selectAllAIDefault()" style="background: #007aff; color: white;">
                                                    ✓ Select All
                                                </button>
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); deselectAllAI()" style="background: #8e8e93; color: white;">
                                                    ✗ Clear
                                                </button>
                                                <button class="btn btn-delete" onclick="event.stopPropagation(); batchDeleteAISelected()" style="background: #007aff; color: white; margin-left: auto;">
                                                    🗑️ Delete Selected
                                                </button>
                                                <span id="${savingsDisplayId}" style="font-size: 0.875rem; color: #007aff; font-weight: 600; white-space: nowrap;"></span>
                                            </div>
                                            <div class="groups-grid">
                            `;
                            groups.forEach((group) => {
                                sectionHtml += renderAIGroup(group, group.group_id);
                            });
                            sectionHtml += '</div></div></div></div>';
                            
                            return sectionHtml;
                        };
                        
                        // Render three sections

                        html += renderAISection(
                            'New AI Discoveries',
                            aiGroupsNewDiscoveries,
                            '🔴',
                            'linear-gradient(135deg, #c41e3a 0%, #8b0000 100%)',
                            '💡 These AI groups contain images <strong>not found in exact or perceptual duplicates</strong>. These are truly new discoveries based on visual content analysis - like different angles of the same scene or semantically similar images.',
                            'ai-new-savings-display'
                        );

                        html += renderAISection(
                            'Overlaps with Exact Matches',
                            aiGroupsOverlapExact,
                            '🟢',
                            'linear-gradient(135deg, #34c759 0%, #30d158 100%)',
                            '💡 These AI groups contain images that are also in <strong>Exact Duplicate</strong> groups. The AI found visual similarity among images that are already exact byte-for-byte matches.',
                            'ai-exact-savings-display'
                        );
                        
                        html += renderAISection(
                            'Overlaps with Perceptual Matches',
                            aiGroupsOverlapPerceptual,
                            '🟠',
                            'linear-gradient(135deg, #ff9500 0%, #ff9f0a 100%)',
                            '💡 These AI groups contain images that are also in <strong>Perceptual Duplicate</strong> groups. The AI found visual similarity among images that were already flagged as perceptually similar.',
                            'ai-perceptual-savings-display'
                        );
                        

                        
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
                
                // Fallback if no tabs
                if (!tabsHtml) {
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
            function renderDuplicateGroup(group, index, type, themeColor) {
                const files = group.files;
                
                // Get type-specific state
                const isActive = type === 'exact' ? activeExactGroups.has(index) : activePerceptualGroups.has(index);
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                const selected = isActive ? (selectedMap.get(index) || new Set()) : new Set();
                const toggleFunction = type === 'exact' ? 'toggleExactImage' : 'togglePerceptualImage';
                
                const totalSize = getGroupSize(files);
                const filesWithInfo = getSortedFilesWithInfo(files);
                const oldestIdx = filesWithInfo[0].originalIdx;
                
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
                html += 'onclick="toggleGroupActive(\\'' + type + '\\', ' + index + ')" ';
                html += 'style="width: 18px; height: 18px; cursor: pointer;" ';
                html += 'title="' + (isActive ? 'Uncheck to exclude from batch delete' : 'Check to include in batch delete') + '">';
                html += '<span style="font-weight: 600; color: #1d1d1f;">#' + (index + 1) + '</span>';
                html += '<span class="similarity-badge" style="background: ' + themeColor + '; font-size: 0.625rem; padding: 0.25rem 0.5rem;">';
                html += files.length;
                html += '</span>';
                html += '</div>';
                
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(' + thumbSize + ', 1fr)); gap: 0.5rem; margin-bottom: 0.75rem;">';
                filesWithInfo.forEach(fileInfo => {
                    const path = fileInfo.path;
                    const fileIdx = fileInfo.originalIdx;
                    const isSelected = selected.has(fileIdx);
                    const isOldest = fileIdx === oldestIdx;
                    const willBeKept = !isSelected;
                    
                    html += '<div style="position: relative;">';
                    
                    // Image with border (orange for delete, green for keep)
                    html += '<img src="/api/image?path=' + encodeURIComponent(path) + '" ';
                    html += 'alt="thumb" ';
                    html += 'style="width: 100%; height: ' + thumbSize + '; object-fit: cover; border-radius: 4px; border: ' + (isSelected ? '2px solid #ff9500' : '2px solid #34c759') + '; position: relative; z-index: 1;">';
                    
                    // Overlay for clicking to toggle (covers entire thumbnail)
                    html += '<div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; cursor: pointer; z-index: 2;" ';
                    html += 'onclick="' + toggleFunction + '(' + index + ', ' + fileIdx + ');" ';
                    html += 'title="' + (willBeKept ? 'Click to select for deletion' : 'Click to keep') + '"></div>';
                    
                    // Badge on top (green for keep, orange for delete)
                    if (willBeKept) {
                        html += '<div style="position: absolute; top: 2px; left: 2px; background: #34c759; color: white; font-size: 0.625rem; padding: 2px 4px; border-radius: 3px; z-index: 3; font-weight: 600; pointer-events: none;">';
                        html += isOldest ? 'KEEP ★' : 'KEEP';
                        html += '</div>';
                    } else {
                        html += '<div style="position: absolute; top: 2px; left: 2px; background: #ff9500; color: white; font-size: 0.625rem; padding: 2px 4px; border-radius: 3px; z-index: 3; font-weight: 600; pointer-events: none;">';
                        html += 'DEL ✗';
                        html += '</div>';
                    }
                    
                    // View button for modal
                    html += '<div style="position: absolute; bottom: 2px; right: 2px; background: rgba(0,0,0,0.7); color: white; font-size: 0.625rem; padding: 2px 6px; border-radius: 3px; z-index: 3; cursor: pointer;" ';
                    html += 'onclick="event.stopPropagation(); openModal(\\'' + path.replace(/'/g, "\\\\'") + '\\', \\'' + path.split('/').pop().replace(/'/g, "\\\\'") + '\\', \\'' + type + '\\', ' + index + ');" ';
                    html += 'title="View full size">🔍</div>';
                    
                    html += '</div>';
                });
                html += '</div>';
                
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
            function renderPerceptualGroup(group, index, color) {
                return renderDuplicateGroup(group, index, 'perceptual', color);
            }
            
            function renderCompactGroup(group, index, type, color) {
                return renderDuplicateGroup(group, index, type, color);
            }
            
            
            async function deleteFromPhotos(filePath) {
                const filename = filePath.split('/').pop();
                try {
                    // This calls the backend delete function
                    const response = await fetch('/api/delete-photo', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ file_path: filePath })
                    });
                    return response.ok;
                } catch (error) {
                    console.error('Delete failed:', error);
                    return false;
                }
            }

            function renderAIGroup(group, index) {
                const files = group.files.map(f => f.path);
                const groupId = group.group_id;
                
                // Get AI-specific state
                const isActive = activeAIGroups.has(groupId);
                const selected = isActive ? (selectedImages.get(groupId) || new Set()) : new Set();
                
                const totalSize = getGroupSize(files);
                
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
                html += 'onclick="toggleGroupActive(\\'' + 'ai' + '\\', ' + groupId + ')" ';
                html += 'style="width: 18px; height: 18px; cursor: pointer;" ';
                html += 'title="' + (isActive ? 'Uncheck to exclude from batch delete' : 'Check to include in batch delete') + '">';
                html += '<span style="font-weight: 600; color: #1d1d1f;">#' + (index + 1) + '</span>';
                html += '<span class="similarity-badge" style="background: #007aff; font-size: 0.625rem; padding: 0.25rem 0.5rem;">';
                html += files.length;
                html += '</span>';
                html += '<span style="font-size: 0.625rem; color: #86868b; margin-left: auto;">' + (group.avg_similarity * 100).toFixed(1) + '% sim</span>';
                html += '</div>';
                
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(' + thumbSize + ', 1fr)); gap: 0.5rem; margin-bottom: 0.75rem;">';
                group.files.forEach((file, fileIdx) => {
                    const path = file.path;
                    const isSelected = selected.has(fileIdx);
                    const willBeKept = !isSelected;
                    
                    html += '<div style="position: relative;">';
                    
                    // Image with border (orange for delete, green for keep)
                    html += '<img src="/api/image?path=' + encodeURIComponent(path) + '" ';
                    html += 'alt="thumb" ';
                    html += 'style="width: 100%; height: ' + thumbSize + '; object-fit: cover; border-radius: 4px; border: ' + (isSelected ? '2px solid #ff9500' : '2px solid #34c759') + '; position: relative; z-index: 1;">';
                    
                    // Overlay for clicking to toggle (covers entire thumbnail)
                    html += '<div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; cursor: pointer; z-index: 2;" ';
                    html += 'onclick="toggleAIImage(' + groupId + ', ' + fileIdx + ');" ';
                    html += 'title="' + (willBeKept ? 'Click to select for deletion' : 'Click to keep') + '"></div>';
                    
                    // Badge on top (green for keep, orange for delete)
                    if (willBeKept) {
                        html += '<div style="position: absolute; top: 2px; left: 2px; background: #34c759; color: white; font-size: 0.625rem; padding: 2px 4px; border-radius: 3px; z-index: 3; font-weight: 600; pointer-events: none;">';
                        html += 'KEEP';
                        html += '</div>';
                    } else {
                        html += '<div style="position: absolute; top: 2px; left: 2px; background: #ff9500; color: white; font-size: 0.625rem; padding: 2px 4px; border-radius: 3px; z-index: 3; font-weight: 600; pointer-events: none;">';
                        html += 'DEL ✗';
                        html += '</div>';
                    }
                    
                    // View button for modal
                    html += '<div style="position: absolute; bottom: 2px; right: 2px; background: rgba(0,0,0,0.7); color: white; font-size: 0.625rem; padding: 2px 6px; border-radius: 3px; z-index: 3; cursor: pointer;" ';
                    html += 'onclick="event.stopPropagation(); openModal(\\'' + path.replace(/'/g, "\\\\'") + '\\', \\'' + path.split('/').pop().replace(/'/g, "\\\\'") + '\\', \\'' + 'ai' + '\\', ' + groupId + ');" ';
                    html += 'title="View full size">🔍</div>';
                    
                    html += '</div>';
                });
                html += '</div>';
                
                html += '<div class="group-stats" style="margin-top: 0.5rem; display: flex; gap: 0.5rem; font-size: 0.75rem; color: #86868b;">';
                html += '<span>📦 ' + formatBytes(totalSize) + '</span>';
                if (hasSelection) {
                    html += '<span style="color: #ff3b30; font-weight: 600;">💾 Save ' + formatBytes(selectedSize) + '</span>';
                }
                html += '</div>';
                
                html += '</div>';
                return html;
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
    return {
        "groups": SIMILAR_GROUPS,
        "threshold": CURRENT_THRESHOLD
    }


@app.get("/api/status")
async def get_status():
    """Get processing status."""
    return PROCESSING_STATUS


@app.get("/api/scan-info")
async def get_scan_info():
    """Get information about the completed scan."""
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
    report_path = PROJECT_ROOT / "scan_duplicates.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="scan_duplicates.json not found (run Scan first)")
    try:
        with open(report_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scan_duplicates.json: {e}")


@app.get("/api/scan-data")
async def get_scan_data():
    """Return full scan data with file sizes."""
    scan_path = PROJECT_ROOT / "scan_for_embeddings.json"
    if not scan_path.exists():
        raise HTTPException(status_code=404, detail="scan_for_embeddings.json not found")
    try:
        with open(scan_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scan data: {e}")


@app.get("/api/ai-groups-categorized")
async def get_ai_groups_categorized():
    """Categorize AI groups by overlap with exact/perceptual duplicates."""
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


@app.post("/api/embeddings")
async def generate_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Generate embeddings from scan results."""
    if PROCESSING_STATUS["running"]:
        raise HTTPException(status_code=409, detail="Already processing")
    
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


@app.post("/api/delete-photo")
async def delete_photo(request: dict):
    """Delete a single photo from Photos Library."""
    file_path = request.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path required")
    
    success = await delete_from_photos(file_path)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete photo")


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
    global PROCESSING_STATUS, SIMILAR_GROUPS, EMBEDDING_STORE, CURRENT_THRESHOLD
    PROCESSING_STATUS["running"] = True
    PROCESSING_STATUS["stage"] = "Generating Embeddings"
    PROCESSING_STATUS["message"] = "Loading image data and initializing AI model..."
    
    try:
        cmd = [
            "python", "-m", "src.embedding.main",
            "scan_for_embeddings.json",
            "--output", "embeddings",
            "--similarity-threshold", str(request.similarity_threshold)
        ]
        
        logger.info(f"Running embeddings: {' '.join(cmd)}")
        
        PROCESSING_STATUS["message"] = f"Generating embeddings (threshold: {request.similarity_threshold})... This may take several minutes."
        
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
            stderr_text = stderr.decode()
            # Write full stderr to log file
            write_stderr_to_log(stderr_text, "embeddings")
            # Show clean error in UI
            error_msg = clean_stderr_message(stderr_text)
            PROCESSING_STATUS["message"] = f"Embedding generation failed: {error_msg}"
            logger.error(f"Embedding generation failed with return code {process.returncode}")
    
    except Exception as e:
        PROCESSING_STATUS["message"] = f"Embedding error: {str(e)}"
        logger.error(f"Embedding error: {e}")
    
    finally:
        PROCESSING_STATUS["running"] = False
        PROCESSING_STATUS["stage"] = ""


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


async def delete_from_photos(file_path: str) -> bool:
    """Delete a photo from Photos Library via AppleScript."""
    # Try to get original path from mapping (if using cached copies)
    original_path = PATH_MAPPING.get(file_path, file_path)
    filename = Path(original_path).name
    
    # Use full path comparison if we have the original path
    if original_path != file_path and "/Photos Library.photoslibrary/" in original_path:
        # We have the original Photos Library path - use it for precise matching
        script = f'''
        tell application "Photos"
            set posixPath to "{original_path}"
            set itemList to media items
            repeat with theItem in itemList
                try
                    -- Compare filename and try to match by path characteristics
                    if filename of theItem is "{filename}" then
                        delete theItem
                        return "success"
                    end if
                end try
            end repeat
            return "not_found"
        end tell
        '''
    else:
        # Fallback to filename-only matching
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
            logger.info(f"Deleted from Photos: {filename} (original: {original_path})")
            return True
        else:
            logger.warning(f"Photo not found in Photos Library: {filename}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to delete photo: {e}")
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
        
        groups_processed += 1
    
    return {
        "success": True,
        "groups_processed": groups_processed,
        "deleted_count": deleted_count
    }


def load_data(
    scan_results: Path,
    similar_groups: Path,
    embeddings_dir: Path,
    path_mapping_file: Optional[Path] = None,
):
    """Load scan results, groups, and initialize feedback learner."""
    global SIMILAR_GROUPS, SCAN_DATA, FEEDBACK_LEARNER, EMBEDDING_STORE, PATH_MAPPING
    
    if similar_groups.exists():
        with open(similar_groups, "r") as f:
            SIMILAR_GROUPS = json.load(f)
        logger.info(f"Loaded {len(SIMILAR_GROUPS)} similar groups")
    
    if scan_results.exists():
        with open(scan_results, "r") as f:
            SCAN_DATA = json.load(f)
        logger.info(f"Loaded {len(SCAN_DATA)} scan results")
    
    # Load path mapping if available
    if path_mapping_file and path_mapping_file.exists():
        with open(path_mapping_file, "r") as f:
            PATH_MAPPING = json.load(f)
        logger.info(f"Loaded path mapping with {len(PATH_MAPPING)} entries")
    else:
        logger.info("No path mapping file found - delete will use filename-only matching")
    
    # Initialize feedback learner
    FEEDBACK_LEARNER = FeedbackLearner()
    feedback_path = embeddings_dir / "feedback.pkl"
    if feedback_path.exists():
        FEEDBACK_LEARNER.load(str(feedback_path))
    
    # Load embedding store
    EMBEDDING_STORE = EmbeddingStore(embeddings_dir)
    logger.info(f"Loaded {len(EMBEDDING_STORE)} embeddings")
