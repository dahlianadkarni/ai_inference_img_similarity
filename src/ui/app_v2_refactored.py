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
    photos_access: str = "applescript"  # "applescript" or "originals"
    path: Optional[str] = None
    limit: Optional[int] = 20
    use_cache: bool = True
    md5_mode: str = "on-demand"  # "on-demand", "always", "never"


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
                    <button class="btn btn-keep" onclick="startEmbeddings()" id="btn-embeddings">🧠 Generate Embeddings</button>
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
            let scanDuplicates = null;
            let scanDataMap = {}; // file_path -> scan data with size
            let selectedImages = new Map(); // groupId -> Set of selected indices
            let selectedExactImages = new Map(); // groupIdx -> Set of image indices (for exact duplicates)
            let selectedPerceptualImages = new Map(); // groupIdx -> Set of image indices
            let collapsedSections = { exact: false, perceptual: false }; // type -> collapsed state
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
                }
                
                // Sort files by date (oldest first)
                const sortedFiles = files.map((path, idx) => ({
                    path,
                    originalIdx: idx,
                    date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                })).sort((a, b) => a.date.localeCompare(b.date));
                
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
                if (modalCurrentGroup.type === 'perceptual' || modalCurrentGroup.type === 'exact') {
                    const selectedMap = modalCurrentGroup.type === 'exact' ? selectedExactImages : selectedPerceptualImages;
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
                if (!modalCurrentGroup) return;
                
                const type = modalCurrentGroup.type;
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                const groupIdx = modalCurrentGroup.groupIdx;
                const imageIdx = modalCurrentGroup.originalIndices ? modalCurrentGroup.originalIndices[modalCurrentIndex] : modalCurrentIndex;
                
                if (!selectedMap.has(groupIdx)) {
                    selectedMap.set(groupIdx, new Set());
                }
                const selected = selectedMap.get(groupIdx);
                if (selected.has(imageIdx)) {
                    selected.delete(imageIdx);
                } else {
                    selected.add(imageIdx);
                }
                
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
            
            function toggleImageSelection(groupIdx, imageIdx, type) {
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                if (!selectedMap.has(groupIdx)) {
                    selectedMap.set(groupIdx, new Set());
                }
                const selected = selectedMap.get(groupIdx);
                if (selected.has(imageIdx)) {
                    selected.delete(imageIdx);
                } else {
                    selected.add(imageIdx);
                }
                render();
            }
            
            function togglePerceptualImage(groupIdx, imageIdx) {
                toggleImageSelection(groupIdx, imageIdx, 'perceptual');
            }
            
            function toggleExactImage(groupIdx, imageIdx) {
                toggleImageSelection(groupIdx, imageIdx, 'exact');
            }
            
            function getOldestImageIndex(files) {
                const filesWithDates = files.map((path, idx) => ({
                    idx,
                    date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99'
                }));
                filesWithDates.sort((a, b) => a.date.localeCompare(b.date));
                return filesWithDates[0].idx;
            }
            
            function selectAllDefault(type) {
                const groups = type === 'exact' ? scanDuplicates?.exact_groups : scanDuplicates?.perceptual_groups;
                if (!groups) return;
                
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                selectedMap.clear();
                
                groups.forEach((group, groupIdx) => {
                    const oldestIdx = getOldestImageIndex(group.files);
                    const selected = new Set(group.files.map((_, idx) => idx).filter(idx => idx !== oldestIdx));
                    if (selected.size > 0) {
                        selectedMap.set(groupIdx, selected);
                    }
                });
                render();
            }
            
            function selectAllPerceptualDefault() {
                selectAllDefault('perceptual');
            }
            
            function selectAllExactDefault() {
                selectAllDefault('exact');
            }
            
            function deselectAll(type) {
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                selectedMap.clear();
                render();
            }
            
            function deselectAllPerceptual() {
                deselectAll('perceptual');
            }
            
            function deselectAllExact() {
                deselectAll('exact');
            }
            
            function clearGroup(type, groupIdx) {
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                selectedMap.delete(groupIdx);
                render();
            }
            
            function clearPerceptualGroup(groupIdx) {
                clearGroup('perceptual', groupIdx);
            }
            
            function clearExactGroup(groupIdx) {
                clearGroup('exact', groupIdx);
            }
            
            async function batchDeletePerceptualSelected() {
                const filesToDelete = [];
                let totalSize = 0;
                
                selectedPerceptualImages.forEach((selected, groupIdx) => {
                    const group = scanDuplicates.perceptual_groups[groupIdx];
                    if (!group) return;
                    
                    selected.forEach(imgIdx => {
                        const path = group.files[imgIdx];
                        filesToDelete.push(path);
                        totalSize += scanDataMap[path]?.file_size || 0;
                    });
                });
                
                if (filesToDelete.length === 0) return;
                
                if (!confirm(`Delete ${totalFiles} selected photo(s)?\\n\\nSpace saved: ${formatBytes(totalSize)}`)) {
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
                    loadGroups();
                } catch (error) {
                    alert('Failed: ' + error.message);
                }
            }
            
            function clearPerceptualGroup(groupIdx) {
                selectedPerceptualImages.delete(groupIdx);
                render();
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
                statusDiv.textContent = message;
            }

            async function loadGroups() {
                try {
                    console.log('Loading groups...');
                    const response = await fetch('/api/groups');
                    const data = await response.json();
                    groups = data.groups || data; // Handle both old and new format
                    const threshold = data.threshold || 0.85;
                    console.log('Loaded AI groups:', groups.length);

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

                    // Auto-select default deletions (keep oldest)
                    if (scanDuplicates?.exact_groups && scanDuplicates.exact_groups.length > 0) {
                        selectAllExactDefault();
                    }
                    if (scanDuplicates?.perceptual_groups && scanDuplicates.perceptual_groups.length > 0) {
                        selectAllPerceptualDefault();
                    }

                    console.log('About to call render()');
                    updateStats(threshold);
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
            
            function calculateSavings(groupType) {
                const selected = selectedGroups[groupType];
                let totalFiles = 0;
                let totalBytes = 0;
                
                const groupsList = groupType === 'exact' ? 
                    scanDuplicates?.exact_groups : scanDuplicates?.perceptual_groups;
                    
                if (!groupsList) return { files: 0, bytes: 0 };
                
                groupsList.forEach((group, idx) => {
                    if (selected.has(idx)) {
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
                    }
                });
                
                return { files: totalFiles, bytes: totalBytes };
            }
            
            function toggleGroupSelection(groupType, groupIndex) {
                if (selectedGroups[groupType].has(groupIndex)) {
                    selectedGroups[groupType].delete(groupIndex);
                } else {
                    selectedGroups[groupType].add(groupIndex);
                }
                render();
            }
            
            function selectAllGroups(groupType) {
                const groupsList = groupType === 'exact' ? 
                    scanDuplicates?.exact_groups : scanDuplicates?.perceptual_groups;
                if (!groupsList) return;
                
                selectedGroups[groupType].clear();
                groupsList.forEach((_, idx) => selectedGroups[groupType].add(idx));
                render();
            }
            
            function deselectAllGroups(groupType) {
                selectedGroups[groupType].clear();
                render();
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
                            
                            // Calculate stats based on individual image selection
                            let totalSelectedImages = 0;
                            let totalSelectedSize = 0;
                            selectedExactImages.forEach((selected, idx) => {
                                const group = exactGroups[idx];
                                if (group) {
                                    selected.forEach(imgIdx => {
                                        totalSelectedImages++;
                                        totalSelectedSize += scanDataMap[group.files[imgIdx]]?.file_size || 0;
                                    });
                                }
                            });
                            
                            const savings = { files: totalSelectedImages, bytes: totalSelectedSize };
                            const totalSavings = calculateTotalSavings('exact');
                            const isCollapsed = collapsedSections.exact;
                            
                            html += `
                                <div class="type-section">
                                    <div class="type-header exact ${isCollapsed ? 'collapsed' : ''}" onclick="toggleSection('exact')">
                                        <div style="flex: 1;">
                                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                                <span style="font-size: 1.5rem;">${isCollapsed ? '▶' : '▼'}</span>
                                                <div>
                                                    <h2 style="margin: 0; font-size: 1.25rem;">🟢 Exact Duplicates (MD5 Hash)</h2>
                                                    <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.9;">
                                                        ${exactGroups.length} groups • ${totalSavings.files} files can be deleted (default: keep oldest)
                                                        ${totalSelectedImages > 0 ? ` • <strong>${totalSelectedImages} images selected</strong>` : ''}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-size: 1.125rem; font-weight: 600;">💾 ${formatBytes(totalSavings.bytes)}</div>
                                            <div style="font-size: 0.75rem; opacity: 0.9;">${totalSelectedImages > 0 ? `Selected: ${formatBytes(totalSelectedSize)}` : 'Potential savings'}</div>
                                        </div>
                                    </div>
                                    <div class="type-content ${isCollapsed ? 'collapsed' : ''}">
                                        <div style="background: white; padding: 1rem; border-radius: 0 0 12px 12px;">
                                            <div style="margin-bottom: 1rem; padding: 0.75rem; background: #e8f5e9; border-radius: 6px; color: #2e7d32; font-size: 0.875rem;">
                                                💡 <strong>Tip:</strong> Default keeps oldest photo per group. Click images to change selection. Click image to view full size.
                                            </div>
                                            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); selectAllExactDefault()" style="background: #34c759; color: white;">
                                                    ✓ Select Default (All Groups)
                                                </button>
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); deselectAllExact()" style="background: #8e8e93; color: white;">
                                                    ✗ Clear All
                                                </button>
                                            </div>
                                            ${totalSelectedImages > 0 ? `
                                                <div class="selection-summary exact">
                                                    <div>
                                                        <strong style="font-size: 1.125rem;">${totalSelectedImages} images selected</strong>
                                                        <div style="margin-top: 0.25rem; opacity: 0.9;">
                                                            Will delete ${savings.files} photos • Save ${formatBytes(savings.bytes)}
                                                        </div>
                                                    </div>
                                                    <button class="btn btn-delete" onclick="event.stopPropagation(); batchDeleteSelected('exact')" style="background: white; color: #34c759; font-weight: 600;">
                                                        🗑️ Delete Selected
                                                    </button>
                                                </div>
                                            ` : ''}
                                            <div class="groups-grid">
                            `;
                            exactGroups.forEach((group, idx) => {
                                html += renderExactGroup(group, idx, '#34c759');
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
                                                        ${perceptualGroups.length} groups • ${totalPotentialFiles} files can be deleted (default: keep oldest)
                                                        ${totalSelectedImages > 0 ? ` • <strong>${totalSelectedImages} images selected</strong>` : ''}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-size: 1.125rem; font-weight: 600;">💾 ${formatBytes(totalPotentialSize)}</div>
                                            <div style="font-size: 0.75rem; opacity: 0.9;">${totalSelectedImages > 0 ? `Selected: ${formatBytes(totalSelectedSize)}` : 'Potential savings'}</div>
                                        </div>
                                    </div>
                                    <div class="type-content ${isCollapsedPerceptual ? 'collapsed' : ''}">
                                        <div style="background: white; padding: 1rem; border-radius: 0 0 12px 12px;">
                                            <div style="margin-bottom: 1rem; padding: 0.75rem; background: #fff3cd; border-radius: 6px; color: #856404; font-size: 0.875rem;">
                                                💡 <strong>Tip:</strong> Default keeps oldest photo per group. Click images to change selection. Click image to view full size.
                                            </div>
                                            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); selectAllPerceptualDefault()" style="background: #ff9500; color: white;">
                                                    ✓ Select Default (All Groups)
                                                </button>
                                                <button class="btn btn-keep" onclick="event.stopPropagation(); deselectAllPerceptual()" style="background: #8e8e93; color: white;">
                                                    ✗ Clear All
                                                </button>
                                            </div>
                                            ${totalSelectedImages > 0 ? `
                                                <div class="selection-summary perceptual">
                                                    <div>
                                                        <strong style="font-size: 1.125rem;">${totalSelectedImages} images selected</strong>
                                                        <div style="margin-top: 0.25rem; opacity: 0.9;">
                                                            Will delete ${totalSelectedImages} photos • Save ${formatBytes(totalSelectedSize)}
                                                        </div>
                                                    </div>
                                                    <button class="btn btn-delete" onclick="event.stopPropagation(); batchDeletePerceptualSelected()" style="background: white; color: #ff9500; font-weight: 600;">
                                                        🗑️ Delete Selected
                                                    </button>
                                                </div>
                                            ` : ''}
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
                        html += `
                            <div class="type-section">
                                <div class="type-header ai">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.25rem;">🔵 AI Similar Groups (Embedding Analysis)</h2>
                                        <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; opacity: 0.9;">${groups.length} groups • Deep learning similarity</p>
                                    </div>
                                </div>
                        `;
                        groups.forEach((group, idx) => {
                            html += renderAIGroup(group, idx);
                        });
                        html += '</div>';
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
            
            function renderDuplicateGroup(group, index, type, color) {
                const files = group.files;
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                const selected = selectedMap.get(index) || new Set();
                const totalSize = getGroupSize(files);
                const filesWithInfo = getSortedFilesWithInfo(files);
                const oldestIdx = filesWithInfo[0].originalIdx;
                
                // Calculate selection stats
                const selectedSize = Array.from(selected).reduce((sum, idx) => 
                    sum + (scanDataMap[files[idx]]?.file_size || 0), 0);
                const hasSelection = selected.size > 0;
                
                let html = '<div class="mini-group-card" style="';
                if (hasSelection) html += 'border: 2px solid #ff9500; background: #fff9f5;';
                html += '">';
                
                html += '<div class="mini-group-header" style="margin-bottom: 0.75rem;">';
                html += '<span style="font-weight: 600; color: #1d1d1f;">#' + (index + 1) + '</span>';
                html += '<span class="similarity-badge" style="background: ' + color + '; font-size: 0.625rem; padding: 0.25rem 0.5rem;">';
                html += files.length;
                html += '</span>';
                html += '</div>';
                
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 0.5rem; margin-bottom: 0.75rem;">';
                filesWithInfo.forEach(fileInfo => {
                    const path = fileInfo.path;
                    const fileIdx = fileInfo.originalIdx;
                    const isSelected = selected.has(fileIdx);
                    const isOldest = fileIdx === oldestIdx;
                    const willBeKept = !isSelected;
                    
                    html += '<div style="position: relative;">';
                    
                    html += '<img src="/api/image?path=' + encodeURIComponent(path) + '" ';
                    html += 'alt="thumb" ';
                    html += 'style="width: 100%; height: 70px; object-fit: cover; border-radius: 4px; border: ' + (isSelected ? '2px solid #ff9500' : '2px solid #34c759') + '; position: relative; z-index: 1;">';
                    
                    // Overlay for clicking to toggle (covers entire thumbnail)
                    html += '<div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; cursor: pointer; z-index: 2;" ';
                    html += 'onclick="toggleImageSelection(' + index + ', ' + fileIdx + ', &quot;' + type + '&quot;);" ';
                    html += 'title="' + (willBeKept ? 'Click to select for deletion' : 'Click to keep') + '"></div>';
                    
                    // Badge or checkbox on top
                    if (willBeKept) {
                        html += '<div style="position: absolute; top: 2px; left: 2px; background: #34c759; color: white; font-size: 0.625rem; padding: 2px 4px; border-radius: 3px; z-index: 3; font-weight: 600; pointer-events: none;">';
                        html += isOldest ? 'KEEP ★' : 'KEEP';
                        html += '</div>';
                    } else {
                        html += '<input type="checkbox" checked ';
                        html += 'style="position: absolute; top: 2px; left: 2px; width: 18px; height: 18px; cursor: pointer; z-index: 3;" ';
                        html += 'onclick="event.stopPropagation(); toggleImageSelection(' + index + ', ' + fileIdx + ', &quot;' + type + '&quot;);">';
                    }
                    
                    // View button for modal
                    html += '<div style="position: absolute; bottom: 2px; right: 2px; background: rgba(0,0,0,0.7); color: white; font-size: 0.625rem; padding: 2px 6px; border-radius: 3px; z-index: 3; cursor: pointer;" ';
                    html += 'onclick="event.stopPropagation(); openModal(&quot;' + path.replace(/'/g, "\\\\'") + '&quot;, &quot;' + path.split('/').pop().replace(/'/g, "\\\\'") + '&quot;, &quot;' + type + '&quot;, ' + index + ');" ';
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
                
                if (hasSelection) {
                    html += '<div style="margin-top: 0.5rem; padding: 0.5rem; background: #fff5f5; border-radius: 4px; font-size: 0.75rem; color: #ff3b30; text-align: center;">';
                    html += selected.size + ' selected for deletion';
                    html += '</div>';
                }
                
                html += '<div style="display: flex; gap: 0.25rem; margin-top: 0.5rem;">';
                html += '<button class="btn btn-delete" onclick="deleteSelected(&quot;' + type + '&quot;, ' + index + ')" ';
                html += 'style="flex: 1; padding: 0.5rem; font-size: 0.75rem;"';
                if (!hasSelection) html += ' disabled';
                html += '>🗑️ Delete (' + selected.size + ')</button>';
                
                html += '<button class="btn btn-keep" onclick="clearGroup(&quot;' + type + '&quot;, ' + index + ')" ';
                html += 'style="flex: 1; padding: 0.5rem; font-size: 0.75rem; background: #34c759; color: white;"';
                if (!hasSelection) html += ' disabled';
                html += '>✓ Keep All</button>';
                html += '</div>';
                
                html += '</div>';
                return html;
            }
            
            function renderPerceptualGroup(group, index, color) {
                return renderDuplicateGroup(group, index, 'perceptual', color);
            }
            
            function renderExactGroup(group, index, color) {
                return renderDuplicateGroup(group, index, 'exact', color);
            }
            
            async function deleteSelected(type, groupIdx) {
                const selectedMap = type === 'exact' ? selectedExactImages : selectedPerceptualImages;
                const selected = selectedMap.get(groupIdx);
                if (!selected || selected.size === 0) return;
                
                const groups = type === 'exact' ? scanDuplicates.exact_groups : scanDuplicates.perceptual_groups;
                const group = groups[groupIdx];
                const filesToDelete = Array.from(selected).map(idx => group.files[idx]);
                const totalSize = filesToDelete.reduce((sum, path) => 
                    sum + (scanDataMap[path]?.file_size || 0), 0);
                
                if (!confirm(`Delete ${selected.size} selected photo(s)?\\n\\nSpace saved: ${formatBytes(totalSize)}`)) {
                    return;
                }
                
                try {
                    let deletedCount = 0;
                    for (const path of filesToDelete) {
                        const success = await deleteFromPhotos(path);
                        if (success) deletedCount++;
                    }
                    
                    alert(`✓ Deleted ${deletedCount} photos, saved ${formatBytes(totalSize)}`);
                    selectedMap.delete(groupIdx);
                    loadGroups();
                } catch (error) {
                    alert('Failed: ' + error.message);
                }
            }
            
            function deletePerceptualSelected(groupIdx) {
                return deleteSelected('perceptual', groupIdx);
            }
            
            function deleteExactSelected(groupIdx) {
                return deleteSelected('exact', groupIdx);
            }

            function renderCompactGroup(group, index, type, color) {
                const files = group.files;
                const isSelected = selectedGroups[type].has(index);
                const totalSize = getGroupSize(files);
                
                // Calculate what will be deleted (all except oldest by photo date)
                const filesWithDates = files.map(path => ({
                    path,
                    // Use photo_date (EXIF capture date) if available, fall back to created_at (file creation)
                    date: scanDataMap[path]?.photo_date || scanDataMap[path]?.created_at || '9999-99-99',
                    size: scanDataMap[path]?.file_size || 0
                }));
                filesWithDates.sort((a, b) => a.date.localeCompare(b.date));
                const deleteSize = filesWithDates.slice(1).reduce((sum, f) => sum + f.size, 0);
                
                let html = '<div class="mini-group-card' + (isSelected ? ' selected' : '') + '" ';
                html += 'onclick="toggleGroupSelection(\\'' + type + '\\', ' + index + ')">';
                html += '<div class="mini-group-header">';
                html += '<input type="checkbox" ' + (isSelected ? 'checked' : '') + ' ';
                html += 'onclick="event.stopPropagation(); toggleGroupSelection(\\'' + type + '\\', ' + index + ');" ';
                html += 'style="margin: 0; cursor: pointer;">';
                html += '<span style="font-weight: 600; color: #1d1d1f;">#' + (index + 1) + '</span>';
                html += '<span class="similarity-badge" style="background: ' + color + '; font-size: 0.625rem; padding: 0.25rem 0.5rem;">';
                html += files.length;
                html += '</span>';
                html += '</div>';
                html += '<div class="mini-images-preview">';
                
                files.slice(0, 4).forEach(path => {
                    html += '<img src="/api/image?path=' + encodeURIComponent(path) + '" ';
                    html += 'alt="thumb" ';
                    html += 'onclick="event.stopPropagation(); openModal(\\'' + path.replace(/'/g, "\\\\'") + '\\', \\'' + path.split('/').pop().replace(/'/g, "\\\\'") + '\\', \\'' + type + '\\', ' + index + ');">';
                });
                
                if (files.length > 4) {
                    html += '<div style="display: flex; align-items: center; justify-content: center; background: #f5f5f7; border-radius: 4px; font-size: 0.75rem; color: #86868b;">+' + (files.length - 4) + '</div>';
                }
                
                html += '</div>';
                html += '<div class="group-stats">';
                html += '<span>📦 ' + formatBytes(totalSize) + '</span>';
                html += '<span>💾 Save ' + formatBytes(deleteSize) + '</span>';
                html += '</div>';
                html += '</div>';
                
                return html;
            }
            
            async function batchDeleteSelected(groupType) {
                const selectedMap = groupType === 'exact' ? selectedExactImages : selectedPerceptualImages;
                if (selectedMap.size === 0) return;
                
                // Count total selected images and calculate size
                let totalFiles = 0;
                let totalSize = 0;
                const groupsList = groupType === 'exact' ? 
                    scanDuplicates.exact_groups : scanDuplicates.perceptual_groups;
                
                selectedMap.forEach((selected, groupIdx) => {
                    const group = groupsList[groupIdx];
                    if (group) {
                        selected.forEach(imgIdx => {
                            totalFiles++;
                            totalSize += scanDataMap[group.files[imgIdx]]?.file_size || 0;
                        });
                    }
                });
                
                if (totalFiles === 0) return;
                
                if (!confirm(`Delete ${totalFiles} photos?\n\nSpace saved: ${formatBytes(totalSize)}`)) {
                    return;
                }
                
                try {
                    let totalDeleted = 0;
                    
                    // Process each group
                    for (const [groupIdx, selected] of selectedMap.entries()) {
                        const group = groupsList[groupIdx];
                        if (!group) continue;
                        
                        // Delete selected images in this group
                        for (const imgIdx of selected) {
                            const filePath = group.files[imgIdx];
                            const success = await deleteFromPhotos(filePath);
                            if (success) totalDeleted++;
                        }
                    }
                    
                    alert(`✓ Deleted ${totalDeleted} photos, saved ${formatBytes(totalSize)}`);
                    selectedMap.clear();
                    loadGroups();
                } catch (error) {
                    alert('Failed: ' + error.message);
                }
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
                const selected = selectedImages.get(group.group_id) || new Set();
                const hasSelection = selected.size > 0;
                
                return `
                    <div class="group-card" style="${group.reviewed ? 'opacity: 0.6;' : ''} margin-top: 0; border-radius: 0;">
                        <div class="group-header">
                            <h3>Group ${index + 1} • ${group.size} images</h3>
                            <span class="similarity-badge" style="background: #007aff;">${(group.avg_similarity * 100).toFixed(1)}% similar</span>
                        </div>
                        
                        ${!group.reviewed ? `
                            <div class="feedback-notice">
                                💡 <strong>Tip:</strong> Click images to expand. Select duplicates to delete, or mark as not similar to teach the AI.
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
                                         alt="Image ${fileIdx + 1}"
                                         title="Click to expand"
                                         onclick="event.stopPropagation(); toggleImageExpand(event);">
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


@app.get("/api/image")
async def get_image(path: str, thumb: bool = False):
    """Serve an image file, optionally as a thumbnail."""
    file_path = Path(path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
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
    PROCESSING_STATUS["stage"] = "scanning"
    PROCESSING_STATUS["message"] = "Scanning for photos..."
    
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
        else:
            # Resolve relative paths from project root
            scan_path = Path(request.path)
            if not scan_path.is_absolute():
                scan_path = PROJECT_ROOT / scan_path
            cmd.append(str(scan_path))

        if not request.use_cache:
            cmd.append("--no-cache")
        
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
