# Implementation Notes / Project History

This file preserves the original README content as a running implementation/history document.

---

# Photo Near-Duplicate Detection Platform

A two-phase AI-powered system to identify and manage near-duplicate photos on macOS, with an architecture designed for distributed GPU inference.

## Project Goals

Build practical experience with:
- AI model serving and inference frameworks
- OpenAI-compatible API design
- Kubernetes and containerization
- GPU infrastructure and optimization
- Distributed system observability

## Architecture Overview

### Two-Phase Inference Design

**Phase 1: Local Coarse Pass (CPU/MPS)**
- Fast, lightweight embeddings on Mac
- Filter obvious non-duplicates
- Minimize remote GPU cost

**Phase 2: Remote GPU Refinement (Future)**
- High-quality embeddings for candidate pairs or groups
- Cost-optimized inference at scale

**Phase 3: Human Review**
- Side-by-side comparison UI
- Safe deletion workflow
- Metadata-driven decision support

## Current Status

### ✅ Completed
- Project initialization
- **Phase 1a: Photo scanner with hash-based duplicate detection**
  - File discovery and metadata extraction
  - Exact duplicate detection (MD5 hash)
  - Perceptual duplicate detection (dHash)
  - Parallel processing with progress bars
  - **macOS Photos Library integration via AppleScript** (no Full Disk Access needed!)
  - Direct file access fallback option
  - CLI tool for testing
- **Phase 1b: ML embedding generation (coarse pass)**
  - OpenCLIP integration for semantic similarity
  - Batch embedding generation with MPS/CUDA/CPU support
  - Embedding storage and retrieval
  - Similarity search with configurable thresholds
  - Finds near-duplicates missed by perceptual hashing
- **Phase 3: Human-review web UI (Enhanced)**
  - FastAPI backend with RESTful endpoints
  - Tabbed interface for Scanner Results vs AI Results
  - **Exact Duplicates Section:**
    - Group-level activation with checkboxes
    - Auto-selected by default (keeps oldest, marks rest for deletion)
    - Individual image toggle (KEEP ★ / DEL ✗ badges)
    - Date-sorted display (oldest first)
    - Full-screen modal with prev/next navigation
    - Batch deletion with disk savings display
  - **Perceptual Duplicates Section:**
    - Same UI/UX as exact duplicates (unified codebase)
    - Manual group activation (unchecked by default)
    - Orange theme for deletion markers
    - Review-first workflow
  - **AI Similar Groups Section:**
    - Side-by-side comparison interface
    - Mark groups as "Not Similar" for feedback
    - Re-analyze with learned feedback
  - AppleScript integration to delete from Photos Library
  - Real-time disk savings calculations
  - Dynamic thumbnail sizing based on group size

### 🚧 In Progress
- Performance optimization for large photo libraries
- Enhanced feedback learning for AI similarity

### 📋 Planned
- Phase 4: OpenAI-compatible API wrapper
- Containerization and K8s deployment
- Observability stack (Prometheus + Grafana)
- Distributed GPU inference for Phase 2 refinement

## Technical Stack

- **Backend**: Python 3.11+
- **ML Framework**: PyTorch (MPS support for Apple Silicon)
- **Embedding Model**: MobileCLIP / OpenCLIP (lightweight variants)
- **Vector Search**: FAISS / hnswlib
- **API Framework**: FastAPI
- **Frontend**: React + TailwindCSS
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Observability**: Prometheus + Grafana

## Project Structure

```
inference_1/
├── src/
│   ├── scanner/         # PhotoKit integration + image preprocessing
│   ├── embedding/       # Model loading + inference
│   ├── grouping/        # Similarity search + clustering
│   ├── api/            # OpenAI-compatible endpoints
│   └── ui/             # Human-review interface
├── deploy/
│   ├── docker/         # Dockerfiles
│   └── k8s/            # Kubernetes manifests
├── infra/              # Infrastructure as code
├── tests/              # Unit and integration tests
└── notebooks/          # Exploration and evaluation
```

## Getting Started

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test (Phase 1a: Scanner)

**Recommended: Use AppleScript (No Full Disk Access needed!)**
```bash
# Activate venv (required)
source venv/bin/activate

# Export and scan photos via AppleScript
python -m src.scanner.main --photos-library --use-applescript --limit 50
```

On first run, macOS will prompt: **"Terminal wants to control Photos"** - Click **OK**. This only requires "Automation" permission, not Full Disk Access!

**Alternative Options:**

**Option 1: Test with sample images first**
```bash
source venv/bin/activate

# Create test directory with your own photos
mkdir -p ~/Pictures/test_photos
# Copy some photos there, then:
python -m src.scanner.main ~/Pictures/test_photos --limit 50
```

**Option 2: Direct file access** (requires Full Disk Access)
```bash
source venv/bin/activate
python -m src.scanner.main --photos-library --limit 100
# If permission denied, grant Full Disk Access in System Preferences
```

**Results:**
- `scan_results.json`: Full metadata for all images
- Console output: Exact and perceptual duplicate groups

**What this demonstrates:**
- Direct file access to macOS Photos (no Swift/Objective-C needed!)
- Fast pre-filtering before expensive ML inference
- Parallel image processing
- Metadata extraction (EXIF, dimensions, hashes)
- Multi-level duplicate detection strategy

### Phase 1b: Generate ML Embeddings

After scanning, generate semantic embeddings to catch near-duplicates that hashes miss:

```bash
# First, install ML dependencies
pip install torch torchvision open_clip_torch numpy

# Generate embeddings from scan results
python -m src.embedding.main scan_results.json

# Or with custom settings
python -m src.embedding.main scan_results.json \
  --model ViT-B-32 \
  --similarity-threshold 0.90 \
  --batch-size 32
```

**What this finds:**
- Cropped versions of the same photo
- Photos with different filters/edits applied
- Same scene with different exposure/white balance
- Burst shots with slight composition changes

**Output:**
- `embeddings/` directory with stored embeddings
- `embeddings/similar_pairs.json` with detected near-duplicates and similarity scores

### Phase 3: Review and Delete Duplicates

Launch the web UI to visually review duplicate pairs and delete them from Photos:

```bash
# Start the review UI server
python -m src.ui.main

# Opens at http://127.0.0.1:8000
```

**Features:**
- 📸 Side-by-side image comparison
- 📊 Similarity scores for each pair
- ✅ Mark pairs as reviewed
- 🗑️ Delete duplicates directly from Photos Library (via AppleScript)
- 📈 Progress tracking (reviewed vs total)

**Actions per pair:**
- **Keep Both**: Mark as reviewed, no deletion
- **Delete Left**: Remove left image from Photos
- **Delete Right**: Remove right image from Photos

### Next Steps
- Add ML embeddings for semantic similarity (catching edits, crops, filters)
- Build web UI for visual review
- Create OpenAI-compatible API wrapper

## Privacy & Security

- **Default**: Local-only processing (photos never leave your Mac)
- **Optional**: Remote GPU mode with thumbnail-only upload
- **Controls**: Explicit consent, encrypted transit, no retention policy

## License

MIT
