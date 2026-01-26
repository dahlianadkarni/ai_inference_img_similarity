# Photo Near-Duplicate Detection (macOS Photos)

Finds groups of near-duplicate photos using OpenCLIP embeddings, then lets you review them in a local web UI and optionally delete selected photos from Apple Photos.

If you’re looking for the older “project history / phase notes”, it lives in docs/IMPLEMENTATION_NOTES.md.

## What It Does

- **Scans photos** from Apple Photos (via AppleScript) or any directory
- **Detects duplicates** at three levels:
  - Exact duplicates (MD5 hash matching)
  - Perceptual duplicates (dHash - finds resized/edited versions)
  - AI similar groups (OpenCLIP embeddings - finds semantically similar photos)
- **Review UI** with separate workflows:
  - **Exact Duplicates**: Auto-selected by default, batch delete with one click
  - **Perceptual Duplicates**: Manual review, check groups to activate
  - **AI Groups**: Fine-grained selection, teach the AI with feedback
- **Smart features**:
  - Keeps oldest photo by default (based on EXIF date)
  - Shows disk savings in real-time
  - Full-screen modal with keyboard navigation
  - Individual image toggle within groups

## Quickstart

See DEMO_SETUP.md and DEMO_SOLUTION.md for demo

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start UI only (port 8000)
python -m src.ui.main

# Or start both inference + UI
python start_services.py

# Or use demo dataset (UI on port 8001; inference runs on port 8002)
python start_services.py --ui-demo
```

Open http://127.0.0.1:8000 (or the demo UI at http://127.0.0.1:8001). The inference service runs on http://127.0.0.1:8002.

## End-to-End Workflow (How the Pieces Fit)

### Step 1 — Scan

Produces a JSON list of images to analyze.

- Output: scan_for_embeddings.json
- UI: Use the “Control Panel → Start Scan” button
- CLI (optional):

```bash
source venv/bin/activate
python -m src.scanner.main --photos-library --use-applescript --keep-export --limit 200
```

Tip: With `--keep-export`, AppleScript exports into `.cache/photos_export` by default and subsequent scans can be incremental (it skips re-exporting files that are already present).

macOS will prompt “Terminal wants to control Photos” on first run. Allow it.

If you want estimates before running a full scan of your Photos “originals” (Full Disk Access required), run:

```bash
source venv/bin/activate
python -m src.scanner.main --photos-library --estimate --estimate-sample 200
```

### Step 2 — Generate Embeddings + Similarity Groups

Computes embeddings and writes similarity groups to disk.

- Output directory: embeddings/
- Group output: embeddings/similar_groups.json
- UI: Use “Control Panel → Generate Embeddings” and set your threshold
- CLI (optional):

```bash
source venv/bin/activate
python -m src.embedding.main scan_for_embeddings.json --output embeddings --similarity-threshold 0.85
```

Notes:
- Higher thresholds (e.g., 0.95–0.99) are stricter.
- Embeddings are cleared before regenerating so old data doesn’t accumulate.

### Step 3 — Review Groups (Delete / Keep / Feedback)

The UI has three tabs:

**Scanner Results Tab:**
- **Exact Duplicates** (green theme): All groups checked by default, oldest photo kept
  - Uncheck groups to exclude from batch delete
  - Click thumbnails to toggle individual images (KEEP ★ / DEL ✗)
  - Click Delete Selected to remove marked photos
- **Perceptual Duplicates** (orange theme): Manual review workflow
  - Check groups to activate them
  - Same toggle functionality as exact duplicates
  - Review and delete in batches

**AI Results Tab:**
- Side-by-side comparison of similar groups
- Delete Selected: deletes chosen photos from Apple Photos
- Keep All: marks the group reviewed
- Not Similar (Teach AI): stores "negative examples" in embeddings/feedback.pkl

**Universal Features:**
- 🔍 Click to open full-screen modal with prev/next navigation
- Real-time disk savings calculation
- Keyboard shortcuts in modal (← → arrows, ESC to close)

### Step 4 — Re-analyze With Feedback

After giving feedback, click “Re-analyze with Feedback” (or regenerate embeddings) to apply the learned penalties/boosts during grouping.

## Outputs

- scan_for_embeddings.json — scan results used for embedding generation
- embeddings/embeddings.npy + embeddings/metadata.json — stored embeddings
- embeddings/similar_groups.json — groups to review
- embeddings/feedback.pkl — persisted feedback examples

## Repo Map

- src/scanner/ — scanning + AppleScript Photos export
- src/embedding/ — OpenCLIP embedding generation + storage
- src/grouping/ — clustering + feedback learner
- src/ui/ — FastAPI UI server

## Safety Notes

- Everything runs locally by default.
- Deletion uses AppleScript to delete photos from the Photos library; start by testing on a small limit.



Step	Time	Resource	Bottleneck
1. Scanner	1-5 min	I/O + CPU	File reading, hash computation
2. Embedding	10-30 sec	GPU/MPS	Neural network (MAIN BOTTLENECK)
3. Grouping	<1 sec	CPU	Just math on vectors
4. UI	Instant	Memory	Display