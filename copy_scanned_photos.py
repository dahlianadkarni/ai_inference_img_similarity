#!/usr/bin/env python3
"""Copy scanned photos to local directory and update paths."""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
scan_file = Path("scan_results.json")
output_file = Path("scan_for_embeddings.json")
duplicates_file = Path("scan_duplicates.json")
mapping_file = Path(".cache/path_mapping.json")
local_dir = Path(".cache/photos_local")

# Load scan results
print(f"Loading {scan_file}...")
with open(scan_file) as f:
    scan_data = json.load(f)

print(f"Found {len(scan_data)} images")

# Copy files and update paths
local_dir.mkdir(parents=True, exist_ok=True)
updated_data = []
copied = 0
skipped = 0

# Build path mapping (originals -> local)
path_mapping = {}
reverse_mapping = {}  # For updating duplicates file

for item in tqdm(scan_data, desc="Copying photos"):
    src_path = Path(item["file_path"])
    filename = src_path.name
    dest_path = local_dir / filename
    
    # Store mapping (local -> original for delete operations)
    path_mapping[str(dest_path)] = str(src_path)
    # Store reverse (original -> local for updating duplicates)
    reverse_mapping[str(src_path)] = str(dest_path)
    
    # Copy if doesn't exist
    if not dest_path.exists():
        try:
            shutil.copy2(src_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"\nWarning: Failed to copy {filename}: {e}")
            skipped += 1
            continue
    
    # Update path in metadata
    item["file_path"] = str(dest_path)
    updated_data.append(item)

# Save updated scan results
print(f"\nSaving updated scan to {output_file}...")
with open(output_file, "w") as f:
    json.dump(updated_data, f, indent=2, default=str)

# Update duplicates file if it exists
if duplicates_file.exists():
    print(f"Updating paths in {duplicates_file}...")
    with open(duplicates_file) as f:
        dup_data = json.load(f)
    
    # Update paths in exact groups (original -> local)
    for group in dup_data.get("exact_groups", []):
        group["files"] = [reverse_mapping.get(p, p) for p in group["files"]]
    
    # Update paths in perceptual groups (original -> local)
    for group in dup_data.get("perceptual_groups", []):
        group["files"] = [reverse_mapping.get(p, p) for p in group["files"]]
    
    with open(duplicates_file, "w") as f:
        json.dump(dup_data, f, indent=2)
    
    print(f"Updated {duplicates_file}")

# Save path mapping for delete operations
print(f"\nSaving path mapping to {mapping_file}...")
with open(mapping_file, "w") as f:
    json.dump(path_mapping, f, indent=2)

print(f"Done! Copied {copied} new files, {len(updated_data) - copied} already existed, {skipped} failed")
print(f"Total size: {sum(f.stat().st_size for f in local_dir.glob('*')) / (1024**3):.2f} GB")
print(f"\nUpdated scan results saved to {output_file}")
print(f"Path mapping saved to {mapping_file}")
if duplicates_file.exists():
    print(f"Updated duplicate paths in {duplicates_file}")
print("You can now view images in the UI without Full Disk Access!")
