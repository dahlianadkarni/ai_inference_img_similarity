#!/usr/bin/env python3
"""Fetch photo dates from Photos app via AppleScript and update scan results."""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Parse AppleScript date format: "Wednesday, August 19, 2015 at 5:05:23 PM"
def parse_applescript_date(date_str):
    """Parse AppleScript date format to ISO format."""
    try:
        # Remove day of week
        parts = date_str.split(', ', 1)
        if len(parts) == 2:
            date_str = parts[1]
        
        # Parse: "August 19, 2015 at 5:05:23 PM"
        dt = datetime.strptime(date_str, "%B %d, %Y at %I:%M:%S %p")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Failed to parse date '{date_str}': {e}")
        return None


print("Fetching photo dates from Photos app...")
print("This may take a few minutes for large libraries...")

# Get all photos with their IDs, filenames, and dates
# We'll use the id property to create a stable identifier
script = '''
tell application "Photos"
    set itemList to media items
    set dataList to {}
    repeat with theItem in itemList
        set itemId to id of theItem
        set itemName to filename of theItem
        set itemDate to date of theItem as string
        set end of dataList to itemId & tab & itemName & tab & itemDate
    end repeat
    set AppleScript's text item delimiters to linefeed
    return dataList as text
end tell
'''

try:
    result = subprocess.run(
        ['osascript', '-e', script],
        capture_output=True,
        text=True,
        check=True,
        timeout=300
    )
    
    # Parse the results and save to a mapping file
    photo_metadata = []
    for line in result.stdout.strip().split('\n'):
        if line.count('\t') >= 2:
            parts = line.split('\t')
            photo_id = parts[0]
            filename = parts[1]
            date_str = '\t'.join(parts[2:])  # Date might contain tabs
            parsed_date = parse_applescript_date(date_str)
            if parsed_date:
                photo_metadata.append({
                    'id': photo_id,
                    'filename': filename,
                    'date': parsed_date
                })
    
    print(f"Fetched dates for {len(photo_metadata)} photos")
    
    # Save the metadata mapping
    metadata_file = Path('.cache/photos_metadata.json')
    metadata_file.parent.mkdir(exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(photo_metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Try to match by filename (best effort)
    # Create mapping: original_filename -> date
    filename_to_date = {item['filename']: item['date'] for item in photo_metadata}
    
    # Also create UUID-based mapping by checking actual exported files
    # The exported files have UUID names, so we'll match those to original filenames
    # by checking what Photos exported
    print("\nNote: Exported files use UUID names, not original filenames.")
    print("To get accurate dates, we need to map UUIDs to Photos metadata.")
    print("This requires the photos_metadata.json file for future scans.")
    
    # Update scan results
    scan_files = [
        'scan_results.json',
        'scan_for_embeddings.json'
    ]
    
    for scan_file in scan_files:
        scan_path = Path(scan_file)
        if not scan_path.exists():
            print(f"Skipping {scan_file} (not found)")
            continue
        
        print(f"\nUpdating {scan_file} (trying UUID extraction from Photos ID)...")
        with open(scan_path) as f:
            scan_data = json.load(f)
        
        updated = 0
        for item in tqdm(scan_data, desc="Updating dates"):
            # Extract UUID from filepath
            filepath = Path(item['file_path'])
            uuid_name = filepath.stem  # e.g., "9DD1E013-A7B1-4072-949E-AC583DE5493A"
            
            # Try to find matching photo by checking if UUID is in the Photos ID
            for photo in photo_metadata:
                photo_id = photo['id']
                # Photos IDs often contain the UUID
                if uuid_name in photo_id or photo_id.endswith(uuid_name):
                    item['photo_date'] = photo['date']
                    updated += 1
                    break
        
        # Save updated data
        with open(scan_path, 'w') as f:
            json.dump(scan_data, f, indent=2, default=str)
        
        print(f"Updated {updated}/{len(scan_data)} records in {scan_file}")
    
    print("\n✅ Done! Photo dates updated from Photos app.")
    print("Restart the UI to see the correct dates.")
    
except subprocess.CalledProcessError as e:
    print(f"Error running AppleScript: {e.stderr}")
except Exception as e:
    print(f"Error: {e}")
