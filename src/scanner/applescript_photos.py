"""AppleScript integration for accessing Photos Library without Full Disk Access."""

import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


def check_photos_app_running() -> bool:
    """Check if Photos.app is running."""
    result = subprocess.run(
        ['osascript', '-e', 'tell application "System Events" to (name of processes) contains "Photos"'],
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout.strip() == "true"


def get_photo_count() -> int:
    """Get total count of photos in Photos Library via AppleScript."""
    script = '''
    tell application "Photos"
        count media items
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Failed to get photo count: {e}")
        return 0


def export_photos_via_applescript(
    output_dir: Path,
    limit: Optional[int] = None,
) -> List[Path]:
    """
    Export photos from Photos Library to a temporary directory via AppleScript.
    
    Args:
        output_dir: Directory to export photos to
        limit: Maximum number of photos to export
    
    Returns:
        List of exported file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build AppleScript to export photos
    limit_clause = f"if item_count > {limit} then set item_count to {limit}" if limit else ""
    
    script = f'''
    tell application "Photos"
        set itemList to media items
        set item_count to count of itemList
        {limit_clause}
        
        set exported_paths to {{}}
        repeat with i from 1 to item_count
            set theItem to item i of itemList
            set itemName to filename of theItem
            set itemDate to date of theItem
            
            -- Export to output directory
            try
                set exportPath to POSIX file "{output_dir}"
                export {{theItem}} to exportPath
                set end of exported_paths to (POSIX path of exportPath) & itemName
            on error errMsg
                log "Export failed for " & itemName & ": " & errMsg
            end try
        end repeat
        
        return exported_paths
    end tell
    '''
    
    logger.info("Requesting photo export via Photos.app...")
    logger.info("This will prompt for Automation permission if not already granted")
    
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse exported paths from output
        exported_files = []
        for item in output_dir.iterdir():
            if item.is_file():
                exported_files.append(item)
        
        logger.info(f"Successfully exported {len(exported_files)} photos")
        return exported_files
        
    except subprocess.TimeoutExpired:
        logger.error("Photo export timed out after 5 minutes")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"AppleScript export failed: {e.stderr}")
        logger.error("Make sure Terminal has Automation permission for Photos")
        return []


def get_photo_metadata_via_applescript(limit: Optional[int] = None) -> List[dict]:
    """
    Get photo metadata from Photos Library via AppleScript.
    
    This is faster than export and useful for getting counts/metadata before export.
    
    Args:
        limit: Maximum number of photos to query
    
    Returns:
        List of metadata dictionaries
    """
    limit_clause = f"if item_count > {limit} then set item_count to {limit}" if limit else ""
    
    script = f'''
    tell application "Photos"
        set itemList to media items
        set item_count to count of itemList
        {limit_clause}
        
        set metadata to {{}}
        repeat with i from 1 to item_count
            set theItem to item i of itemList
            set itemData to {{}}
            set itemData's name to filename of theItem
            set itemData's width to width of theItem
            set itemData's height to height of theItem
            set itemData's size to size of theItem
            set itemData's itemDate to date of theItem as string
            set itemData's isFavorite to favorite of theItem
            set end of metadata to itemData
        end repeat
        
        return metadata
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
        
        # AppleScript returns comma-separated format, we'd need to parse
        # For now, just log success
        logger.info("Successfully retrieved metadata")
        return []
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.error(f"Failed to get metadata: {e}")
        return []


def export_photos_simple(
    output_dir: Path,
    limit: Optional[int] = None,
) -> List[Path]:
    """
    Simplified export using Photos export command.
    
    This prompts the user via GUI but is more reliable than scripting.
    
    Args:
        output_dir: Directory to export photos to
        limit: Maximum number of photos to export
    
    Returns:
        List of exported file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    limit_clause = f"if item_count > {limit} then set item_count to {limit}" if limit else ""
    
    # Simplified script that just gets photo IDs
    script = f'''
    tell application "Photos"
        set itemList to media items
        set item_count to count of itemList
        {limit_clause}
        
        set photo_ids to {{}}
        repeat with i from 1 to item_count
            set theItem to item i of itemList
            try
                export {{theItem}} to POSIX file "{output_dir}"
            end try
        end repeat
    end tell
    '''
    
    logger.info(f"Exporting up to {limit or 'all'} photos to {output_dir}")
    logger.info("This may take a moment...")
    
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 minute timeout for large exports
        )
        
        # Find all exported files
        exported_files = [f for f in output_dir.iterdir() if f.is_file()]
        logger.info(f"Exported {len(exported_files)} photos")
        return exported_files
        
    except subprocess.TimeoutExpired:
        logger.error("Export timed out")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed: {e.stderr}")
        if "not authorized" in e.stderr.lower() or "permission" in e.stderr.lower():
            logger.error("\n⚠️  Terminal needs Automation permission for Photos!")
            logger.error("Go to: System Preferences > Privacy & Security > Automation")
            logger.error("Enable: Terminal → Photos")
        return []


def export_photos_simple_incremental(
    output_dir: Path,
    limit: Optional[int] = None,
) -> List[Tuple[Path, str]]:
    """Export photos via AppleScript, skipping items already exported.
    
    Returns list of (file_path, date_string) tuples where date_string is the photo date from Photos.

    This uses the exported filename as the "already exported" signal.
    It works best when exporting to a persistent directory (not a temp dir).

    Notes:
    - This avoids re-exporting most of your library on subsequent runs.
    - It is heuristic: filename collisions can cause false skips.
    - Now also captures and returns the photo date from Photos metadata
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Here, limit means "export at most N *new* photos".
    limit_clause = "" if limit is None else f"if exported_count ≥ {limit} then exit repeat"

    # Export and capture dates - returns TSV format: filepath<tab>date
    script = f'''
    set exportPath to POSIX file "{output_dir}" as alias
    set exported_count to 0
    set dateList to {{}}

    tell application "Photos"
        set itemList to media items
        repeat with theItem in itemList
            set itemName to filename of theItem
            set itemDate to date of theItem as string

            set alreadyExported to false
            try
                tell application "System Events"
                    set alreadyExported to exists file itemName of folder exportPath
                end tell
            end try

            if alreadyExported is false then
                try
                    export {{theItem}} to exportPath
                    set exported_count to exported_count + 1
                end try
            end if
            
            -- Record date for all files (new or existing)
            set end of dateList to (POSIX path of exportPath) & itemName & tab & itemDate

            {limit_clause}
        end repeat
    end tell
    
    -- Return dates as newline-separated list
    set AppleScript's text item delimiters to linefeed
    return dateList as text
    '''

    logger.info(f"Incremental export to {output_dir} (skipping already-exported filenames, capturing dates)")

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        
        # Parse the output to get file -> date mapping
        file_dates = {}
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if '\t' in line:
                    filepath, date_str = line.split('\t', 1)
                    file_dates[filepath] = date_str
        
        exported_files = []
        for f in output_dir.iterdir():
            if f.is_file():
                date_str = file_dates.get(str(f), None)
                exported_files.append((f, date_str))
        
        logger.info(f"Export directory now contains {len(exported_files)} files with dates")
        return exported_files
    except subprocess.TimeoutExpired:
        logger.error("Incremental export timed out")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"Incremental export failed: {e.stderr}")
        if "not authorized" in e.stderr.lower() or "permission" in e.stderr.lower():
            logger.error("\n⚠️  Terminal needs Automation permission for Photos!")
            logger.error("Go to: System Preferences > Privacy & Security > Automation")
            logger.error("Enable: Terminal → Photos")
        return []
