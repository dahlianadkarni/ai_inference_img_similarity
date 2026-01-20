"""macOS Photos Library integration via direct file access."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_default_photos_library() -> Optional[Path]:
    """Get the default macOS Photos Library path."""
    default_path = Path.home() / "Pictures" / "Photos Library.photoslibrary"
    
    if default_path.exists():
        return default_path
    
    return None


def get_originals_directory(library_path: Path) -> Optional[Path]:
    """
    Get the originals directory from a Photos Library.
    
    Args:
        library_path: Path to .photoslibrary bundle
    
    Returns:
        Path to originals directory, or None if not found
    """
    originals = library_path / "originals"
    
    if originals.exists() and originals.is_dir():
        return originals
    
    logger.warning(f"Originals directory not found at {originals}")
    return None


def validate_photos_library(library_path: Path) -> bool:
    """
    Validate that a path is a valid Photos Library.
    
    Args:
        library_path: Path to validate
    
    Returns:
        True if valid Photos Library
    """
    if not library_path.exists():
        logger.error(f"Path does not exist: {library_path}")
        return False
    
    if not library_path.is_dir():
        logger.error(f"Not a directory: {library_path}")
        return False
    
    if not library_path.name.endswith(".photoslibrary"):
        logger.warning(f"Path does not end with .photoslibrary: {library_path}")
    
    originals = get_originals_directory(library_path)
    if originals is None:
        return False
    
    return True


def count_photos_in_library(library_path: Path) -> int:
    """
    Count photos in a Photos Library.
    
    Args:
        library_path: Path to .photoslibrary bundle
    
    Returns:
        Approximate count of photo files
    """
    originals = get_originals_directory(library_path)
    if originals is None:
        return 0
    
    # Quick count - just count files (not precise but fast)
    count = sum(1 for _ in originals.rglob("*") if _.is_file())
    return count
