"""Image metadata and preprocessing utilities."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import imagehash
from PIL import Image, ExifTags

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC files won't be supported


@dataclass
class ImageMetadata:
    """Metadata for a single image."""
    
    file_path: str
    file_size: int
    width: int
    height: int
    format: str
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    photo_date: Optional[datetime] = None  # Actual photo capture date from EXIF
    
    # Hashes for quick duplicate detection
    md5_hash: Optional[str] = None
    perceptual_hash: Optional[str] = None
    
    # Camera metadata
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    
    # Flags
    is_screenshot: bool = False
    orientation: int = 1
    

def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of file for exact duplicate detection."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_perceptual_hash(image: Image.Image) -> str:
    """Compute perceptual hash for near-duplicate detection."""
    # Using difference hash (dHash) - fast and effective
    phash = imagehash.dhash(image, hash_size=16)
    return str(phash)


def extract_exif_data(image: Image.Image) -> dict:
    """Extract relevant EXIF metadata from image."""
    exif_data = {}
    
    try:
        exif = image._getexif()
        if exif is not None:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
                
            # Parse datetime fields
            for date_field in ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']:
                if date_field in exif_data:
                    try:
                        date_str = exif_data[date_field]
                        if isinstance(date_str, str):
                            # EXIF date format: "YYYY:MM:DD HH:MM:SS"
                            exif_data[f'{date_field}_parsed'] = datetime.strptime(
                                date_str, "%Y:%m:%d %H:%M:%S"
                            )
                    except (ValueError, TypeError):
                        pass
    except (AttributeError, KeyError):
        pass
    
    return exif_data


def load_image_with_orientation(
    file_path: Path,
    max_size: int = 512,
    compute_md5_hash: bool = True,
    compute_perceptual: bool = True,
) -> Tuple[Image.Image, ImageMetadata]:
    """
    Load image, handle EXIF orientation, and extract metadata.
    
    Args:
        file_path: Path to image file
        max_size: Maximum dimension for thumbnail (for efficiency)
    
    Returns:
        Tuple of (processed_image, metadata)
    """
    # Load image
    image = Image.open(file_path)
    
    # Extract EXIF data
    exif_data = extract_exif_data(image)
    
    # Handle orientation
    orientation = exif_data.get("Orientation", 1)
    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(270, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    
    # Convert to RGB if needed
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    
    # Create thumbnail for efficient processing
    original_size = image.size
    if max(original_size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Extract file metadata
    stat = file_path.stat()
    
    # Get actual photo date from EXIF (prefer DateTimeOriginal, fall back to others)
    photo_date = None
    for field in ['DateTimeOriginal_parsed', 'DateTimeDigitized_parsed', 'DateTime_parsed']:
        if field in exif_data:
            photo_date = exif_data[field]
            break
    
    # Build metadata object
    metadata = ImageMetadata(
        file_path=str(file_path),
        file_size=stat.st_size,
        width=original_size[0],
        height=original_size[1],
        format=image.format or "UNKNOWN",
        created_at=datetime.fromtimestamp(stat.st_birthtime) if hasattr(stat, 'st_birthtime') else None,
        modified_at=datetime.fromtimestamp(stat.st_mtime),
        photo_date=photo_date,  # Actual photo capture date
        md5_hash=compute_md5(file_path) if compute_md5_hash else None,
        perceptual_hash=compute_perceptual_hash(image) if compute_perceptual else None,
        camera_make=exif_data.get("Make"),
        camera_model=exif_data.get("Model"),
        is_screenshot="screenshot" in file_path.name.lower(),
        orientation=orientation,
    )
    
    return image, metadata


def is_supported_image(file_path: Path) -> bool:
    """Check if file is a supported image format."""
    supported_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tiff"}
    return file_path.suffix.lower() in supported_extensions
