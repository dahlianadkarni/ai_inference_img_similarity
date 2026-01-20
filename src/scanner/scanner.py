"""Photo scanner for discovering and processing images."""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json

from PIL import Image
from tqdm import tqdm

from .image_utils import (
    ImageMetadata,
    load_image_with_orientation,
    is_supported_image,
)

from .cache import ScanCache


logger = logging.getLogger(__name__)


class PhotoScanner:
    """Scans directories for images and extracts metadata."""
    
    def __init__(self, max_workers: int = 4, thumbnail_size: int = 512):
        """
        Initialize scanner.
        
        Args:
            max_workers: Number of parallel workers for processing
            thumbnail_size: Maximum dimension for image thumbnails
        """
        self.max_workers = max_workers
        self.thumbnail_size = thumbnail_size
    
    def discover_images(self, directory: Path, recursive: bool = True) -> List[Path]:
        """
        Discover all supported images in directory.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
        
        Returns:
            List of image file paths
        """
        pattern = "**/*" if recursive else "*"
        all_files = directory.glob(pattern)
        
        image_files = [
            f for f in all_files
            if f.is_file() and is_supported_image(f)
        ]
        
        logger.info(f"Discovered {len(image_files)} images in {directory}")
        
        if len(image_files) == 0:
            logger.warning(f"No supported images found in {directory}")
            logger.warning("Supported formats: .jpg, .jpeg, .png, .heic, .heif, .webp, .bmp, .tiff")
            logger.warning("Tip: Use --photos-library to scan your Photos Library")
        
        return image_files
    
    def process_image(
        self,
        file_path: Path,
        compute_md5_hash: bool = False,
        compute_perceptual: bool = True,
    ) -> Optional[Tuple[Image.Image, ImageMetadata]]:
        """
        Process a single image file.
        
        Args:
            file_path: Path to image
        
        Returns:
            Tuple of (image, metadata) or None if processing fails
        """
        try:
            image, metadata = load_image_with_orientation(
                file_path, 
                max_size=self.thumbnail_size,
                compute_md5_hash=compute_md5_hash,
                compute_perceptual=compute_perceptual,
            )
            return image, metadata
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            return None
    
    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        limit: Optional[int] = None,
        cache: Optional[ScanCache] = None,
        compute_md5_hash: bool = False,
        compute_perceptual: bool = True,
    ) -> Iterator[Tuple[Optional[Image.Image], ImageMetadata]]:
        """
        Scan directory and yield processed images with metadata.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            limit: Maximum number of images to process (for testing)
        
        Yields:
            Tuples of (image, metadata)
        """
        image_files = self.discover_images(directory, recursive)
        
        if limit:
            image_files = image_files[:limit]
            logger.info(f"Limiting to {limit} images")

        # If we have a cache, use it to skip unchanged files
        cached_metadata: List[ImageMetadata] = []
        to_process: List[Path] = []

        if cache is not None:
            for file_path in image_files:
                if cache.is_fresh(file_path):
                    meta = cache.to_metadata(file_path)
                    if meta is not None:
                        cached_metadata.append(meta)
                        continue
                to_process.append(file_path)
        else:
            to_process = image_files
        
        total = len(image_files)
        with tqdm(total=total, desc="Scanning images") as pbar:
            # Yield cached results first
            for meta in cached_metadata:
                yield None, meta
                pbar.update(1)

            # Process remaining files in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_image,
                        file_path,
                        compute_md5_hash,
                        compute_perceptual,
                    ): file_path
                    for file_path in to_process
                }

                for future in as_completed(futures):
                    file_path = futures[future]
                    result = future.result()
                    if result is not None:
                        image, meta = result
                        if cache is not None:
                            cache.update_from_metadata(file_path, meta)
                        yield image, meta
                    pbar.update(1)
    
    def export_metadata(self, metadata_list: List[ImageMetadata], output_path: Path):
        """
        Export metadata to JSON file.
        
        Args:
            metadata_list: List of image metadata
            output_path: Path to output JSON file
        """
        data = [asdict(m) for m in metadata_list]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported metadata for {len(metadata_list)} images to {output_path}")


def find_exact_duplicates(metadata_list: List[ImageMetadata]) -> List[List[str]]:
    """
    Find exact duplicates using MD5 hash.
    
    Args:
        metadata_list: List of image metadata
    
    Returns:
        List of duplicate groups (each group is a list of file paths)
    """
    hash_to_files = {}
    
    for meta in metadata_list:
        if meta.md5_hash:
            hash_to_files.setdefault(meta.md5_hash, []).append(meta.file_path)
    
    # Return only groups with duplicates
    duplicates = [files for files in hash_to_files.values() if len(files) > 1]
    
    logger.info(f"Found {len(duplicates)} exact duplicate groups")
    return duplicates


def find_perceptual_duplicates(
    metadata_list: List[ImageMetadata],
    max_distance: int = 5,
) -> List[List[str]]:
    """
    Find near-duplicates using perceptual hash.
    
    Args:
        metadata_list: List of image metadata
        max_distance: Maximum Hamming distance for duplicates
    
    Returns:
        List of duplicate groups
    """
    import imagehash
    
    # Build hash lookup
    hash_to_files = {}
    for meta in metadata_list:
        if meta.perceptual_hash:
            hash_to_files.setdefault(meta.perceptual_hash, []).append(meta.file_path)
    
    # Find similar hashes using brute force (good enough for moderate collections)
    duplicates = []
    processed = set()
    
    hashes = list(hash_to_files.keys())
    for i, hash1 in enumerate(tqdm(hashes, desc="Finding perceptual duplicates")):
        if hash1 in processed:
            continue
        
        group = list(hash_to_files[hash1])
        processed.add(hash1)
        
        # Compare with remaining hashes
        for hash2 in hashes[i + 1:]:
            if hash2 in processed:
                continue
            
            # Compute Hamming distance
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            distance = h1 - h2
            
            if distance <= max_distance:
                group.extend(hash_to_files[hash2])
                processed.add(hash2)
        
        if len(group) > 1:
            duplicates.append(group)
    
    logger.info(f"Found {len(duplicates)} perceptual duplicate groups")
    return duplicates
