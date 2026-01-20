"""Hybrid scan: local originals + AppleScript for iCloud-only photos."""

import argparse
import logging
import json
from pathlib import Path
from typing import Set

from .scanner import PhotoScanner
from .cache import ScanCache
from .photos_library import get_default_photos_library, get_originals_directory
from .applescript_photos import export_photos_simple_incremental, get_photo_count

logger = logging.getLogger(__name__)


def get_cached_filenames(cache: ScanCache) -> Set[str]:
    """Extract filenames from cache entries."""
    filenames = set()
    for entry in cache._data.get("entries", {}).values():
        file_path = Path(entry["file_path"])
        filenames.add(file_path.name.lower())
    return filenames


def main():
    """Hybrid scan: local originals + AppleScript for missing photos."""
    parser = argparse.ArgumentParser(
        description="Hybrid scan: fast local scan + AppleScript for iCloud photos"
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path(".cache/scan_cache.json"),
        help="Path to scan cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scan_results.json"),
        help="Output file for combined metadata",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path(".cache/photos_export"),
        help="AppleScript export directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--md5-mode",
        type=str,
        default="on-demand",
        choices=["on-demand", "always", "never"],
        help="When to compute MD5",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate time and disk for AppleScript export/scan (does not run full scan)",
    )
    parser.add_argument(
        "--estimate-sample",
        type=int,
        default=50,
        help="Number of photos to sample for estimate (default: 50)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load cache
    cache = ScanCache(args.cache_file)
    cache.load()
    logger.info(f"Loaded cache with {cache.entry_count()} entries")
    
    # Get Photos Library info
    logger.info("Checking Photos Library via AppleScript...")
    total_in_photos = get_photo_count()
    logger.info(f"Photos app reports {total_in_photos} total photos")
    
    cached_filenames = get_cached_filenames(cache)
    logger.info(f"Already have {len(cached_filenames)} unique filenames in cache")
    
    missing_count = total_in_photos - len(cached_filenames)
    
    if args.estimate:
        logger.info("Running estimate mode for hybrid scan")
        
        if missing_count <= 0:
            logger.info("All photos already cached! No AppleScript export needed.")
            logger.info("Re-run without --estimate to regenerate outputs if needed.")
            return
        
        logger.info(f"Estimated {missing_count} photos need AppleScript export")
        
        # Sample export to measure throughput
        import time
        import tempfile
        sample_n = min(args.estimate_sample, missing_count)
        
        logger.info(f"Sampling {sample_n} photo exports to estimate throughput...")
        temp_dir = Path(tempfile.mkdtemp(prefix="estimate_export_"))
        
        try:
            start = time.time()
            exported = export_photos_simple_incremental(temp_dir, limit=sample_n)
            export_elapsed = time.time() - start
            
            if len(exported) == 0:
                logger.error("No photos exported in sample. Check Automation permissions.")
                return
            
            # Measure scan throughput on exported sample
            scanner = PhotoScanner(max_workers=args.workers)
            sample_scanned = 0
            scan_start = time.time()
            
            for _, _ in scanner.scan_directory(
                temp_dir,
                recursive=False,
                cache=None,
                compute_md5_hash=(args.md5_mode == "always"),
                compute_perceptual=True,
            ):
                sample_scanned += 1
            
            scan_elapsed = time.time() - scan_start
            
            export_rate = len(exported) / export_elapsed if export_elapsed > 0 else 0
            scan_rate = sample_scanned / scan_elapsed if scan_elapsed > 0 else 0
            
            # Export is typically the bottleneck
            total_export_time = missing_count / export_rate if export_rate > 0 else 0
            total_scan_time = missing_count / scan_rate if scan_rate > 0 else 0
            
            # Total time is dominated by slower operation (usually export)
            est_time = max(total_export_time, total_scan_time)
            est_low = est_time * 0.75
            est_high = est_time * 1.5
            
            def human_time(seconds):
                if seconds < 60:
                    return f"{seconds:.0f}s"
                minutes = seconds / 60
                if minutes < 60:
                    return f"{minutes:.1f}m"
                hours = minutes / 60
                return f"{hours:.1f}h"
            
            logger.info(f"AppleScript export: {export_rate:.1f} photos/sec ({len(exported)} in {export_elapsed:.1f}s)")
            logger.info(f"Scan throughput: {scan_rate:.1f} photos/sec ({sample_scanned} in {scan_elapsed:.1f}s)")
            logger.info(f"Estimated hybrid scan time: {human_time(est_low)} – {human_time(est_high)}")
            
            # Disk usage estimate
            avg_file_size = sum((temp_dir / f.name).stat().st_size for f in exported) / len(exported)
            export_disk = missing_count * avg_file_size
            
            def human_bytes(num):
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if num < 1024:
                        return f"{num:.1f} {unit}"
                    num /= 1024
                return f"{num:.1f} PB"
            
            logger.info(f"Estimated disk for exported photos: {human_bytes(export_disk)}")
            logger.info(f"Export location: {args.export_dir}")
            logger.info("Note: Use --keep-export to reuse exports; without it, they'll be cleaned up")
            
        finally:
            # Cleanup sample
            import shutil
            shutil.rmtree(temp_dir)
        
        return
    
    # Step 1: Scan local originals
    logger.info("Step 1: Scanning local originals...")
    library_path = get_default_photos_library()
    if not library_path:
        logger.error("Photos Library not found")
        return
    
    originals_dir = get_originals_directory(library_path)
    scanner = PhotoScanner(max_workers=args.workers)
    
    metadata_list = []
    for _, metadata in scanner.scan_directory(
        originals_dir,
        recursive=True,
        cache=cache,
        compute_md5_hash=(args.md5_mode == "always"),
        compute_perceptual=True,
    ):
        metadata_list.append(metadata)
    
    logger.info(f"Scanned {len(metadata_list)} local originals")
    cache.save()
    
    # Step 2: Check what's missing via AppleScript
    logger.info("Step 2: Checking Photos Library via AppleScript...")
    total_in_photos = get_photo_count()
    logger.info(f"Photos app reports {total_in_photos} total photos")
    
    cached_filenames = get_cached_filenames(cache)
    logger.info(f"Already have {len(cached_filenames)} unique filenames in cache")
    
    missing_count = total_in_photos - len(cached_filenames)
    if missing_count <= 0:
        logger.info("All photos already scanned!")
    else:
        logger.info(f"Estimated {missing_count} photos need AppleScript export")
        logger.info("Step 3: Exporting missing photos via AppleScript...")
        logger.info("(This will skip files already in export directory)")
        
        # Export incrementally (skips existing files)
        exported_files = export_photos_simple_incremental(
            args.export_dir,
            limit=None,  # Export all
        )
        logger.info(f"Exported {len(exported_files)} files")
        
        # Scan exported files, but only those not already in cache by filename
        logger.info("Step 4: Scanning newly exported photos...")
        new_scans = 0
        for _, metadata in scanner.scan_directory(
            args.export_dir,
            recursive=False,
            cache=cache,
            compute_md5_hash=(args.md5_mode == "always"),
            compute_perceptual=True,
        ):
            # Check if this filename was already in originals scan
            filename = Path(metadata.file_path).name.lower()
            if filename not in cached_filenames:
                metadata_list.append(metadata)
                new_scans += 1
        
        logger.info(f"Added {new_scans} new photos from AppleScript export")
        cache.save()
    
    # Export combined results
    logger.info(f"Total photos scanned: {len(metadata_list)}")
    scanner.export_metadata(metadata_list, args.output)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
