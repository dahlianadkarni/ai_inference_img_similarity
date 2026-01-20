"""Command-line interface for photo scanner."""

import argparse
import logging
from pathlib import Path
import sys
import tempfile
import shutil
import time
import json
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cache import ScanCache
from .image_utils import compute_md5

from .scanner import PhotoScanner, find_exact_duplicates, find_perceptual_duplicates
from .photos_library import (
    get_default_photos_library,
    get_originals_directory,
    validate_photos_library,
    count_photos_in_library,
)
from .applescript_photos import (
    get_photo_count,
    export_photos_simple,
    export_photos_simple_incremental,
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scan photos and find duplicates"
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        help="Directory containing photos to scan",
    )
    parser.add_argument(
        "--photos-library",
        action="store_true",
        help="Scan the default macOS Photos Library",
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        help="Path to specific Photos Library (.photoslibrary bundle)",
    )
    parser.add_argument(
        "--use-applescript",
        action="store_true",
        help="Use AppleScript to export photos (no Full Disk Access needed)",
    )
    parser.add_argument(
        "--direct-originals",
        action="store_true",
        help="Scan Photos Library originals directly (requires Full Disk Access)",
    )
    parser.add_argument(
        "--keep-export",
        action="store_true",
        help="Keep exported photos (don't clean up temp directory)",
    )
    parser.add_argument(
        "--applescript-export-dir",
        type=Path,
        default=Path(".cache/photos_export"),
        help="Persistent export directory used with --use-applescript --keep-export (default: .cache/photos_export)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate time and disk usage (does discovery + small sample processing, but does not write outputs)",
    )
    parser.add_argument(
        "--estimate-sample",
        type=int,
        default=200,
        help="Number of images to sample for throughput/size estimates (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scan_results.json"),
        help="Output file for metadata",
    )
    parser.add_argument(
        "--duplicates-output",
        type=Path,
        default=Path("scan_duplicates.json"),
        help="Output file for deterministic duplicate groups (MD5/dHash)",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path(".cache/scan_cache.json"),
        help="Path to incremental scan cache file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable scan caching (always rescan all files)",
    )
    parser.add_argument(
        "--md5-mode",
        type=str,
        default="on-demand",
        choices=["on-demand", "always", "never"],
        help="When to compute MD5 (default: on-demand computes MD5 only for same-size candidates)",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=5,
        help="Maximum Hamming distance for perceptual duplicates",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Determine which directory to scan
    scan_dir = None
    temp_export_dir = None
    
    if args.library_path:
        # Specific Photos Library provided
        if not validate_photos_library(args.library_path):
            logger.error(f"Invalid Photos Library: {args.library_path}")
            sys.exit(1)
        scan_dir = get_originals_directory(args.library_path)
        logger.info(f"Using Photos Library: {args.library_path}")
        photo_count = count_photos_in_library(args.library_path)
        logger.info(f"Estimated {photo_count} photos in library")
        
    elif args.photos_library:
        if args.use_applescript or not args.direct_originals:
            # Use AppleScript to export photos (default unless --direct-originals specified)
            logger.info("Using AppleScript to access Photos Library")
            logger.info("This requires Automation permission: Terminal → Photos")
            
            # Get photo count first
            total_photos = get_photo_count()
            if total_photos > 0:
                logger.info(f"Found {total_photos} photos in library")
                if args.limit:
                    logger.info(f"Will export {min(args.limit, total_photos)} photos")
            
            if args.keep_export:
                # Use a stable export directory so subsequent runs can be incremental.
                temp_export_dir = args.applescript_export_dir
                logger.info(f"Exporting (incremental) photos to: {temp_export_dir}")
                exported_files = export_photos_simple_incremental(
                    temp_export_dir,
                    limit=args.limit,
                )
            else:
                # Create temp directory for one-off export
                temp_export_dir = Path(tempfile.mkdtemp(prefix="photos_export_"))
                logger.info(f"Exporting photos to: {temp_export_dir}")
                exported_files = export_photos_simple(temp_export_dir, limit=args.limit)
            
            if len(exported_files) == 0:
                logger.error("No photos were exported")
                logger.error("Check Automation permissions: System Preferences > Privacy & Security > Automation")
                if temp_export_dir:
                    shutil.rmtree(temp_export_dir)
                sys.exit(1)
            
            scan_dir = temp_export_dir
            logger.info(f"Successfully exported {len(exported_files)} photos")
        else:
            # Use default Photos Library (requires Full Disk Access)
            library_path = get_default_photos_library()
            if library_path is None:
                logger.error("Default Photos Library not found at ~/Pictures/Photos Library.photoslibrary")
                sys.exit(1)
            if not validate_photos_library(library_path):
                logger.error(f"Invalid Photos Library: {library_path}")
                logger.error("Tip: Use --use-applescript to avoid needing Full Disk Access")
                sys.exit(1)
            scan_dir = get_originals_directory(library_path)
            logger.info(f"Using default Photos Library: {library_path}")
            photo_count = count_photos_in_library(library_path)
            logger.info(f"Estimated {photo_count} photos in library")
        
    elif args.directory:
        # Regular directory provided
        scan_dir = args.directory
        
    else:
        logger.error("Must provide either a directory, --photos-library, or --library-path")
        parser.print_help()
        sys.exit(1)
    
    # Validate directory
    if not scan_dir.exists():
        logger.error(f"Directory not found: {scan_dir}")
        sys.exit(1)
    
    # Create scanner
    scanner = PhotoScanner(max_workers=args.workers)

    if args.estimate:
        logger.info("Running estimate mode (no full scan will be performed)")
        logger.info(f"Target directory: {scan_dir}")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"MD5 mode: {args.md5_mode}")

        discovery_started = time.time()
        image_files = scanner.discover_images(scan_dir, recursive=True)
        discovery_elapsed = time.time() - discovery_started
        total_images = len(image_files)

        logger.info(f"Discovery: {total_images} supported images found in {discovery_elapsed:.1f}s")

        if total_images == 0:
            logger.warning("No images found to estimate")
            sys.exit(0)

        # If estimate_sample is 0, use all images; otherwise use the specified sample size
        if args.estimate_sample == 0:
            sample_n = total_images
            logger.info(f"Processing all {sample_n} images to estimate throughput (--estimate-sample 0)...")
        else:
            sample_n = min(args.estimate_sample, total_images)
            logger.info(f"Sampling {sample_n} images to estimate throughput...")

        sample_paths = image_files[:sample_n]

        processed = 0
        failed = 0
        sample_metadata = []

        sample_started = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    scanner.process_image,
                    file_path,
                    (args.md5_mode == "always"),
                    True,
                ): file_path
                for file_path in sample_paths
            }

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    failed += 1
                    continue
                image, meta = result
                try:
                    image.close()
                except Exception:
                    pass
                sample_metadata.append(meta)
                processed += 1

        sample_elapsed = time.time() - sample_started
        rate = (processed / sample_elapsed) if sample_elapsed > 0 else 0.0

        if processed == 0 or rate <= 0:
            logger.error("Sample processing produced no successful results; cannot estimate")
            sys.exit(1)

        est_scan_seconds = total_images / rate
        # Provide a loose range because image types/sizes vary widely.
        est_low = est_scan_seconds * 0.75
        est_high = est_scan_seconds * 1.50

        def _human_time(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}s"
            minutes = seconds / 60
            if minutes < 60:
                return f"{minutes:.1f}m"
            hours = minutes / 60
            return f"{hours:.1f}h"

        logger.info(
            f"Throughput (sample): {rate:.1f} images/sec ({processed} ok, {failed} failed, {sample_elapsed:.1f}s)"
        )
        logger.info(
            f"Estimated full scan time (no embeddings): {_human_time(est_low)} – {_human_time(est_high)} (based on sample)"
        )

        # Disk usage estimates (scan artifacts only)
        # - scan cache: .cache/scan_cache.json (only if caching enabled)
        # - scan results: scan_for_embeddings.json
        # - duplicates report: scan_duplicates.json (small)
        if sample_metadata:
            per_cache_bytes = []
            per_results_bytes = []
            for meta in sample_metadata:
                # Approximate cache entry payload (ScanCache.update_from_metadata normalizes datetimes)
                entry = asdict(meta)
                entry["created_at"] = None
                entry["modified_at"] = None
                entry["mtime"] = 0.0
                entry["file_size"] = meta.file_size
                per_cache_bytes.append(len(json.dumps(entry, separators=(",", ":"), ensure_ascii=False)))

                # Approximate scan_for_embeddings.json entry payload
                per_results_bytes.append(
                    len(json.dumps(asdict(meta), default=str, separators=(",", ":"), ensure_ascii=False))
                )

            avg_cache_entry = sum(per_cache_bytes) / max(len(per_cache_bytes), 1)
            avg_results_entry = sum(per_results_bytes) / max(len(per_results_bytes), 1)

            # Account for pretty-printing (indent=2) + JSON list/dict overhead.
            pretty_overhead_factor = 1.6
            est_cache_total = int(avg_cache_entry * total_images * pretty_overhead_factor)
            est_results_total = int(avg_results_entry * total_images * pretty_overhead_factor)

            def _human_bytes(num: int) -> str:
                kb = 1024
                mb = kb * 1024
                gb = mb * 1024
                if num >= gb:
                    return f"{num / gb:.2f} GB"
                if num >= mb:
                    return f"{num / mb:.1f} MB"
                if num >= kb:
                    return f"{num / kb:.1f} KB"
                return f"{num} B"

            logger.info("Estimated disk usage (scan phase only):")
            if args.no_cache:
                logger.info("  - Scan cache: disabled (--no-cache)")
            else:
                logger.info(f"  - Scan cache: ~{_human_bytes(est_cache_total)} at {args.cache_file}")
            logger.info(f"  - Scan output: ~{_human_bytes(est_results_total)} at {args.output}")
            logger.info(f"  - Duplicate report: typically small at {args.duplicates_output}")

            if args.photos_library and not args.use_applescript:
                logger.info("Note: Direct originals scan does not copy photos; it reads in-place.")
            if args.photos_library and args.use_applescript:
                logger.info(
                    "Note: AppleScript export can occupy significant disk space because it copies media into the export directory."
                )

        sys.exit(0)

    # Load cache (optional)
    cache = None
    if not args.no_cache:
        cache = ScanCache(args.cache_file)
        cache.load()
    
    # Scan directory
    logger.info(f"Scanning directory: {scan_dir}")
    metadata_list = []

    scan_started = time.time()
    progress_interval = 100
    for _image, metadata in scanner.scan_directory(
        scan_dir,
        recursive=True,
        limit=args.limit,
        cache=cache,
        compute_md5_hash=(args.md5_mode == "always"),
        compute_perceptual=True,
    ):
        metadata_list.append(metadata)
        if len(metadata_list) % progress_interval == 0:
            elapsed = time.time() - scan_started
            rate = len(metadata_list) / elapsed if elapsed > 0 else 0
            logger.info(f"Progress: {len(metadata_list)} images processed ({rate:.1f} images/sec)")


    scan_elapsed = time.time() - scan_started
    logger.info(f"Scan completed in {scan_elapsed:.1f}s ({len(metadata_list)} images)")

    if cache is not None:
        cache.save()
        logger.info(
            f"Cache updated: {cache.entry_count()} entries, {cache.size_bytes() / (1024 * 1024):.1f} MB at {args.cache_file}"
        )
    
    logger.info(f"Processed {len(metadata_list)} images")
    
    # Export metadata
    scanner.export_metadata(metadata_list, args.output)
    
    # Compute MD5 hashes if needed (on-demand)
    if args.md5_mode == "on-demand":
        size_to_indices = {}
        for idx, meta in enumerate(metadata_list):
            size_to_indices.setdefault(meta.file_size, []).append(idx)

        candidates = [idx for indices in size_to_indices.values() if len(indices) > 1 for idx in indices]
        if candidates:
            logger.info(f"Computing MD5 for {len(candidates)} same-size candidate files...")
        for idx in candidates:
            meta = metadata_list[idx]
            if not meta.md5_hash:
                try:
                    md5 = compute_md5(Path(meta.file_path))
                    meta.md5_hash = md5
                    if cache is not None:
                        cache.update_md5(Path(meta.file_path), md5)
                except Exception as e:
                    logger.warning(f"Failed to compute MD5 for {meta.file_path}: {e}")

        if cache is not None and candidates:
            cache.save()

    # Find duplicates
    exact_dupes = []
    if args.md5_mode != "never":
        exact_dupes = find_exact_duplicates(metadata_list)
    if exact_dupes:
        logger.info(f"\nFound {len(exact_dupes)} exact duplicate groups:")
        for i, group in enumerate(exact_dupes[:5], 1):  # Show first 5
            logger.info(f"  Group {i}: {len(group)} files")
            for path in group[:3]:  # Show first 3 in group
                logger.info(f"    - {Path(path).name}")
    
    perceptual_dupes = find_perceptual_duplicates(
        metadata_list,
        max_distance=args.max_distance,
    )
    if perceptual_dupes:
        logger.info(f"\nFound {len(perceptual_dupes)} perceptual duplicate groups:")
        for i, group in enumerate(perceptual_dupes[:5], 1):  # Show first 5
            logger.info(f"  Group {i}: {len(group)} files")
            for path in group[:3]:  # Show first 3 in group
                logger.info(f"    - {Path(path).name}")
    
    # Write deterministic duplicate report for UI / comparison
    try:
        report = {
            "scan_results": str(args.output),
            "photos_source": "photos-library" if args.photos_library or args.library_path else "directory",
            "used_applescript": bool(args.use_applescript),
            "max_distance": args.max_distance,
            "md5_mode": args.md5_mode,
            "image_count": len(metadata_list),
            "exact_groups": [{"size": len(g), "files": g} for g in exact_dupes],
            "perceptual_groups": [{"size": len(g), "files": g} for g in perceptual_dupes],
        }
        with open(args.duplicates_output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote deterministic duplicate report to {args.duplicates_output}")
    except Exception as e:
        logger.warning(f"Failed to write duplicates report: {e}")

    logger.info(f"\nResults saved to {args.output}")
    logger.info("Next: Run embedding generation to find more subtle duplicates")
    
    # Cleanup temp directory if we created one
    if temp_export_dir and temp_export_dir.exists() and not args.keep_export:
        logger.info(f"Cleaning up temporary export directory: {temp_export_dir}")
        shutil.rmtree(temp_export_dir)
    elif temp_export_dir and args.keep_export:
        logger.info(f"Keeping exported photos in: {temp_export_dir}")


if __name__ == "__main__":
    main()
