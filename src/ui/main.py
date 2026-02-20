"""Command-line interface for review UI server."""

import argparse
import logging
from pathlib import Path
import sys

import uvicorn

from .app_v6 import app, load_data, get_paths


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
        description="Start the duplicate photo review UI"
    )
    parser.add_argument(
        "--scan-results",
        type=Path,
        default=Path("scan_for_embeddings.json"),
        help="Path to scan results JSON",
    )
    parser.add_argument(
        "--similar-groups",
        type=Path,
        default=Path("embeddings/similar_groups.json"),
        help="Path to similar groups JSON",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("embeddings"),
        help="Path to embeddings directory",
    )
    parser.add_argument(
        "--path-mapping",
        type=Path,
        default=Path(".cache/path_mapping.json"),
        help="Path to mapping file (local paths -> original paths)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Start UI in demo mode (use demo dataset)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Resolve file paths: use get_paths() with the correct mode so the same
    # logic lives in one place (app_v6.get_paths) rather than being duplicated here.
    mode = "demo" if args.demo else "main"
    p = get_paths(mode)

    scan_results_path   = p.scan_results
    similar_groups_path = p.groups_file
    embeddings_dir_path = p.embeddings_dir
    path_mapping_path   = (p.scan_cache.parent / "path_mapping.json")
    path_mapping_path   = path_mapping_path if path_mapping_path.exists() else None

    if args.demo:
        logger.info(f"Demo mode: using {p.scan_results} and {p.embeddings_dir}/")
    
    # Validate files (only require similar_groups for main mode; demo may not have them yet)
    if not args.demo and not similar_groups_path.exists():
        logger.error(f"Similar groups file not found: {similar_groups_path}")
        logger.error("Run embedding generation first: python -m src.embedding.main ...")
        sys.exit(1)
    
    # Load data into app (support demo mode)
    load_data(
        scan_results_path,
        similar_groups_path,
        embeddings_dir_path,
        path_mapping_path,
        is_demo=args.demo,
    )
    
    # Start server
    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info" if args.verbose else "warning",
    )


if __name__ == "__main__":
    main()
