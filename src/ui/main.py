"""Command-line interface for review UI server."""

import argparse
import logging
from pathlib import Path
import sys

import uvicorn

from .app_v3 import app, load_data


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
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate files
    if not args.similar_groups.exists():
        logger.error(f"Similar groups file not found: {args.similar_groups}")
        logger.error("Run embedding generation first: python -m src.embedding.main ...")
        sys.exit(1)
    
    # Load data into app
    load_data(
        args.scan_results, 
        args.similar_groups, 
        args.embeddings_dir,
        args.path_mapping if args.path_mapping.exists() else None
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
