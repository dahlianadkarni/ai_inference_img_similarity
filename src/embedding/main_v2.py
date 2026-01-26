"""Embedding generation with support for both local and remote inference.

This allows the app to work in two modes:
1. Local: Embedding generation runs inline (original behavior)
2. Remote: Calls an inference service (new behavior for distributed setup)
"""

import argparse
import logging
from pathlib import Path
import sys
import json

import numpy as np
from tqdm import tqdm
from PIL import Image

from ..scanner.scanner import PhotoScanner
from .embedder import ImageEmbedder, compute_similarity_matrix
from .storage import EmbeddingStore
from ..grouping import find_similarity_groups, compute_group_similarities
from ..inference_service.client import InferenceClient


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def generate_embeddings_local(
    scan_data: list,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
) -> tuple:
    """
    Generate embeddings locally (original behavior).
    
    Args:
        scan_data: List of image metadata dicts
        model_name: CLIP model to use
        pretrained: Pretrained weights
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (embeddings, file_paths)
    """
    logger = logging.getLogger(__name__)
    
    # Initialize embedder
    logger.info("Initializing embedding model...")
    embedder = ImageEmbedder(
        model_name=model_name,
        pretrained=pretrained,
    )
    model_info = embedder.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Generate embeddings in chunks to avoid memory issues
    logger.info("Generating embeddings (LOCAL MODE)...")
    all_embeddings = []
    valid_file_paths = []
    chunk_size = 100
    
    for chunk_start in range(0, len(scan_data), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(scan_data))
        chunk_data = scan_data[chunk_start:chunk_end]
        
        # Load images for this chunk
        images = []
        chunk_paths = []
        
        for item in tqdm(
            chunk_data,
            desc=f"Loading images ({chunk_start+1}-{chunk_end}/{len(scan_data)})",
            leave=False
        ):
            file_path = Path(item["file_path"])
            
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path}")
                continue
            
            try:
                image = Image.open(file_path).convert("RGB")
                images.append(image)
                chunk_paths.append(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        if len(images) > 0:
            logger.info(
                f"Processing chunk {chunk_start//chunk_size + 1}/"
                f"{(len(scan_data) + chunk_size - 1)//chunk_size}: {len(images)} images"
            )
            chunk_embeddings = embedder.embed_images_batch(images, batch_size=batch_size)
            all_embeddings.append(chunk_embeddings)
            valid_file_paths.extend(chunk_paths)
            images.clear()
    
    if len(valid_file_paths) == 0:
        logger.error("No images could be loaded")
        return None, None
    
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Generated embeddings for {len(valid_file_paths)} images")
    
    return embeddings, valid_file_paths, model_info


def generate_embeddings_remote(
    scan_data: list,
    service_url: str = "http://127.0.0.1:8001",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
) -> tuple:
    """
    Generate embeddings via remote inference service.
    
    Args:
        scan_data: List of image metadata dicts
        service_url: URL of inference service
        model_name: CLIP model to use
        pretrained: Pretrained weights
        batch_size: Batch size for requests
    
    Returns:
        Tuple of (embeddings, file_paths, model_info)
    """
    logger = logging.getLogger(__name__)
    
    # Initialize client
    logger.info(f"Connecting to inference service at {service_url}...")
    client = InferenceClient(service_url=service_url)
    
    # Check health
    if not client.health_check():
        logger.error(f"Inference service not available at {service_url}")
        logger.error("Start the service with: python -m src.inference_service.server")
        return None, None, None
    
    logger.info("✓ Connected to inference service")
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Generate embeddings in batches
    logger.info("Generating embeddings (REMOTE MODE)...")
    all_embeddings = []
    valid_file_paths = []
    
    for batch_start in range(0, len(scan_data), batch_size):
        batch_end = min(batch_start + batch_size, len(scan_data))
        batch_data = scan_data[batch_start:batch_end]
        
        # Load images for this batch
        images = []
        batch_paths = []
        
        for item in tqdm(
            batch_data,
            desc=f"Loading images ({batch_start+1}-{batch_end}/{len(scan_data)})",
            leave=False
        ):
            file_path = Path(item["file_path"])
            
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path}")
                continue
            
            try:
                image = Image.open(file_path).convert("RGB")
                images.append(image)
                batch_paths.append(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        if len(images) > 0:
            logger.info(
                f"Sending batch {batch_start//batch_size + 1}/"
                f"{(len(scan_data) + batch_size - 1)//batch_size}: {len(images)} images"
            )
            try:
                batch_embeddings = client.embed_images_base64(
                    images,
                    model_name=model_name,
                    pretrained=pretrained,
                )
                all_embeddings.append(batch_embeddings)
                valid_file_paths.extend(batch_paths)
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                return None, None, None
            
            images.clear()
    
    if len(valid_file_paths) == 0:
        logger.error("No images could be loaded")
        return None, None, None
    
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Generated embeddings for {len(valid_file_paths)} images (via remote service)")
    
    return embeddings, valid_file_paths, model_info


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for scanned photos (local or remote)"
    )
    parser.add_argument(
        "scan_results",
        type=Path,
        help="Path to scan_results.json from scanner",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "remote", "auto"],
        default="auto",
        help="Inference mode: 'local' (inline), 'remote' (via service), or 'auto' (try remote, fallback to local)",
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default="http://127.0.0.1:8001",
        help="URL of inference service (for remote mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="CLIP model to use (default: ViT-B-32)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights (default: openai)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embeddings"),
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for finding duplicates (0-1, higher is more strict)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load scan results
    if not args.scan_results.exists():
        logger.error(f"Scan results not found: {args.scan_results}")
        logger.error("Run scanner first: python -m src.scanner.main ...")
        sys.exit(1)
    
    with open(args.scan_results, "r") as f:
        scan_data = json.load(f)
    
    logger.info(f"Loaded {len(scan_data)} images from scan results")
    
    # Generate embeddings
    embeddings = None
    file_paths = None
    model_info = None
    
    if args.mode == "remote":
        logger.info("Using REMOTE inference mode")
        embeddings, file_paths, model_info = generate_embeddings_remote(
            scan_data,
            service_url=args.service_url,
            model_name=args.model,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
        )
    elif args.mode == "local":
        logger.info("Using LOCAL inference mode")
        embeddings, file_paths, model_info = generate_embeddings_local(
            scan_data,
            model_name=args.model,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
        )
    else:  # auto
        logger.info("Using AUTO mode (trying remote, will fallback to local)")
        embeddings, file_paths, model_info = generate_embeddings_remote(
            scan_data,
            service_url=args.service_url,
            model_name=args.model,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
        )
        
        if embeddings is None:
            logger.warning("Remote inference failed, falling back to local mode")
            embeddings, file_paths, model_info = generate_embeddings_local(
                scan_data,
                model_name=args.model,
                pretrained=args.pretrained,
                batch_size=args.batch_size,
            )
    
    if embeddings is None or file_paths is None:
        logger.error("Failed to generate embeddings")
        sys.exit(1)
    
    # Store embeddings
    logger.info("Saving embeddings...")
    store = EmbeddingStore(args.output)
    store.clear()
    store.add_embeddings(embeddings, file_paths, model_info)
    store.save()
    
    # Find similar images using grouping
    logger.info(f"Finding similar image groups (threshold: {args.similarity_threshold})...")
    groups = find_similarity_groups(
        embeddings,
        file_paths,
        similarity_threshold=args.similarity_threshold,
        min_group_size=2,
    )
    
    if groups:
        logger.info(f"\nFound {len(groups)} similarity groups:")
        
        # Save groups
        groups_file = args.output / "similar_groups.json"
        groups_data = []
        
        for group_idx, group_indices in enumerate(groups):
            group_sims = compute_group_similarities(embeddings, group_indices)
            avg_sim = sum(group_sims.values()) / len(group_sims) if group_sims else 0.0
            
            group_data = {
                "group_id": group_idx,
                "size": len(group_indices),
                "avg_similarity": float(avg_sim),
                "files": [
                    {
                        "path": file_paths[idx],
                        "name": Path(file_paths[idx]).name,
                        "index": idx,
                    }
                    for idx in group_indices
                ],
                "similarities": {
                    f"{i}-{j}": float(sim)
                    for (i, j), sim in group_sims.items()
                },
            }
            groups_data.append(group_data)
            
            if group_idx < 5:
                logger.info(f"  Group {group_idx + 1}: {len(group_indices)} images, avg similarity: {avg_sim:.3f}")
                for idx in group_indices[:3]:
                    logger.info(f"    - {Path(file_paths[idx]).name}")
        
        with open(groups_file, "w") as f:
            json.dump(groups_data, f, indent=2)
        logger.info(f"\nSaved similar groups to {groups_file}")
        
        # Also save in old pairs format for backward compatibility
        pairs_file = args.output / "similar_pairs.json"
        pairs_data = []
        for group in groups_data:
            files = group["files"]
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    idx_i = files[i]["index"]
                    idx_j = files[j]["index"]
                    key = f"{idx_i}-{idx_j}"
                    sim = group["similarities"].get(key, group["avg_similarity"])
                    pairs_data.append({
                        "file1": files[i]["path"],
                        "file2": files[j]["path"],
                        "similarity": sim,
                        "group_id": group["group_id"],
                    })
        
        with open(pairs_file, "w") as f:
            json.dump(pairs_data, f, indent=2)
    
    else:
        logger.info(f"\nNo similar groups found above threshold {args.similarity_threshold}")
        logger.info("Try lowering --similarity-threshold (e.g., 0.85)")
    
    logger.info("\nDone! Embeddings saved to {args.output}")


if __name__ == "__main__":
    main()
