"""Command-line interface for generating embeddings."""

import argparse
import logging
from pathlib import Path
import sys
import json

import numpy as np
from tqdm import tqdm

from ..scanner.scanner import PhotoScanner
from .embedder import ImageEmbedder, compute_similarity_matrix
from .storage import EmbeddingStore
from ..grouping import find_similarity_groups, compute_group_similarities


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
        description="Generate embeddings for scanned photos"
    )
    parser.add_argument(
        "scan_results",
        type=Path,
        help="Path to scan_results.json from scanner",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="CLIP model to use (default: ViT-B-32, lightweight option: ViT-B-16)",
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
    
    # Initialize embedder
    logger.info("Initializing embedding model...")
    embedder = ImageEmbedder(
        model_name=args.model,
        pretrained=args.pretrained,
    )
    model_info = embedder.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Generate embeddings in batches to avoid memory issues
    logger.info("Generating embeddings...")
    from PIL import Image
    
    # Process images in chunks to avoid loading all into memory
    all_embeddings = []
    valid_file_paths = []
    chunk_size = 100  # Process 100 images at a time
    
    for chunk_start in range(0, len(scan_data), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(scan_data))
        chunk_data = scan_data[chunk_start:chunk_end]
        
        # Load images for this chunk
        images = []
        chunk_paths = []
        
        for item in tqdm(chunk_data, desc=f"Loading images ({chunk_start+1}-{chunk_end}/{len(scan_data)})", leave=False):
            file_path = Path(item["file_path"])
            
            # Skip if file no longer exists (e.g., temp export cleaned up)
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
            # Generate embeddings for this chunk
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(scan_data) + chunk_size - 1)//chunk_size}: {len(images)} images")
            chunk_embeddings = embedder.embed_images_batch(images, batch_size=args.batch_size)
            all_embeddings.append(chunk_embeddings)
            valid_file_paths.extend(chunk_paths)
            
            # Clear images from memory
            images.clear()
    
    if len(valid_file_paths) == 0:
        logger.error("No images could be loaded")
        sys.exit(1)
    
    # Combine all embeddings
    logger.info(f"Combining embeddings from {len(all_embeddings)} chunks...")
    embeddings = np.vstack(all_embeddings)
    file_paths = valid_file_paths
    logger.info(f"Generated embeddings for {len(file_paths)} images")
    
    # Store embeddings (clear old data first)
    logger.info("Saving embeddings...")
    store = EmbeddingStore(args.output)
    store.clear()  # Clear old embeddings before adding new ones
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
            # Compute pairwise similarities within group
            group_sims = compute_group_similarities(embeddings, group_indices)
            
            # Get average similarity
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
            
            # Log first few groups
            if group_idx < 5:
                logger.info(f"  Group {group_idx + 1}: {len(group_indices)} images, avg similarity: {avg_sim:.3f}")
                for idx in group_indices[:3]:  # Show first 3
                    logger.info(f"    - {Path(file_paths[idx]).name}")
        
        with open(groups_file, "w") as f:
            json.dump(groups_data, f, indent=2)
        logger.info(f"\nSaved similar groups to {groups_file}")
        
        # Also save in old pairs format for backward compatibility
        pairs_file = args.output / "similar_pairs.json"
        pairs_data = []
        for group in groups_data:
            # Convert groups to pairs
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
