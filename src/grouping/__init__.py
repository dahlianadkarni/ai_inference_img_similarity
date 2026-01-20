"""Grouping and clustering for similar images."""

import logging
from typing import List, Set, Dict
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def find_similarity_groups(
    embeddings: np.ndarray,
    file_paths: List[str],
    similarity_threshold: float = 0.85,
    min_group_size: int = 2,
) -> List[List[int]]:
    """
    Find groups of similar images using union-find clustering.
    
    Args:
        embeddings: Array of embeddings, shape (n, embedding_dim)
        file_paths: List of file paths
        similarity_threshold: Minimum similarity to group together
        min_group_size: Minimum size for a group to be returned
    
    Returns:
        List of groups, where each group is a list of indices
    """
    n = len(embeddings)
    
    # Compute similarity matrix
    similarity_matrix = embeddings @ embeddings.T
    
    # Union-Find data structure
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Group similar images
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)
    
    # Collect groups
    groups = defaultdict(list)
    for i in range(n):
        root = find(i)
        groups[root].append(i)
    
    # Filter by minimum size and sort by size
    result = [group for group in groups.values() if len(group) >= min_group_size]
    result.sort(key=len, reverse=True)
    
    logger.info(f"Found {len(result)} groups (threshold: {similarity_threshold})")
    for i, group in enumerate(result[:5]):  # Log first 5
        logger.info(f"  Group {i+1}: {len(group)} images")
    
    return result


def compute_group_similarities(
    embeddings: np.ndarray,
    group_indices: List[int],
) -> Dict[tuple, float]:
    """
    Compute pairwise similarities within a group.
    
    Args:
        embeddings: Array of embeddings
        group_indices: Indices of images in the group
    
    Returns:
        Dictionary mapping (i, j) tuples to similarity scores
    """
    similarities = {}
    
    for i, idx_i in enumerate(group_indices):
        for j, idx_j in enumerate(group_indices):
            if i < j:
                sim = float(embeddings[idx_i] @ embeddings[idx_j])
                similarities[(idx_i, idx_j)] = sim
    
    return similarities


class FeedbackLearner:
    """Learn from user feedback to improve similarity detection."""
    
    def __init__(self):
        """Initialize feedback learner."""
        self.negative_pairs: List[tuple[np.ndarray, np.ndarray]] = []
        self.positive_pairs: List[tuple[np.ndarray, np.ndarray]] = []
    
    def add_negative_feedback(self, emb1: np.ndarray, emb2: np.ndarray):
        """
        Add a negative example (marked as not similar).
        
        Args:
            emb1: First embedding
            emb2: Second embedding
        """
        self.negative_pairs.append((emb1, emb2))
        logger.info(f"Added negative feedback (total: {len(self.negative_pairs)})")
    
    def add_positive_feedback(self, emb1: np.ndarray, emb2: np.ndarray):
        """
        Add a positive example (marked as similar).
        
        Args:
            emb1: First embedding
            emb2: Second embedding
        """
        self.positive_pairs.append((emb1, emb2))
        logger.info(f"Added positive feedback (total: {len(self.positive_pairs)})")
    
    def adjust_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        base_similarity: float,
    ) -> float:
        """
        Adjust similarity score based on learned feedback.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            base_similarity: Original similarity score
        
        Returns:
            Adjusted similarity score
        """
        if not self.negative_pairs and not self.positive_pairs:
            return base_similarity
        
        # Check if this pair is similar to any negative examples
        negative_penalty = 0.0
        for neg_emb1, neg_emb2 in self.negative_pairs:
            # Compute similarity to negative example
            sim1 = max(
                float(emb1 @ neg_emb1),
                float(emb1 @ neg_emb2),
                float(emb2 @ neg_emb1),
                float(emb2 @ neg_emb2),
            )
            
            # If very similar to a negative example, penalize
            if sim1 > 0.85:
                negative_penalty += 0.1 * sim1
        
        # Check if similar to positive examples (boost)
        positive_boost = 0.0
        for pos_emb1, pos_emb2 in self.positive_pairs:
            sim1 = max(
                float(emb1 @ pos_emb1),
                float(emb1 @ pos_emb2),
                float(emb2 @ pos_emb1),
                float(emb2 @ pos_emb2),
            )
            
            if sim1 > 0.85:
                positive_boost += 0.05 * sim1
        
        adjusted = base_similarity - negative_penalty + positive_boost
        adjusted = max(0.0, min(1.0, adjusted))  # Clamp to [0, 1]
        
        if abs(adjusted - base_similarity) > 0.01:
            logger.debug(
                f"Adjusted similarity: {base_similarity:.3f} → {adjusted:.3f} "
                f"(penalty: {negative_penalty:.3f}, boost: {positive_boost:.3f})"
            )
        
        return adjusted
    
    def save(self, path: str):
        """Save feedback to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "negative_pairs": self.negative_pairs,
                "positive_pairs": self.positive_pairs,
            }, f)
        logger.info(f"Saved feedback to {path}")
    
    def load(self, path: str):
        """Load feedback from disk."""
        import pickle
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.negative_pairs = data.get("negative_pairs", [])
                self.positive_pairs = data.get("positive_pairs", [])
            logger.info(
                f"Loaded feedback: {len(self.negative_pairs)} negative, "
                f"{len(self.positive_pairs)} positive"
            )
        except FileNotFoundError:
            logger.info("No saved feedback found")
