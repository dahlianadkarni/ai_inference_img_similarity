"""Storage for image embeddings."""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import json
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Store and retrieve image embeddings."""
    
    def __init__(self, store_path: Path):
        """
        Initialize embedding store.
        
        Args:
            store_path: Path to store embeddings (directory)
        """
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = store_path / "embeddings.npy"
        self.metadata_file = store_path / "metadata.json"
        
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        
        # Load existing data if available
        if self.embeddings_file.exists() and self.metadata_file.exists():
            self.load()
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        file_paths: List[str],
        model_info: Dict,
    ):
        """
        Add embeddings to store.
        
        Args:
            embeddings: Array of embeddings, shape (n, embedding_dim)
            file_paths: List of file paths corresponding to embeddings
            model_info: Model metadata
        """
        if len(embeddings) != len(file_paths):
            raise ValueError("Number of embeddings must match number of file paths")
        
        # Create metadata entries
        new_metadata = [
            {
                "file_path": path,
                "model_name": model_info["model_name"],
                "model_version": model_info["pretrained"],
                "index": i + len(self.metadata),
            }
            for i, path in enumerate(file_paths)
        ]
        
        # Append to existing data
        if self.embeddings is None:
            self.embeddings = embeddings
            self.metadata = new_metadata
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata.extend(new_metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings (total: {len(self.metadata)})")
    
    def save(self):
        """Save embeddings and metadata to disk."""
        if self.embeddings is None or len(self.metadata) == 0:
            logger.warning("No embeddings to save")
            return
        
        # Save embeddings as numpy array
        np.save(self.embeddings_file, self.embeddings)
        
        # Save metadata as JSON
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Saved {len(self.metadata)} embeddings to {self.store_path}")
    
    def clear(self):
        """Clear all embeddings and metadata."""
        self.embeddings = None
        self.metadata = []
        
        # Remove files if they exist
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        logger.info(f"Cleared all embeddings from {self.store_path}")
    
    def load(self):
        """Load embeddings and metadata from disk."""
        if not self.embeddings_file.exists() or not self.metadata_file.exists():
            logger.warning("No saved embeddings found")
            return
        
        self.embeddings = np.load(self.embeddings_file)
        
        with open(self.metadata_file, "r") as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded {len(self.metadata)} embeddings from {self.store_path}")
    
    def get_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific file.
        
        Args:
            file_path: Path to image file
        
        Returns:
            Embedding vector or None if not found
        """
        for meta in self.metadata:
            if meta["file_path"] == file_path:
                idx = meta["index"]
                return self.embeddings[idx]
        
        return None
    
    def get_all_embeddings(self) -> tuple[np.ndarray, List[str]]:
        """
        Get all embeddings and their file paths.
        
        Returns:
            Tuple of (embeddings array, list of file paths)
        """
        if self.embeddings is None:
            return np.array([]), []
        
        file_paths = [meta["file_path"] for meta in self.metadata]
        return self.embeddings, file_paths
    
    def __len__(self) -> int:
        """Return number of stored embeddings."""
        return len(self.metadata)
