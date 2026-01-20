"""Image embedding generation using CLIP models."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

import torch
from PIL import Image
import open_clip

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    
    file_path: str
    embedding: np.ndarray
    model_name: str
    model_version: str


class ImageEmbedder:
    """Generate image embeddings using CLIP-based models."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to use
            device: Device to use (cuda/mps/cpu), auto-detected if None
        """
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model: {model_name} ({pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: PIL Image
        
        Returns:
            Normalized embedding vector
        """
        # Preprocess and move to device
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            # Move to CPU and convert to numpy
            embedding = embedding.cpu().numpy().squeeze()
        
        return embedding
    
    def embed_images_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
        
        Returns:
            Array of embeddings, shape (n_images, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            image_tensors = torch.stack([
                self.preprocess(img) for img in batch
            ]).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(image_tensors)
                
                # Normalize
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                
                # Move to CPU
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": str(self.device),
            "embedding_dim": self.model.visual.output_dim,
        }


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (-1 to 1, higher is more similar)
    """
    # Embeddings should already be normalized from the model
    similarity = np.dot(embedding1, embedding2)
    return float(similarity)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Array of embeddings, shape (n, embedding_dim)
    
    Returns:
        Similarity matrix, shape (n, n)
    """
    # For normalized vectors, cosine similarity is just dot product
    similarity_matrix = embeddings @ embeddings.T
    return similarity_matrix


def estimate_embedding_throughput(
    embedder: ImageEmbedder,
    sample_images: List[Image.Image],
    batch_size: int = 32,
) -> dict:
    """
    Estimate embedding generation throughput by processing a sample.
    
    Args:
        embedder: ImageEmbedder instance
        sample_images: Sample images to process
        batch_size: Batch size for processing
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    if len(sample_images) == 0:
        return {
            "images_per_second": 0,
            "sample_size": 0,
            "total_time": 0,
        }
    
    start_time = time.time()
    _ = embedder.embed_images_batch(sample_images, batch_size=batch_size)
    elapsed = time.time() - start_time
    
    images_per_second = len(sample_images) / elapsed if elapsed > 0 else 0
    
    return {
        "images_per_second": images_per_second,
        "sample_size": len(sample_images),
        "total_time": elapsed,
    }
