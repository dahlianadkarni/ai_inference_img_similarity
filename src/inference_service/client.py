"""Client for calling remote inference service.

This module handles communication with the inference service,
including image encoding and response handling.
"""

import base64
import io
import logging
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for calling the inference service."""
    
    def __init__(
        self,
        service_url: str = "http://127.0.0.1:8002",
        timeout: float = 300.0,
    ):
        """
        Initialize client.
        
        Args:
            service_url: Base URL of inference service
            timeout: Request timeout in seconds
        """
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
    
    def health_check(self) -> bool:
        """
        Check if inference service is healthy.
        
        Returns:
            True if service is accessible and healthy
        """
        try:
            response = self.client.get(f"{self.service_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Model information dict
        """
        try:
            response = self.client.get(f"{self.service_url}/model-info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def embed_images_base64(
        self,
        images: List[Image.Image],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> np.ndarray:
        """
        Embed images by encoding them as base64 and sending to service.
        
        This method is useful for small batches or testing, but less
        efficient than file uploads for large images.
        
        Args:
            images: List of PIL Images
            model_name: CLIP model to use
            pretrained: Pretrained weights
        
        Returns:
            Array of embeddings, shape (n_images, embedding_dim)
        """
        # Encode images to base64
        b64_images = []
        for img in images:
            # Convert to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            img_bytes = buffer.getvalue()
            
            # Encode to base64
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            b64_images.append(b64)
        
        # Send request
        try:
            response = self.client.post(
                f"{self.service_url}/embed/base64",
                json={
                    "images": b64_images,
                    "model_name": model_name,
                    "pretrained": pretrained,
                },
            )
            response.raise_for_status()
            result = response.json()
            
            # Convert back to numpy array
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to embed images: {e}")
            raise
    
    def embed_images_files(
        self,
        image_paths: List[Path],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> np.ndarray:
        """
        Embed images by uploading files.
        
        This method is more efficient for large images or batches
        since it avoids base64 encoding overhead.
        
        Args:
            image_paths: List of Path objects to image files
            model_name: CLIP model to use
            pretrained: Pretrained weights
        
        Returns:
            Array of embeddings, shape (n_images, embedding_dim)
        """
        # Prepare files for upload
        files = []
        for path in image_paths:
            with open(path, "rb") as f:
                files.append(("files", (path.name, f.read(), "image/jpeg")))
        
        # Send request
        try:
            response = self.client.post(
                f"{self.service_url}/embed/batch",
                files=files,
                data={
                    "model_name": model_name,
                    "pretrained": pretrained,
                },
            )
            response.raise_for_status()
            result = response.json()
            
            # Convert back to numpy array
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to embed images: {e}")
            raise
    
    def embed_image_files_batched(
        self,
        image_paths: List[Path],
        batch_size: int = 32,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> np.ndarray:
        """
        Embed images in batches.
        
        This is useful for processing large numbers of images while
        managing memory and service load.
        
        Args:
            image_paths: List of Path objects to image files
            batch_size: Number of images per batch
            model_name: CLIP model to use
            pretrained: Pretrained weights
        
        Returns:
            Array of embeddings, shape (n_images, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}: {len(batch_paths)} images")
            
            batch_embeddings = self.embed_images_files(
                batch_paths,
                model_name=model_name,
                pretrained=pretrained,
            )
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    client = InferenceClient()
    
    # Check health
    if client.health_check():
        print("✓ Service is healthy")
        print(f"Model info: {client.get_model_info()}")
    else:
        print("✗ Service is not available")
        print("Start the inference service with:")
        print("  python -m src.inference_service.server")
