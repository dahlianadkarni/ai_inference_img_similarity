"""Stateless inference server for generating image embeddings.

This service is responsible for:
- Loading the vision model once at startup
- Accepting image data via HTTP
- Computing embeddings
- Returning results

Key principle: The service is stateless regarding photos/metadata.
The client handles photo management, storage, and organization.
"""

import base64
import io
import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from ..embedding.embedder import ImageEmbedder

logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request to embed one or more images."""
    
    images: List[str]  # Base64-encoded images
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"


class EmbeddingResponse(BaseModel):
    """Response with computed embeddings."""
    
    embeddings: List[List[float]]  # List of embedding vectors
    model_info: dict
    count: int


class InferenceService:
    """Manages model loading and inference."""
    
    _instance = None
    _embedder: Optional[ImageEmbedder] = None
    _current_model: Optional[str] = None
    _current_pretrained: Optional[str] = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def load_model(
        cls,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        """Load a model, reusing if same model/pretrained is requested."""
        instance = cls.get_instance()
        
        # Reuse existing model if same
        if instance._embedder and instance._current_model == model_name and instance._current_pretrained == pretrained:
            logger.info(f"Model {model_name} ({pretrained}) already loaded")
            return instance._embedder
        
        # Load new model
        logger.info(f"Loading model: {model_name} ({pretrained})")
        instance._embedder = ImageEmbedder(
            model_name=model_name,
            pretrained=pretrained,
        )
        instance._current_model = model_name
        instance._current_pretrained = pretrained
        
        return instance._embedder
    
    @classmethod
    def embed_images(
        cls,
        images: List[Image.Image],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for images.
        
        Args:
            images: List of PIL Images
            model_name: CLIP model to use
            pretrained: Pretrained weights
            batch_size: Batch size for processing
        
        Returns:
            Array of embeddings
        """
        embedder = cls.load_model(model_name, pretrained)
        embeddings = embedder.embed_images_batch(images, batch_size=batch_size)
        return embeddings


def create_app() -> FastAPI:
    """Create FastAPI application for inference service."""
    
    app = FastAPI(
        title="Embedding Inference Service",
        description="Stateless service for computing image embeddings",
        version="0.1.0",
    )
    
    @app.on_event("startup")
    async def startup():
        """Load default model at startup."""
        logger.info("Starting inference service...")
        InferenceService.load_model("ViT-B-32", "openai")
        logger.info("Inference service ready")
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}
    
    @app.get("/model-info")
    async def get_model_info():
        """Get information about currently loaded model."""
        embedder = InferenceService._instance._embedder if InferenceService._instance else None
        
        if not embedder:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        return embedder.get_model_info()
    
    @app.post("/embed/base64", response_model=EmbeddingResponse)
    async def embed_base64(request: EmbeddingRequest):
        """
        Embed images provided as base64 strings.
        
        This is useful for client-server scenarios where images are
        transmitted over HTTP.
        
        Args:
            request: EmbeddingRequest with base64-encoded images
        
        Returns:
            EmbeddingResponse with embeddings
        """
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        try:
            # Decode base64 images
            images = []
            for i, b64_image in enumerate(request.images):
                try:
                    # Handle data URI format if present
                    if b64_image.startswith("data:image"):
                        b64_image = b64_image.split(",")[1]
                    
                    image_data = base64.b64decode(b64_image)
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to decode image {i}: {e}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decode image {i}: {str(e)}"
                    )
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(images)} images")
            embeddings = InferenceService.embed_images(
                images,
                model_name=request.model_name,
                pretrained=request.pretrained,
                batch_size=32,
            )
            
            # Convert to list of lists for JSON serialization
            embeddings_list = embeddings.tolist()
            
            # Get model info
            model_info = InferenceService.load_model(
                request.model_name,
                request.pretrained,
            ).get_model_info()
            
            return EmbeddingResponse(
                embeddings=embeddings_list,
                model_info=model_info,
                count=len(embeddings_list),
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error during embedding")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/embed/batch", response_model=EmbeddingResponse)
    async def embed_batch(
        files: List[UploadFile] = File(...),
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        """
        Embed images provided as multipart file uploads.
        
        This is useful for direct file uploads and is more efficient
        than base64 encoding for large images.
        
        Args:
            files: List of image files
            model_name: CLIP model to use
            pretrained: Pretrained weights
        
        Returns:
            EmbeddingResponse with embeddings
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        try:
            images = []
            for i, file in enumerate(files):
                try:
                    image_data = await file.read()
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to read file {i} ({file.filename}): {e}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to read file {i}: {str(e)}"
                    )
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(images)} images")
            embeddings = InferenceService.embed_images(
                images,
                model_name=model_name,
                pretrained=pretrained,
                batch_size=32,
            )
            
            # Convert to list of lists for JSON serialization
            embeddings_list = embeddings.tolist()
            
            # Get model info
            model_info = InferenceService.load_model(
                model_name,
                pretrained,
            ).get_model_info()
            
            return EmbeddingResponse(
                embeddings=embeddings_list,
                model_info=model_info,
                count=len(embeddings_list),
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error during embedding")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def main():
    """Main entry point."""
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8002)


if __name__ == "__main__":
    main()
