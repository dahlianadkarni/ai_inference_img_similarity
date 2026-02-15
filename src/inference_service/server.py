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
import os
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
        import torch
        
        logger.info("Starting inference service...")
        
        # Log GPU availability
        logger.info("="*60)
        logger.info("GPU Configuration:")
        if torch.cuda.is_available():
            logger.info(f"  CUDA Available: YES")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.2f} GB)")
        else:
            logger.warning("  CUDA Available: NO - Running on CPU!")
        logger.info("="*60)
        
        InferenceService.load_model("ViT-B-32", "openai")
        
        # Verify model is on GPU
        embedder = InferenceService._instance._embedder if InferenceService._instance else None
        if embedder:
            model_device = str(next(embedder.model.parameters()).device)
            logger.info(f"Model loaded on device: {model_device}")
            if torch.cuda.is_available() and "cuda" not in model_device:
                logger.error("⚠️  WARNING: CUDA is available but model is on CPU!")
            elif "cuda" in model_device:
                logger.info("✓ Model successfully loaded on GPU")
        
        logger.info("Inference service ready")
    
    @app.get("/health")
    @app.get("/healthz")  # Alias for K8s-style health checks
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
    
    @app.get("/gpu-info")
    async def gpu_info():
        """Get detailed GPU information and verify GPU is being used."""
        import torch
        
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info["devices"] = []
            for i in range(torch.cuda.device_count()):
                info["devices"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                    "current_device": i == torch.cuda.current_device()
                })
            
            # Check if model is actually on GPU
            embedder = InferenceService._instance._embedder if InferenceService._instance else None
            if embedder:
                model_device = str(next(embedder.model.parameters()).device)
                info["model_device"] = model_device
                info["model_on_gpu"] = "cuda" in model_device
            else:
                info["model_device"] = "not loaded"
                info["model_on_gpu"] = False
        else:
            info["model_on_gpu"] = False
            info["warning"] = "CUDA not available - running on CPU"
        
        return JSONResponse(content=info)
    
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
    import argparse
    import uvicorn
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference service for image embeddings")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1 or HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8002")),
        help="Port to bind to (default: 8002 or PORT env var)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "ViT-B-32"),
        help="Model name to use (default: ViT-B-32 or MODEL_NAME env var)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=os.getenv("MODEL_PRETRAINED", "openai"),
        help="Pretrained weights (default: openai or MODEL_PRETRAINED env var)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info or LOG_LEVEL env var)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info(f"Starting inference service on {args.host}:{args.port}")
    logger.info(f"Default model: {args.model_name} ({args.pretrained})")
    
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
