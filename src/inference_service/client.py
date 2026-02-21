"""Client for calling remote inference service.

This module handles communication with the inference service,
including image encoding and response handling.

Supports multiple backends:
- pytorch: FastAPI + PyTorch (default)
- triton: NVIDIA Triton Inference Server
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
from PIL import Image

try:
    import tritonclient.http as httpclient
    TRITONCLIENT_AVAILABLE = True
except ImportError:
    TRITONCLIENT_AVAILABLE = False

try:
    import tritonclient.grpc as grpcclient
    TRITONCLIENT_GRPC_AVAILABLE = True
except ImportError:
    TRITONCLIENT_GRPC_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for calling the inference service.
    
    Supports multiple backends via environment variables:
    - INFERENCE_BACKEND: "pytorch" (default), "triton", or "triton_grpc"
    - INFERENCE_SERVICE_URL: Service URL (default: http://127.0.0.1:8002)
    - TRITON_GRPC_URL: gRPC URL for triton_grpc backend (default: auto-derived as HTTP_PORT+1)
    """
    
    def __init__(
        self,
        service_url: str = None,
        backend: str = None,
        timeout: float = 300.0,
        grpc_url: str = None,
    ):
        """
        Initialize client.
        
        Args:
            service_url: Base URL of inference service (or use INFERENCE_SERVICE_URL env)
            backend: Backend type - "pytorch", "triton", or "triton_grpc" (or use INFERENCE_BACKEND env)
            timeout: Request timeout in seconds
            grpc_url: gRPC endpoint for triton_grpc backend, e.g. "host:8004"
                      (or use TRITON_GRPC_URL env; defaults to HTTP port+1)
        """
        self.service_url = (service_url or os.getenv("INFERENCE_SERVICE_URL", "http://127.0.0.1:8002")).rstrip("/")
        self.backend = (backend or os.getenv("INFERENCE_BACKEND", "pytorch")).lower()
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

        # Resolve gRPC URL for triton_grpc backend
        if self.backend == "triton_grpc":
            self.grpc_url = (
                grpc_url
                or os.getenv("TRITON_GRPC_URL")
                or self._derive_grpc_url()
            ).replace("grpc://", "").replace("http://", "")
        
        logger.info(f"InferenceClient initialized: backend={self.backend}, url={self.service_url}")

    def _derive_grpc_url(self) -> str:
        """Derive Triton gRPC URL from HTTP service URL by incrementing port by 1.
        
        Triton always exposes gRPC on HTTP_PORT+1 in our docker-compose setup:
          local:  HTTP 8003 → gRPC 8004
          step6a: HTTP 8010 → gRPC 8011, HTTP 8020 → gRPC 8021
        """
        url = self.service_url.replace("http://", "").replace("https://", "")
        host, _, port_str = url.rpartition(":")
        if host and port_str.isdigit():
            return f"{host}:{int(port_str) + 1}"
        return url
    
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
            if self.backend in ("triton", "triton_grpc"):
                response = self.client.get(f"{self.service_url}/v2/health/ready")
            else:  # pytorch
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
            if self.backend in ("triton", "triton_grpc"):
                response = self.client.get(f"{self.service_url}/v2/models/openclip_vit_b32")
                response.raise_for_status()
                data = response.json()
                # Reformat to match PyTorch response
                return {
                    "model_name": data.get("name"),
                    "platform": data.get("platform"),
                    "backend": self.backend,
                    "max_batch_size": data.get("max_batch_size"),
                    "pretrained": "openai",  # Default for Triton ONNX model
                }
            else:  # pytorch
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
        if self.backend == "triton_grpc":
            return self._embed_triton_grpc(images)
        elif self.backend == "triton":
            return self._embed_triton(images)
        else:  # pytorch
            return self._embed_pytorch_base64(images, model_name, pretrained)
    
    def _embed_pytorch_base64(
        self,
        images: List[Image.Image],
        model_name: str,
        pretrained: str,
    ) -> np.ndarray:
        """Embed images using PyTorch backend."""
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
    
    def _embed_triton(self, images: List[Image.Image]) -> np.ndarray:
        """Embed images using Triton backend with binary protocol."""
        # Preprocess images to model input format
        # Triton expects: [batch, channels, height, width] float32 in range [0, 1]
        batch = []
        for img in images:
            # Resize to 224x224
            img_resized = img.resize((224, 224), Image.BILINEAR)
            
            # Convert to numpy array
            img_array = np.array(img_resized).astype(np.float32)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Convert HWC to CHW
            img_array = np.transpose(img_array, (2, 0, 1))
            
            batch.append(img_array)
        
        # Stack into batch
        batch_array = np.array(batch, dtype=np.float32)
        
        # Send request using binary protocol if available
        try:
            if TRITONCLIENT_AVAILABLE:
                # Use tritonclient with binary data (1000x+ faster than JSON)
                url_parts = self.service_url.replace('http://', '').replace('https://', '').split(':')
                host = url_parts[0]
                port = url_parts[1] if len(url_parts) > 1 else '8000'
                
                triton_client = httpclient.InferenceServerClient(url=f"{host}:{port}")
                inputs = [httpclient.InferInput("image", batch_array.shape, "FP32")]
                inputs[0].set_data_from_numpy(batch_array)
                outputs = [httpclient.InferRequestedOutput("embedding")]
                
                result = triton_client.infer("openclip_vit_b32", inputs, outputs=outputs)
                embeddings = result.as_numpy("embedding")
                
                return embeddings
            else:
                # Fallback to JSON (slow, but works without tritonclient)
                logger.warning("tritonclient not available, using slow JSON protocol. Install with: pip install tritonclient[http]")
                payload = {
                    "inputs": [
                        {
                            "name": "image",
                            "shape": list(batch_array.shape),
                            "datatype": "FP32",
                            "data": batch_array.flatten().tolist()
                        }
                    ],
                    "outputs": [
                        {
                            "name": "embedding"
                        }
                    ]
                }
                
                response = self.client.post(
                    f"{self.service_url}/v2/models/openclip_vit_b32/infer",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract embeddings
                embeddings = np.array(result['outputs'][0]['data'], dtype=np.float32)
                
                # Reshape to (batch_size, embedding_dim)
                embeddings = embeddings.reshape(len(images), -1)
                
                return embeddings
        
        except Exception as e:
            logger.error(f"Failed to embed images with Triton HTTP: {e}")
            raise

    def _embed_triton_grpc(self, images: List[Image.Image]) -> np.ndarray:
        """Embed images using Triton gRPC backend (HTTP/2 + protobuf binary).

        Advantages over HTTP:
        - Persistent connection (no TCP handshake per call)
        - HTTP/2 multiplexing for concurrent requests
        - Slightly smaller wire format (protobuf vs HTTP headers)
        """
        if not TRITONCLIENT_GRPC_AVAILABLE:
            logger.warning(
                "tritonclient[grpc] not installed, falling back to HTTP. "
                "Install with: pip install tritonclient[grpc]"
            )
            return self._embed_triton(images)

        # Preprocess images into [batch, C, H, W] float32
        batch = []
        for img in images:
            img_resized = img.resize((224, 224), Image.BILINEAR)
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
            batch.append(img_array)
        batch_array = np.array(batch, dtype=np.float32)

        try:
            client = grpcclient.InferenceServerClient(url=self.grpc_url)
            inputs = [grpcclient.InferInput("image", batch_array.shape, "FP32")]
            inputs[0].set_data_from_numpy(batch_array)
            outputs = [grpcclient.InferRequestedOutput("embedding")]

            result = client.infer("openclip_vit_b32", inputs, outputs=outputs)
            embeddings = result.as_numpy("embedding")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to embed images with Triton gRPC: {e}")
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
        
        Note: Only supported by PyTorch backend currently.
        
        Args:
            image_paths: List of Path objects to image files
            model_name: CLIP model to use
            pretrained: Pretrained weights
        
        Returns:
            Array of embeddings, shape (n_images, embedding_dim)
        """
        if self.backend in ("triton", "triton_grpc"):
            # For Triton backends, load images and use tensor method
            images = [Image.open(path) for path in image_paths]
            return self.embed_images_base64(images, model_name, pretrained)
        
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
