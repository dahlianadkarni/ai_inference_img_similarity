"""Dual startup script for inference service + UI client.

This starts both services:
1. Inference service (port 8001) - Stateless model inference
2. UI client (port 8000) - Photo review interface

They communicate via HTTP, allowing for independent scaling.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import requests


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def wait_for_service(url: str, timeout: float = 30.0) -> bool:
    """
    Wait for a service to become available.
    
    Args:
        url: Service URL to check
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if service became available, False if timeout
    """
    logger = logging.getLogger(__name__)
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        
        time.sleep(0.5)
    
    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start inference service and UI client"
    )
    parser.add_argument(
        "--inference-host",
        type=str,
        default="127.0.0.1",
        help="Host for inference service",
    )
    parser.add_argument(
        "--inference-port",
        type=int,
        default=8001,
        help="Port for inference service",
    )
    parser.add_argument(
        "--ui-host",
        type=str,
        default="127.0.0.1",
        help="Host for UI server",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8000,
        help="Port for UI server",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip starting inference service (useful if running separately)",
    )
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    inference_url = f"http://{args.inference_host}:{args.inference_port}/health"
    ui_url = f"http://{args.ui_host}:{args.ui_port}"
    
    # Start inference service
    inference_process = None
    if not args.no_inference:
        logger.info(f"Starting inference service on {args.inference_host}:{args.inference_port}...")
        logger.info("(This loads the vision model and may take ~5-10 seconds on first run)")
        
        inference_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.inference_service.server",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        
        logger.info("Waiting for inference service to become ready...")
        if wait_for_service(inference_url, timeout=60.0):
            logger.info("✓ Inference service is ready")
        else:
            logger.error("✗ Inference service failed to start")
            if inference_process:
                inference_process.terminate()
            sys.exit(1)
    
    # Start UI server
    logger.info(f"\nStarting UI server on {args.ui_host}:{args.ui_port}...")
    
    ui_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.ui.main",
            "--host",
            args.ui_host,
            "--port",
            str(args.ui_port),
        ],
        cwd=Path(__file__).parent.parent.parent,
    )
    
    logger.info("Waiting for UI server to become ready...")
    if wait_for_service(ui_url, timeout=30.0):
        logger.info("✓ UI server is ready")
    else:
        logger.error("✗ UI server failed to start")
        if inference_process:
            inference_process.terminate()
        ui_process.terminate()
        sys.exit(1)
    
    # Both services running
    logger.info("\n" + "="*60)
    logger.info("✓ BOTH SERVICES RUNNING")
    logger.info("="*60)
    
    if not args.no_inference:
        logger.info(f"Inference Service: http://{args.inference_host}:{args.inference_port}")
        logger.info(f"  (Stateless embedding API)")
    
    logger.info(f"UI Client:        http://{args.ui_host}:{args.ui_port}")
    logger.info(f"  Open this in your browser")
    logger.info("\nArchitecture:")
    logger.info("  Client (UI) ---> Inference Service")
    logger.info("                   └─> Model inference")
    logger.info("                   └─> Returns embeddings")
    logger.info("\nPress Ctrl+C to stop both services")
    logger.info("="*60 + "\n")
    
    # Wait for processes
    try:
        if inference_process:
            inference_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        if inference_process:
            inference_process.terminate()
        ui_process.terminate()
        
        # Wait for graceful shutdown
        try:
            if inference_process:
                inference_process.wait(timeout=3)
            ui_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            logger.warning("Forcing shutdown...")
            if inference_process:
                inference_process.kill()
            ui_process.kill()


if __name__ == "__main__":
    main()
