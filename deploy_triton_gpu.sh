#!/bin/bash
# Build and deploy Triton Inference Server to Vast.ai GPU instance
#
# The GPU-optimized config.pbtxt is already baked into the Docker image
# (see model_repository/openclip_vit_b32/config.pbtxt). This script just
# builds for linux/amd64 and pushes to Docker Hub.

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Triton GPU Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if Docker Hub username is set
DOCKER_USER="${DOCKER_USER:-dahlianadkarni}"
IMAGE_NAME="photo-duplicate-triton"
IMAGE_TAG="gpu-linux-amd64"

echo -e "\n${YELLOW}Docker Image: ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}${NC}\n"

# Step 1: Verify config is GPU-optimized
echo -e "${YELLOW}Step 1: Verifying model config...${NC}"
CONFIG_FILE="model_repository/openclip_vit_b32/config.pbtxt"

if grep -q "KIND_GPU" "$CONFIG_FILE" && grep -q "graphs: true" "$CONFIG_FILE"; then
    echo -e "${GREEN}✓ Config is GPU-optimized (KIND_GPU + CUDA graphs)${NC}"
else
    echo -e "${RED}⚠ Config may not be GPU-optimized. Current instance_group:${NC}"
    grep -A3 "instance_group" "$CONFIG_FILE"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted. Update config.pbtxt first.${NC}"
        exit 1
    fi
fi

QUEUE_DELAY=$(grep "max_queue_delay_microseconds" "$CONFIG_FILE" | grep -o '[0-9]*')
echo -e "  Queue delay: ${BLUE}${QUEUE_DELAY}µs${NC}"

# Step 2: Verify ONNX model exists
echo -e "\n${YELLOW}Step 2: Checking ONNX model...${NC}"
if [ ! -f "model_repository/openclip_vit_b32/1/model.onnx" ]; then
    echo -e "${RED}✗ ONNX model not found!${NC}"
    echo -e "Run: ${YELLOW}python scripts/export_to_onnx.py --test${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ONNX model exists${NC}"

# Step 3: Build for linux/amd64 and push to Docker Hub
echo -e "\n${YELLOW}Step 3: Building Docker image for linux/amd64...${NC}"
echo -e "${BLUE}This may take 5-10 minutes...${NC}"

docker buildx build --platform linux/amd64 \
  -f Dockerfile.triton \
  -t ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG} \
  --push \
  .

echo -e "\n${GREEN}✓ Build complete and pushed to Docker Hub${NC}"

# Step 4: Instructions
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Deployment Image Ready${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Docker Image: ${GREEN}${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e ""
echo -e "${YELLOW}Baked-in config:${NC}"
echo -e "  GPU:         KIND_GPU (CUDA graphs enabled)"
echo -e "  Queue delay: ${QUEUE_DELAY}µs"
echo -e "  Batching:    preferred [4, 8, 16, 32], max 32"
echo -e ""
echo -e "${YELLOW}Deploy on Vast.ai:${NC}"
echo -e ""
echo -e "1. Go to https://vast.ai/ and create an instance"
echo -e ""
echo -e "2. Instance Configuration:"
echo -e "   ${BLUE}GPU:${NC} RTX 3090, RTX 4090, A40, A10 (16GB+ VRAM)"
echo -e "   ${BLUE}Docker Image:${NC} ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"
echo -e "   ${BLUE}Expose Ports:${NC} 8000, 8001, 8002"
echo -e ""
echo -e "3. Test the deployment:"
echo -e "   ${GREEN}curl http://<instance-ip>:<port>/v2/health/ready${NC}"
echo -e "   ${GREEN}curl http://<instance-ip>:<port>/v2/models/openclip_vit_b32${NC}"
echo -e ""
echo -e "4. Run benchmarks:"
echo -e "   ${GREEN}python scripts/benchmark_backends.py --backend triton \\${NC}"
echo -e "   ${GREEN}  --triton-url http://<instance-ip>:<port> --iterations 30${NC}"
echo -e ""
echo -e "5. Connect your UI:"
echo -e "   ${GREEN}export INFERENCE_BACKEND=triton${NC}"
echo -e "   ${GREEN}export INFERENCE_SERVICE_URL=http://<instance-ip>:<port>${NC}"
echo -e "   ${GREEN}python -m src.ui.main${NC}"
echo -e ""
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Cost Estimate:${NC} ~\$0.20-0.80/hour depending on GPU"
echo -e "${BLUE}========================================${NC}"
