#!/bin/bash
# Build and deploy Triton Inference Server to Vast.ai GPU instance

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

# Step 1: Update config for GPU
echo -e "${YELLOW}Step 1: Updating model config for GPU deployment...${NC}"

# Create GPU version of config
cat > model_repository/openclip_vit_b32/config.pbtxt << 'EOF'
name: "openclip_vit_b32"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "embedding"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]

# Dynamic batching configuration
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

# Instance configuration - GPU
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# Version policy
version_policy: { specific { versions: 1 }}

# Optimization settings
optimization {
  cuda {
    graphs: true
  }
}
EOF

echo -e "${GREEN}✓ Config updated for GPU${NC}"

# Step 2: Build for linux/amd64
echo -e "\n${YELLOW}Step 2: Building Docker image for linux/amd64...${NC}"
echo -e "${BLUE}This may take 5-10 minutes...${NC}"

docker buildx build --platform linux/amd64 \
  -f Dockerfile.triton \
  -t ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG} \
  --push \
  .

echo -e "\n${GREEN}✓ Build complete and pushed to Docker Hub${NC}"

# Step 3: Restore local config (CPU)
echo -e "\n${YELLOW}Step 3: Restoring local config for CPU testing...${NC}"

cat > model_repository/openclip_vit_b32/config.pbtxt << 'EOF'
name: "openclip_vit_b32"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "embedding"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]

# Dynamic batching configuration
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

# Instance configuration - CPU for local testing
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

# Version policy
version_policy: { specific { versions: 1 }}
EOF

echo -e "${GREEN}✓ Local config restored${NC}"

# Step 4: Instructions
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Deployment Image Ready${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Docker Image: ${GREEN}${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e ""
echo -e "${YELLOW}Next Steps:${NC}"
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
echo -e "4. Connect your UI:"
echo -e "   ${GREEN}export INFERENCE_BACKEND=triton${NC}"
echo -e "   ${GREEN}export INFERENCE_SERVICE_URL=http://<instance-ip>:<port>${NC}"
echo -e "   ${GREEN}python -m src.ui.main${NC}"
echo -e ""
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Cost Estimate:${NC} ~$0.20-0.80/hour depending on GPU"
echo -e "${BLUE}========================================${NC}"
