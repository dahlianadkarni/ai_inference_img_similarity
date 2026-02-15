#!/bin/bash
# =============================================================================
# Build and deploy Triton + TensorRT image to Vast.ai GPU instance (Step 5B)
#
# This builds the TensorRT-enabled Triton image and pushes to Docker Hub.
# The image includes both ONNX and TensorRT model configs. On first startup,
# the TensorRT engine is auto-built for the target GPU.
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 5B: TensorRT GPU Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if Docker Hub username is set
DOCKER_USER="${DOCKER_USER:-dahlianadkarni}"
IMAGE_NAME="photo-duplicate-triton"
IMAGE_TAG="tensorrt-gpu"

echo -e "\n${YELLOW}Docker Image: ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}${NC}\n"

# Step 1: Verify files exist
echo -e "${YELLOW}Step 1: Verifying required files...${NC}"

# Check ONNX model
if [ ! -f "model_repository/openclip_vit_b32/1/model.onnx" ]; then
    echo -e "${RED}✗ ONNX model not found!${NC}"
    echo -e "Run: ${YELLOW}python scripts/export_to_onnx.py --test${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ONNX model exists${NC}"

# Check ONNX config (for baseline comparison)
if [ ! -f "model_repository/openclip_vit_b32/config.pbtxt" ]; then
    echo -e "${RED}✗ ONNX config not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ONNX model config exists${NC}"

# Check TRT config
if [ ! -f "model_repository/openclip_vit_b32_trt/config.pbtxt" ]; then
    echo -e "${RED}✗ TensorRT model config not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ TensorRT model config exists${NC}"

# Check entrypoint
if [ ! -f "scripts/triton_trt_entrypoint.sh" ]; then
    echo -e "${RED}✗ Entrypoint script not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Entrypoint script exists${NC}"

# Check Dockerfile
if [ ! -f "Dockerfile.tensorrt" ]; then
    echo -e "${RED}✗ Dockerfile.tensorrt not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dockerfile.tensorrt exists${NC}"

# Step 2: Show configs
echo -e "\n${YELLOW}Step 2: Configuration summary${NC}"

echo -e "\n  ${BLUE}ONNX model (baseline):${NC}"
echo -e "    Platform: onnxruntime_onnx"
grep -E "max_batch_size|max_queue_delay" model_repository/openclip_vit_b32/config.pbtxt | \
    sed 's/^/    /'

echo -e "\n  ${BLUE}TensorRT model (optimized):${NC}"
echo -e "    Platform: tensorrt_plan"
grep -E "max_batch_size|max_queue_delay" model_repository/openclip_vit_b32_trt/config.pbtxt | \
    sed 's/^/    /'

echo -e "\n  ${BLUE}TRT will be built on first startup with:${NC}"
echo -e "    FP16: enabled (TRT_FP16=1)"
echo -e "    Max batch: 32"
echo -e "    Opt batch: 16"

# Step 3: Build and push
echo -e "\n${YELLOW}Step 3: Building Docker image for linux/amd64...${NC}"
echo -e "${BLUE}This may take 5-10 minutes...${NC}"

docker buildx build --platform linux/amd64 \
  -f Dockerfile.tensorrt \
  -t ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG} \
  --push \
  .

echo -e "\n${GREEN}✓ Build complete and pushed to Docker Hub${NC}"

# Step 4: Instructions
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ TensorRT Deployment Image Ready${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Docker Image: ${GREEN}${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e ""
echo -e "${YELLOW}What happens on first startup:${NC}"
echo -e "  1. Container detects GPU architecture"
echo -e "  2. Converts ONNX → TensorRT engine (FP16) — ~2-10 min"
echo -e "  3. Starts Triton with BOTH models:"
echo -e "     - openclip_vit_b32     (ONNX Runtime)"
echo -e "     - openclip_vit_b32_trt (TensorRT)"
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
echo -e "3. ${RED}IMPORTANT:${NC} Wait for TRT engine build (~2-10 min after start)"
echo -e "   Check status: ${GREEN}curl http://<ip>:<port>/v2/health/ready${NC}"
echo -e ""
echo -e "4. Verify both models loaded:"
echo -e "   ${GREEN}curl http://<ip>:<port>/v2/models/openclip_vit_b32${NC}"
echo -e "   ${GREEN}curl http://<ip>:<port>/v2/models/openclip_vit_b32_trt${NC}"
echo -e ""
echo -e "5. Run 3-way benchmark:"
echo -e "   ${GREEN}python scripts/benchmark_tensorrt.py \\${NC}"
echo -e "   ${GREEN}  --triton-url http://<ip>:<port> --iterations 30${NC}"
echo -e ""
echo -e "${YELLOW}Environment variable overrides:${NC}"
echo -e "  TRT_FP16=0          Disable FP16 (use FP32)"
echo -e "  SKIP_TRT_BUILD=1    Skip TRT build (ONNX-only mode)"
echo -e "  TRT_MAX_BATCH=64    Increase max batch (needs more VRAM)"
echo -e ""
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Cost Estimate:${NC} ~\$0.20-0.80/hour depending on GPU"
echo -e "${YELLOW}First startup:${NC} +2-10 min for TRT engine build"
echo -e "${BLUE}========================================${NC}"
