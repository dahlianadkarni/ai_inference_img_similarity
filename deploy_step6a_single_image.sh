#!/bin/bash
# =============================================================================
# Build and Deploy Step 6A All-in-One Image
# =============================================================================
#
# This script builds a single Docker image containing all 3 backends
# and pushes it to Docker Hub for Vast.ai deployment.
#
# Usage:
#   ./deploy_step6a_single_image.sh
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 6A: Build All-in-One Image${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration
DOCKER_USER="${DOCKER_USER:-dahlianadkarni}"
IMAGE_NAME="photo-duplicate-step6a"
IMAGE_TAG="latest"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "\n${YELLOW}Docker Image: ${FULL_IMAGE}${NC}\n"

# =============================================================================
# Step 1: Verify Required Files
# =============================================================================

echo -e "${YELLOW}Step 1: Verifying required files...${NC}"

REQUIRED_FILES=(
    "Dockerfile.step6a-all"
    "scripts/step6a_entrypoint.sh"
    "requirements.txt"
    "requirements-ml.txt"
    "src/inference_service/server.py"
    "model_repository/openclip_vit_b32/1/model.onnx"
    "model_repository/openclip_vit_b32/config.pbtxt"
    "model_repository/openclip_vit_b32_trt/config.pbtxt"
)

all_exist=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ] && [ ! -d "$file" ]; then
        echo -e "${RED}✗ Missing: $file${NC}"
        all_exist=false
    else
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

if [ "$all_exist" = false ]; then
    echo -e "\n${RED}Some required files are missing. Please check the list above.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required files present${NC}"

# =============================================================================
# Step 2: Check ONNX Model
# =============================================================================

echo -e "\n${YELLOW}Step 2: Checking ONNX model...${NC}"

ONNX_MODEL="model_repository/openclip_vit_b32/1/model.onnx"

if [ ! -f "$ONNX_MODEL" ]; then
    echo -e "${RED}✗ ONNX model not found: $ONNX_MODEL${NC}"
    echo -e "${YELLOW}Exporting model...${NC}"
    python scripts/export_to_onnx.py
fi

ONNX_SIZE=$(du -h "$ONNX_MODEL" | cut -f1)
echo -e "${GREEN}✓ ONNX model ready ($ONNX_SIZE)${NC}"

# =============================================================================
# Step 3: Build Docker Image
# =============================================================================

echo -e "\n${YELLOW}Step 3: Building Docker image (linux/amd64)...${NC}"
echo -e "${YELLOW}This will take 5-10 minutes...${NC}\n"

# Build for linux/amd64 (Vast.ai architecture)
docker buildx build --platform linux/amd64 \
    -f Dockerfile.step6a-all \
    -t "$FULL_IMAGE" \
    --load \
    .

echo -e "\n${GREEN}✓ Build complete${NC}"

# =============================================================================
# Step 4: Test Image Locally (Optional)
# =============================================================================

echo -e "\n${YELLOW}Step 4: Testing image locally...${NC}"

# Stop any existing test container
docker rm -f step6a-test 2>/dev/null || true

echo -e "${YELLOW}Starting test container...${NC}"
docker run -d \
    --name step6a-test \
    -p 18002:8002 \
    -p 18010:8010 \
    -p 18011:8011 \
    -p 18012:8012 \
    -p 18020:8020 \
    -p 18021:8021 \
    -p 18022:8022 \
    "$FULL_IMAGE"

echo -e "${YELLOW}Waiting for services to start (60s)...${NC}"
sleep 60

# Check PyTorch
echo -n "Testing PyTorch... "
if curl -sf http://localhost:18002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Check Triton ONNX
echo -n "Testing Triton ONNX... "
if curl -sf http://localhost:18010/v2/health/ready > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Check Triton TRT (may not be ready yet)
echo -n "Testing Triton TRT... "
if curl -sf http://localhost:18020/v2/health/ready > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⏳ (still compiling, check logs)${NC}"
fi

# Show logs
echo -e "\n${YELLOW}Recent logs:${NC}"
docker logs --tail 20 step6a-test

# Cleanup
docker stop step6a-test > /dev/null 2>&1
docker rm step6a-test > /dev/null 2>&1

echo -e "\n${GREEN}✓ Local test complete${NC}"

# =============================================================================
# Step 5: Push to Docker Hub
# =============================================================================

echo -e "\n${YELLOW}Step 5: Pushing to Docker Hub...${NC}"

# Login check
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo -e "${YELLOW}Docker Hub login required${NC}"
    docker login
fi

# Push
docker push "$FULL_IMAGE"

echo -e "\n${GREEN}✓ Image pushed to Docker Hub${NC}"

# =============================================================================
# Step 6: Deployment Instructions
# =============================================================================

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Instructions${NC}"
echo -e "${BLUE}========================================${NC}"

cat << EOF

Image: ${FULL_IMAGE}

Deploy on Vast.ai:
------------------
1. Go to https://cloud.vast.ai/
2. Create new instance:
   - GPU: RTX A4000, A5000, or similar (1 GPU)
   - Docker Image: ${FULL_IMAGE}
   - Disk Space: 50GB+
   - Open ports: 8002, 8010, 8011, 8012, 8020, 8021, 8022

3. Map ports in Vast.ai dashboard:
   Container Port → Vast.ai Public Port
   8002  → <your_choice_1>  (PyTorch)
   8010  → <your_choice_2>  (Triton ONNX HTTP)
   8011  → <your_choice_3>  (Triton ONNX gRPC)
   8012  → <your_choice_4>  (Triton ONNX Metrics)
   8020  → <your_choice_5>  (Triton TRT HTTP)
   8021  → <your_choice_6>  (Triton TRT gRPC)
   8022  → <your_choice_7>  (Triton TRT Metrics)

4. Wait 5-10 minutes for startup (TRT compilation takes time)

5. Check health:
   curl http://<VAST_IP>:<PORT_8002>/health
   curl http://<VAST_IP>:<PORT_8010>/v2/health/ready
   curl http://<VAST_IP>:<PORT_8020>/v2/health/ready

6. Run benchmark from your Mac:
   python scripts/benchmark_all_three.py \\
     --host <VAST_IP> \\
     --pytorch-port <PORT_8002> \\
     --triton-onnx-http <PORT_8010> \\
     --triton-onnx-metrics <PORT_8012> \\
     --triton-trt-http <PORT_8020> \\
     --triton-trt-metrics <PORT_8022> \\
     --iterations 20


Troubleshooting:
----------------
- Check logs in Vast.ai web console or via SSH
- Logs are in /workspace/logs/ inside container:
  - pytorch.log
  - triton_onnx.log
  - triton_trt.log

- If TRT not ready after 10 min, check triton_trt.log for errors
- GPU memory: All 3 backends share ~8-10GB VRAM on RTX A4000


Next Steps:
-----------
1. Deploy to Vast.ai
2. Run benchmark: scripts/benchmark_all_three.py
3. Analyze results: scripts/analyze_step6a_results.py
4. Document in STEP_6A_RESULTS.md

EOF

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build and push complete!${NC}"
echo -e "${GREEN}========================================${NC}"
