#!/bin/bash
# =============================================================================
# Step 6A Entrypoint: Start all 3 backends in one container
# =============================================================================
#
# This script starts:
#   1. PyTorch FastAPI backend (port 8002)
#   2. Triton ONNX CUDA EP (ports 8010/8011/8012)
#   3. Triton TensorRT EP (ports 8020/8021/8022)
#
# All services run in the background, monitored by this script.
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 6A: Starting All 3 Backends${NC}"
echo -e "${BLUE}========================================${NC}"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"

# Model repository paths
MODEL_REPO="/models"
ONNX_MODEL="${MODEL_REPO}/openclip_vit_b32/1/model.onnx"
TRT_MODEL_DIR="${MODEL_REPO}/openclip_vit_b32_trt/1"

# =============================================================================
# GPU Check
# =============================================================================

echo -e "\n${YELLOW}Checking GPU availability...${NC}"
if nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✓ GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}✗ No GPU detected - services will fail${NC}"
fi

# =============================================================================
# Setup TRT Model (Symlink)
# =============================================================================

echo -e "\n${YELLOW}Setting up TRT model directory...${NC}"
mkdir -p "$TRT_MODEL_DIR"

if [ -f "$ONNX_MODEL" ]; then
    ln -sf "$ONNX_MODEL" "${TRT_MODEL_DIR}/model.onnx"
    echo -e "${GREEN}✓ Symlinked ONNX model for TRT EP${NC}"
else
    echo -e "${RED}✗ ONNX model not found at $ONNX_MODEL${NC}"
    exit 1
fi

# =============================================================================
# Start Backend 1: PyTorch FastAPI
# =============================================================================

echo -e "\n${YELLOW}Starting Backend 1: PyTorch FastAPI (port 8002)...${NC}"

python3 -m uvicorn src.inference_service.server:app \
    --host 0.0.0.0 \
    --port 8002 \
    --log-level info \
    > "$LOG_DIR/pytorch.log" 2>&1 &

PYTORCH_PID=$!
echo -e "${GREEN}✓ PyTorch started (PID: $PYTORCH_PID)${NC}"

# =============================================================================
# Start Backend 2: Triton ONNX CUDA EP
# =============================================================================

echo -e "\n${YELLOW}Starting Backend 2: Triton ONNX CUDA EP (ports 8010-8012)...${NC}"

# Create temp model repo for ONNX only
TRITON_ONNX_REPO="/tmp/triton_onnx_repo"
mkdir -p "$TRITON_ONNX_REPO"
cp -r "${MODEL_REPO}/openclip_vit_b32" "$TRITON_ONNX_REPO/"

tritonserver \
    --model-repository="$TRITON_ONNX_REPO" \
    --http-port=8010 \
    --grpc-port=8011 \
    --metrics-port=8012 \
    --strict-model-config=false \
    --log-verbose=0 \
    > "$LOG_DIR/triton_onnx.log" 2>&1 &

TRITON_ONNX_PID=$!
echo -e "${GREEN}✓ Triton ONNX started (PID: $TRITON_ONNX_PID)${NC}"

# =============================================================================
# Start Backend 3: Triton TensorRT EP
# =============================================================================

echo -e "\n${YELLOW}Starting Backend 3: Triton TensorRT EP (ports 8020-8022)...${NC}"
echo -e "${YELLOW}Note: First load takes 2-5 minutes to compile TRT engine${NC}"

# Create temp model repo for TRT only
TRITON_TRT_REPO="/tmp/triton_trt_repo"
mkdir -p "$TRITON_TRT_REPO"
cp -r "${MODEL_REPO}/openclip_vit_b32_trt" "$TRITON_TRT_REPO/"

tritonserver \
    --model-repository="$TRITON_TRT_REPO" \
    --http-port=8020 \
    --grpc-port=8021 \
    --metrics-port=8022 \
    --strict-model-config=false \
    --log-verbose=0 \
    > "$LOG_DIR/triton_trt.log" 2>&1 &

TRITON_TRT_PID=$!
echo -e "${GREEN}✓ Triton TRT started (PID: $TRITON_TRT_PID)${NC}"

# =============================================================================
# Summary
# =============================================================================

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}All Services Started${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "PyTorch FastAPI:      http://localhost:8002  (PID: $PYTORCH_PID)"
echo -e "Triton ONNX:          http://localhost:8010  (PID: $TRITON_ONNX_PID)"
echo -e "Triton TRT:           http://localhost:8020  (PID: $TRITON_TRT_PID)"
echo -e ""
echo -e "Logs available in: $LOG_DIR/"
echo -e "  - pytorch.log"
echo -e "  - triton_onnx.log"
echo -e "  - triton_trt.log"
echo -e ""
echo -e "${YELLOW}Waiting for services to initialize...${NC}"

# =============================================================================
# Wait and Monitor
# =============================================================================

# Wait for PyTorch to be ready
echo -n "Waiting for PyTorch... "
for i in {1..30}; do
    if curl -sf http://localhost:8002/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    sleep 2
done

# Wait for Triton ONNX to be ready
echo -n "Waiting for Triton ONNX... "
for i in {1..30}; do
    if curl -sf http://localhost:8010/v2/health/ready > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    sleep 2
done

# Wait for Triton TRT to be ready (takes longer)
echo -n "Waiting for Triton TRT (may take 2-5 min for first load)... "
for i in {1..180}; do
    if curl -sf http://localhost:8020/v2/health/ready > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    sleep 2
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All services ready!${NC}"
echo -e "${GREEN}========================================${NC}"

# =============================================================================
# Keep Container Running and Monitor Processes
# =============================================================================

cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $PYTORCH_PID $TRITON_ONNX_PID $TRITON_TRT_PID 2>/dev/null || true
    wait $PYTORCH_PID $TRITON_ONNX_PID $TRITON_TRT_PID 2>/dev/null || true
    echo -e "${GREEN}✓ All services stopped${NC}"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Monitor processes
while true; do
    # Check if any process died
    if ! kill -0 $PYTORCH_PID 2>/dev/null; then
        echo -e "${RED}✗ PyTorch died, restarting...${NC}"
        python3 -m uvicorn src.inference_service.server:app \
            --host 0.0.0.0 --port 8002 --log-level info \
            > "$LOG_DIR/pytorch.log" 2>&1 &
        PYTORCH_PID=$!
    fi
    
    if ! kill -0 $TRITON_ONNX_PID 2>/dev/null; then
        echo -e "${RED}✗ Triton ONNX died, restarting...${NC}"
        tritonserver --model-repository="$TRITON_ONNX_REPO" \
            --http-port=8010 --grpc-port=8011 --metrics-port=8012 \
            --strict-model-config=false --log-verbose=0 \
            > "$LOG_DIR/triton_onnx.log" 2>&1 &
        TRITON_ONNX_PID=$!
    fi
    
    if ! kill -0 $TRITON_TRT_PID 2>/dev/null; then
        echo -e "${RED}✗ Triton TRT died, restarting...${NC}"
        tritonserver --model-repository="$TRITON_TRT_REPO" \
            --http-port=8020 --grpc-port=8021 --metrics-port=8022 \
            --strict-model-config=false --log-verbose=0 \
            > "$LOG_DIR/triton_trt.log" 2>&1 &
        TRITON_TRT_PID=$!
    fi
    
    sleep 10
done
