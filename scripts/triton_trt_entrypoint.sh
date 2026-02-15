#!/bin/bash
# =============================================================================
# Triton + TensorRT EP Entrypoint Script
# =============================================================================
#
# This script runs at container startup and:
#   1. Checks GPU availability
#   2. Sets up the TRT model directory (symlinks the ONNX model)
#   3. Creates TRT engine cache directory
#   4. Starts Triton Inference Server with both ONNX and TRT EP models
#
# The TensorRT Execution Provider within ONNX Runtime automatically
# compiles TRT subgraphs on first load (~2-5 min), then caches them.
# No separate trtexec or conversion script is needed.
#
# Environment variables:
#   MODEL_REPO  - Model repository path (default: /models)
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MODEL_REPO="${MODEL_REPO:-/models}"
ONNX_MODEL="${MODEL_REPO}/openclip_vit_b32/1/model.onnx"
TRT_MODEL_DIR="${MODEL_REPO}/openclip_vit_b32_trt/1"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Triton + TensorRT EP Startup${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Check GPU
echo -e "\n${YELLOW}Step 1: GPU Detection${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}✓ GPU: ${GPU_NAME} (${GPU_MEMORY})${NC}"
else
    echo -e "${RED}✗ No GPU detected (nvidia-smi not found)${NC}"
    echo -e "${YELLOW}TRT EP requires a GPU. Falling back to ONNX-only mode.${NC}"
    # Remove TRT model so Triton doesn't try to load it
    rm -rf "${MODEL_REPO}/openclip_vit_b32_trt"
fi

# Step 2: Check ONNX model
echo -e "\n${YELLOW}Step 2: ONNX Model Check${NC}"
if [ -f "$ONNX_MODEL" ]; then
    ONNX_SIZE=$(du -h "$ONNX_MODEL" | cut -f1)
    echo -e "${GREEN}✓ ONNX model found: ${ONNX_MODEL} (${ONNX_SIZE})${NC}"
else
    echo -e "${RED}✗ ONNX model not found: ${ONNX_MODEL}${NC}"
    echo -e "${RED}Cannot proceed without ONNX model${NC}"
    exit 1
fi

# Step 3: Set up TRT model (symlink to same ONNX file)
echo -e "\n${YELLOW}Step 3: TensorRT EP Model Setup${NC}"
if [ -d "${MODEL_REPO}/openclip_vit_b32_trt" ]; then
    mkdir -p "$TRT_MODEL_DIR"

    # Symlink the ONNX model so both configs share the same file
    if [ ! -f "${TRT_MODEL_DIR}/model.onnx" ]; then
        ln -sf "$ONNX_MODEL" "${TRT_MODEL_DIR}/model.onnx"
        echo -e "${GREEN}✓ Symlinked ONNX model for TRT EP model${NC}"
    else
        echo -e "${GREEN}✓ TRT EP model already has ONNX file${NC}"
    fi

    # Create TRT engine cache directory
    mkdir -p /tmp/trt_cache
    echo -e "${GREEN}✓ TRT engine cache dir: /tmp/trt_cache${NC}"
    echo -e "${YELLOW}  Note: First load will compile TRT subgraphs (~2-5 min)${NC}"
    echo -e "${YELLOW}  Subsequent loads will use cached engines.${NC}"
fi

# Step 4: List available models
echo -e "\n${YELLOW}Step 4: Available Models${NC}"
for model_dir in "${MODEL_REPO}"/*/; do
    model_name=$(basename "$model_dir")
    if [ -f "${model_dir}config.pbtxt" ]; then
        # Show platform and whether TRT EP is enabled
        platform=$(grep "^platform:" "${model_dir}config.pbtxt" | head -1 | awk '{print $2}' | tr -d '"')
        if grep -q "tensorrt" "${model_dir}config.pbtxt" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} ${model_name} (${platform} + TensorRT EP, FP16)"
        else
            echo -e "  ${GREEN}✓${NC} ${model_name} (${platform})"
        fi
    fi
done

# Step 5: Start Triton
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Triton Inference Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

exec tritonserver \
    --model-repository="$MODEL_REPO" \
    --strict-model-config=false \
    --log-verbose=1
