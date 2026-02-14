#!/bin/bash
# Build and test Triton Docker image locally

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Triton Inference Server - Local Build${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Export model to ONNX
echo -e "\n${YELLOW}Step 1: Exporting model to ONNX...${NC}"
if [ ! -f "model_repository/openclip_vit_b32/1/model.onnx" ]; then
    python scripts/export_to_onnx.py --test
else
    echo -e "${GREEN}✓ ONNX model already exists${NC}"
fi

# Step 2: Build Docker image
echo -e "\n${YELLOW}Step 2: Building Triton Docker image...${NC}"
docker build -f Dockerfile.triton -t photo-duplicate-triton:latest .

echo -e "\n${GREEN}✓ Build complete${NC}"

# Step 3: Run container
echo -e "\n${YELLOW}Step 3: Starting Triton server...${NC}"

# Stop and remove existing container if it exists
docker rm -f triton-inference-service 2>/dev/null || true

# Run container with port mapping
# Local ports: 8003 (HTTP), 8004 (gRPC), 8005 (metrics)
# Container ports: 8000 (HTTP), 8001 (gRPC), 8002 (metrics)
docker run -d \
  --name triton-inference-service \
  -p 8003:8000 \
  -p 8004:8001 \
  -p 8005:8002 \
  photo-duplicate-triton:latest

echo -e "\n${GREEN}✓ Container started${NC}"
echo -e "\n${YELLOW}Waiting for Triton to load models...${NC}"

# Wait for health check
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
  if curl -f http://localhost:8003/v2/health/ready > /dev/null 2>&1; then
    echo -e "\n${GREEN}✓ Triton server is ready!${NC}"
    break
  fi
  attempt=$((attempt + 1))
  echo -n "."
  sleep 3
done

if [ $attempt -eq $max_attempts ]; then
  echo -e "\n${RED}Error: Triton server did not become ready within expected time${NC}"
  echo -e "Check logs with: docker logs triton-inference-service"
  exit 1
fi

# Step 4: Test endpoints
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Triton Endpoints${NC}"
echo -e "${BLUE}========================================${NC}"

# Test health
echo -e "\n${YELLOW}1. Health check:${NC}"
curl http://localhost:8003/v2/health/ready
echo ""

# Test model readiness
echo -e "\n${YELLOW}2. Model readiness:${NC}"
curl http://localhost:8003/v2/models/openclip_vit_b32/ready
echo ""

# Test model metadata
echo -e "\n${YELLOW}3. Model metadata:${NC}"
curl http://localhost:8003/v2/models/openclip_vit_b32
echo ""

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Triton Server Running Successfully${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "HTTP Endpoint:    ${GREEN}http://localhost:8003${NC}"
echo -e "gRPC Endpoint:    ${GREEN}localhost:8004${NC}"
echo -e "Metrics:          ${GREEN}http://localhost:8005/metrics${NC}"
echo -e ""
echo -e "Model Status:     ${GREEN}http://localhost:8003/v2/models/openclip_vit_b32${NC}"
echo -e "API Docs:         ${GREEN}https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md${NC}"
echo -e ""
echo -e "View logs:        ${YELLOW}docker logs -f triton-inference-service${NC}"
echo -e "Stop service:     ${YELLOW}docker stop triton-inference-service${NC}"
echo -e "Remove:           ${YELLOW}docker rm triton-inference-service${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Test with client: ${GREEN}python scripts/test_triton_client.py${NC}"
echo -e "2. Update UI to use Triton backend"
echo -e "3. Run benchmarks: ${GREEN}python scripts/benchmark_backends.py${NC}"
