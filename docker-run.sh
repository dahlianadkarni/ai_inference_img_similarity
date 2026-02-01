#!/bin/bash
# Build and run Docker container for inference service

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building Inference Service Docker Image${NC}"
echo -e "${BLUE}========================================${NC}"

# Build the image
docker build -t photo-duplicate-inference:latest .

echo -e "\n${GREEN}✓ Build complete${NC}\n"

# Run the container
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Inference Service Container${NC}"
echo -e "${BLUE}========================================${NC}"

# Stop and remove existing container if it exists
docker rm -f inference-service 2>/dev/null || true

# Run container
docker run -d \
  --name inference-service \
  -p 8002:8002 \
  -e HOST=0.0.0.0 \
  -e MODEL_NAME="${MODEL_NAME:-ViT-B-32}" \
  -e MODEL_PRETRAINED="${MODEL_PRETRAINED:-openai}" \
  -e LOG_LEVEL="${LOG_LEVEL:-info}" \
  photo-duplicate-inference:latest

echo -e "\n${GREEN}✓ Container started${NC}"
echo -e "\n${YELLOW}Waiting for service to be ready...${NC}"

# Wait for health check
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
  if curl -f http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "\n${GREEN}✓ Service is healthy!${NC}"
    break
  fi
  attempt=$((attempt + 1))
  echo -n "."
  sleep 2
done

if [ $attempt -eq $max_attempts ]; then
  echo -e "\n${YELLOW}Warning: Service did not become healthy within expected time${NC}"
  echo -e "Check logs with: docker logs inference-service"
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Inference Service Running${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Service URL:  ${GREEN}http://localhost:8002${NC}"
echo -e "Health Check: ${GREEN}http://localhost:8002/health${NC}"
echo -e "API Docs:     ${GREEN}http://localhost:8002/docs${NC}"
echo -e ""
echo -e "View logs:    ${YELLOW}docker logs -f inference-service${NC}"
echo -e "Stop service: ${YELLOW}docker stop inference-service${NC}"
echo -e "Remove:       ${YELLOW}docker rm inference-service${NC}"
echo -e "${BLUE}========================================${NC}"
