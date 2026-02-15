#!/bin/bash
# =============================================================================
# Step 6A: Deploy all 3 backends to Vast.ai for comparison
# =============================================================================
#
# This script helps you deploy the 3-service setup for Step 6A benchmarking.
#
# What it does:
#   1. Verifies required Docker images are pushed to Docker Hub
#   2. Validates docker-compose-step6a.yml
#   3. Provides deployment instructions for Vast.ai
#
# Prerequisites:
#   - Docker images must be pushed to Docker Hub:
#       * dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64
#       * dahlianadkarni/photo-duplicate-triton:gpu-linux-amd64
#       * dahlianadkarni/photo-duplicate-triton:tensorrt-gpu
#   - Vast.ai instance with GPU (RTX A4000, A5000, or similar)
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
echo -e "${BLUE}Step 6A: Deploy 3-Backend Comparison${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if docker-compose file exists
if [ ! -f "docker-compose-step6a.yml" ]; then
    echo -e "${RED}✗ docker-compose-step6a.yml not found${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Step 1: Checking Docker images...${NC}"

# Required images
IMAGES=(
    "dahlianadkarni/photo-duplicate-inference:gpu-linux-amd64"
    "dahlianadkarni/photo-duplicate-triton:gpu-linux-amd64"
    "dahlianadkarni/photo-duplicate-triton:tensorrt-gpu"
)

all_exist=true

for image in "${IMAGES[@]}"; do
    echo -n "  Checking $image... "
    if docker pull "$image" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Not found on Docker Hub${NC}"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo -e "\n${RED}Some images are missing. Build and push them first:${NC}"
    echo "  ./deploy_step3.sh       # PyTorch image"
    echo "  ./deploy_triton_gpu.sh  # Triton ONNX image"
    echo "  ./deploy_tensorrt_gpu.sh # Triton TRT image"
    exit 1
fi

echo -e "\n${GREEN}✓ All Docker images are available${NC}"

echo -e "\n${YELLOW}Step 2: Validating docker-compose file...${NC}"
docker-compose -f docker-compose-step6a.yml config > /dev/null 2>&1
echo -e "${GREEN}✓ docker-compose-step6a.yml is valid${NC}"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Instructions${NC}"
echo -e "${BLUE}========================================${NC}"

cat << 'EOF'

Option 1: Deploy via Vast.ai Web UI
------------------------------------
1. Go to https://cloud.vast.ai/
2. Create new instance:
   - GPU: RTX A4000, A5000, or similar (1 GPU)
   - Image: ubuntu:22.04 or nvidia/cuda:12.4.0-runtime-ubuntu22.04
   - Disk Space: 50GB+
   - Open ports: 8002, 8010, 8011, 8012, 8020, 8021, 8022
   
3. Once instance is running, SSH in:
   ssh -p <SSH_PORT> root@<INSTANCE_IP>

4. Install Docker Compose:
   apt-get update
   apt-get install -y docker-compose-plugin

5. Upload docker-compose file:
   # From your Mac (in new terminal):
   scp -P <SSH_PORT> docker-compose-step6a.yml root@<INSTANCE_IP>:/root/

6. Start services:
   cd /root
   docker-compose -f docker-compose-step6a.yml up -d

7. Check status:
   docker-compose -f docker-compose-step6a.yml ps
   docker logs step6a-pytorch
   docker logs step6a-triton-onnx
   docker logs step6a-triton-trt

8. Note: Triton TRT will take 2-5 minutes to compile on first start.
   Watch logs: docker logs -f step6a-triton-trt


Option 2: Run Benchmark from Your Mac
--------------------------------------
After deployment, map Vast.ai ports and run benchmark from your Mac:

Example port mapping (from Vast.ai dashboard):
  Container Port → Vast.ai Public Port
  8002           → 12345
  8010           → 23456
  8012           → 23458
  8020           → 34567
  8022           → 34569

Run benchmark:
  python scripts/benchmark_all_three.py \
    --host <INSTANCE_IP> \
    --pytorch-port 12345 \
    --triton-onnx-http 23456 --triton-onnx-metrics 23458 \
    --triton-trt-http 34567 --triton-trt-metrics 34569 \
    --iterations 20


Option 3: Run Benchmark Locally (After SSH)
--------------------------------------------
For more accurate results (no network latency), SSH into instance:

1. SSH in:
   ssh -p <SSH_PORT> root@<INSTANCE_IP>

2. Install Python dependencies:
   apt-get install -y python3-pip
   pip3 install numpy pillow tritonclient[http] requests

3. Copy benchmark script:
   # From your Mac:
   scp -P <SSH_PORT> scripts/benchmark_all_three_local.py root@<INSTANCE_IP>:/root/

4. Run benchmark:
   cd /root
   python3 benchmark_all_three_local.py --iterations 50

5. Copy results back:
   # From your Mac:
   scp -P <SSH_PORT> root@<INSTANCE_IP>:/root/benchmark_results/step6a_local_comparison.json ./benchmark_results/


Health Checks
-------------
PyTorch:     curl http://localhost:8002/health
Triton ONNX: curl http://localhost:8010/v2/health/ready
Triton TRT:  curl http://localhost:8020/v2/health/ready

Model Status:
  curl http://localhost:8010/v2/models/openclip_vit_b32/ready
  curl http://localhost:8020/v2/models/openclip_vit_b32_trt/ready


Troubleshooting
---------------
Q: Triton TRT not starting?
A: First start takes 2-5 min for TRT engine compilation. Check logs:
   docker logs -f step6a-triton-trt

Q: Out of memory?
A: All 3 services share 1 GPU. If using RTX A4000 (16GB), should be fine.
   Monitor with: nvidia-smi

Q: Services not accessible?
A: Check Vast.ai port mapping in web UI. All ports must be exposed.

EOF

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nNext steps:"
echo -e "  1. Deploy to Vast.ai (see instructions above)"
echo -e "  2. Wait for all 3 services to be ready"
echo -e "  3. Run benchmark (remote or local)"
echo -e "  4. Analyze results in ${YELLOW}benchmark_results/step6a_*_comparison.json${NC}"
