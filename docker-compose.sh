#!/bin/bash
# Quick script to use docker-compose

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "$1" in
  up|start)
    echo -e "${BLUE}Starting services with docker-compose...${NC}"
    docker-compose up -d
    echo -e "\n${GREEN}✓ Services started${NC}"
    echo -e "View logs: ${YELLOW}docker-compose logs -f${NC}"
    ;;
  
  down|stop)
    echo -e "${BLUE}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
    ;;
  
  restart)
    echo -e "${BLUE}Restarting services...${NC}"
    docker-compose restart
    echo -e "${GREEN}✓ Services restarted${NC}"
    ;;
  
  logs)
    docker-compose logs -f
    ;;
  
  build)
    echo -e "${BLUE}Building images...${NC}"
    docker-compose build
    echo -e "${GREEN}✓ Build complete${NC}"
    ;;
  
  status)
    docker-compose ps
    ;;
  
  *)
    echo "Usage: $0 {up|down|restart|logs|build|status}"
    echo ""
    echo "Commands:"
    echo "  up/start   - Start services"
    echo "  down/stop  - Stop services"
    echo "  restart    - Restart services"
    echo "  logs       - View logs"
    echo "  build      - Build images"
    echo "  status     - Show service status"
    exit 1
    ;;
esac
