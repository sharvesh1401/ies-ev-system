#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting Phase 0 Verification...${NC}"
ERROR=0

check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}❌ Missing file: $1${NC}"
        ERROR=1
    else
        echo -e "${GREEN}✅ Found file: $1${NC}"
    fi
}

# 1. Check Key Files
echo "Checking Project Structure..."
check_file "backend/app/main.py"
check_file "backend/app/config.py"
check_file "backend/Dockerfile"
check_file "frontend/src/App.tsx"
check_file "frontend/vite.config.ts"
check_file "docker-compose.yml"
check_file ".env.example"
check_file "Makefile"

# 2. Check Docker Config
echo "Checking Docker Configuration..."
if docker-compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker Compose config is valid${NC}"
else
    echo -e "${RED}❌ Docker Compose config is INVALID${NC}"
    ERROR=1
fi

# 3. Check Dependencies
echo "Checking Dependencies..."
if grep -q "fastapi" backend/requirements.txt; then
    echo -e "${GREEN}✅ Backend requirements OK${NC}"
else
    echo -e "${RED}❌ Backend requirements missing${NC}"
    ERROR=1
fi

if [ -f "frontend/package.json" ]; then
    echo -e "${GREEN}✅ Frontend package.json OK${NC}"
else
    echo -e "${RED}❌ Frontend package.json missing${NC}"
    ERROR=1
fi

if [ $ERROR -eq 0 ]; then
    echo -e "\n${GREEN}✨ VERIFICATION SUCCESSFUL! Phase 0 is complete. ✨${NC}"
    exit 0
else
    echo -e "\n${RED}verification FAILED. Please fix errors above.${NC}"
    exit 1
fi
