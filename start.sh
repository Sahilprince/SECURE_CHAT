#!/bin/bash

# SecureVault Backend - Production Startup Script
set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔒 SecureVault Backend Starting...${NC}"
echo -e "${BLUE}======================================${NC}"

# Set default port if not provided
PORT=${PORT:-8000}

# Validate port number
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo -e "${RED}❌ Error: Invalid PORT number: $PORT${NC}"
    exit 1
fi

echo -e "${GREEN}🌐 Port: $PORT${NC}"
echo -e "${GREEN}🏠 Host: 0.0.0.0${NC}"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ Error: main.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Application file found${NC}"

# Check environment variables (optional)
if [ -z "$MONGODB_URL" ]; then
    echo -e "${YELLOW}⚠️  Warning: MONGODB_URL not set${NC}"
fi

if [ -z "$JWT_SECRET" ]; then
    echo -e "${YELLOW}⚠️  Warning: JWT_SECRET not set${NC}"
fi

echo -e "${BLUE}🚀 Starting uvicorn server...${NC}"

# Start the FastAPI application with uvicorn
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --access-log
