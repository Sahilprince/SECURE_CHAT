#!/bin/bash

# SecureVault Backend Startup Script
# Handles PORT environment variable properly

# Set default port if not provided
PORT=${PORT:-8000}

echo "🚀 Starting SecureVault Backend on port $PORT"
echo "🔒 Privacy-first messaging with AI camera detection"

# Start the FastAPI application with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
