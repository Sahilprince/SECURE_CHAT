#!/bin/bash

echo "🚀 Deploying SecureVault Backend to Production"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "production_backend.py" ]; then
    print_error "production_backend.py not found. Are you in the right directory?"
    exit 1
fi

# Copy production backend as main.py for deployment
cp production_backend.py main.py
print_success "Prepared main.py for deployment"

echo ""
echo "📋 Deployment Options:"
echo "1. Railway (Recommended)"
echo "2. Heroku"
echo "3. Docker"
echo "4. Local Testing"
echo ""

read -p "Choose deployment option (1-4): " choice

case $choice in
    1)
        echo "🚂 Deploying to Railway..."
        echo ""
        echo "Prerequisites:"
        echo "1. Install Railway CLI: curl -fsSL https://railway.app/install.sh | sh"
        echo "2. Login: railway login"
        echo "3. Create project: railway new"
        echo ""
        echo "Commands to run:"
        echo "railway login"
        echo "railway new"
        echo "railway add"
        echo "railway deploy"
        echo ""
        print_warning "Make sure to set environment variables in Railway dashboard!"
        ;;
    2)
        echo "🟣 Deploying to Heroku..."
        echo ""
        echo "Prerequisites:"
        echo "1. Install Heroku CLI"
        echo "2. Login: heroku login"
        echo ""
        echo "Commands to run:"
        echo "heroku create your-securevault-api"
        echo "heroku config:set MONGODB_URL='your-mongodb-url'"
        echo "heroku config:set JWT_SECRET='your-jwt-secret'"
        echo "git add ."
        echo "git commit -m 'Deploy to Heroku'"
        echo "git push heroku main"
        ;;
    3)
        echo "🐳 Building Docker image..."
        if command -v docker &> /dev/null; then
            docker build -t securevault-api .
            print_success "Docker image built successfully"
            echo ""
            echo "To run locally:"
            echo "docker run -p 8000:8000 --env-file .env securevault-api"
            echo ""
            echo "To deploy to cloud:"
            echo "docker tag securevault-api your-registry/securevault-api"
            echo "docker push your-registry/securevault-api"
        else
            print_error "Docker not found. Please install Docker first."
        fi
        ;;
    4)
        echo "💻 Starting local testing server..."
        echo ""
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            print_success "Created virtual environment"
        fi

        source venv/bin/activate
        pip install -r requirements.txt
        print_success "Installed dependencies"

        echo ""
        print_success "Starting server on http://localhost:8000"
        echo "📖 API docs available at: http://localhost:8000/docs"
        echo "🔍 Health check: http://localhost:8000/health"
        echo ""
        python main.py
        ;;
    *)
        print_error "Invalid option selected"
        exit 1
        ;;
esac
