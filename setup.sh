#!/bin/bash

echo "🔒 SecureVault Backend Setup"
echo "============================"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created virtual environment"
fi

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Setup environment
if [ ! -f ".env" ]; then
    cp .env.production .env
    echo "✅ Environment file created"
    echo "⚠️  Please edit .env with your configuration"
else
    echo "✅ Environment file exists"
fi

echo ""
echo "🚀 Setup complete! Next steps:"
echo "1. Edit .env file with your MongoDB URL and secrets"
echo "2. Run: source venv/bin/activate"  
echo "3. Run: python production_backend.py"
echo "4. Test: curl http://localhost:8000/health"
echo ""
echo "📱 For mobile testing, deploy to Railway or Heroku"
