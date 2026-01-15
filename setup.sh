#!/bin/bash

# Setup script for Synthetic Review Generator

echo "=========================================="
echo "Synthetic Review Generator - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Dependencies installed"
echo ""

# Setup environment file
echo "Setting up environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys:"
    echo "   - GEMINI_API_KEY (required for Gemini)"
    echo "   - HUGGINGFACE_TOKEN (optional for Hugging Face)"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Check for API keys
echo "Checking configuration..."
if grep -q "your_gemini_api_key_here" .env 2>/dev/null; then
    echo "⚠️  Gemini API key not configured"
    echo "   Get your free API key: https://makersuite.google.com/app/apikey"
else
    echo "✓ Gemini API key configured"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your Gemini API key"
echo "2. Run: python cli.py generate -p gemini -n 10"
echo "3. Check MIGRATION.md for detailed usage"
echo ""
echo "Quick test:"
echo "  python example.py"
echo ""
