#!/bin/bash

echo "=========================================="
echo "Stryktipset Predictor - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API-Football key!"
    echo "   Get your key from: https://www.api-football.com/"
    echo ""
fi

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/raw data/processed data/cache models logs

echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API_FOOTBALL_KEY"
echo "   Get it from: https://www.api-football.com/"
echo ""
echo "2. Activate virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   venv\\Scripts\\activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "3. Verify setup:"
echo "   python test_setup.py"
echo ""
echo "4. Fetch data (uses API calls!):"
echo "   python data_fetcher.py"
echo ""
echo "5. Follow the workflow:"
echo "   python data_explorer.py"
echo "   python feature_engineering.py"
echo "   python ml_predictor.py"
echo "   python main.py"
echo ""
echo "For detailed instructions, see README.md"
echo ""