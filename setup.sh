#!/bin/bash

# Exit on error
set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
uv pip install ruff

# Run ruff check
echo "Running initial ruff check..."
ruff check .

echo "Setup complete! Activate the virtual environment with: source .venv/bin/activate" 