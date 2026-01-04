#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "==============================================="
echo "   Gravitational Lensing Toolkit (Mac/Linux)"
echo "   Launching API Server..."
echo "==============================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
fi

# Run API
echo "Starting Uvicorn..."
python3 -m uvicorn api.main:app --reload
