#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "==============================================="
echo "   Gravitational Lensing Toolkit (Mac/Linux)"
echo "   Launching Streamlit App..."
echo "==============================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
fi

# Run Streamlit
python3 -m streamlit run app/Home.py
