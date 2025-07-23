#!/bin/bash

# Startup script for Render deployment
echo "🚀 Starting Heart Disease ML API..."

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI application
cd api && python main.py
