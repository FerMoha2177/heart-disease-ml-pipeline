#!/bin/bash
echo "Starting Heart Disease ML API..."

# Use gunicorn for production
cd api && gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:10000