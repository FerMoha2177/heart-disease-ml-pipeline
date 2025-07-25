# Production Dockerfile for Render deployment
FROM python:3.10-slim

# Set working directory
WORKDIR /opt/render/project/src

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/render/project/src \
    PORT=10000

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /opt/render/project/src
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/v1/health || exit 1

# Expose port
EXPOSE $PORT

# Production command with gunicorn
CMD ["sh", "-c", "cd api && gunicorn main:app -w ${API_WORKERS:-1} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-10000}"]