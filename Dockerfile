# Multi-stage Dockerfile for Inference Service
# Optimized for both CPU and GPU deployment

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements
COPY requirements.txt requirements-ml.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Environment variables for model configuration
ENV MODEL_NAME=ViT-B-32
ENV MODEL_PRETRAINED=openai
ENV HOST=127.0.0.1
ENV PORT=8002
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE ${PORT}

# Run inference service
CMD ["sh", "-c", "python -m src.inference_service.server --host ${HOST} --port ${PORT} --model-name ${MODEL_NAME} --pretrained ${MODEL_PRETRAINED} --log-level ${LOG_LEVEL}"]
