# Multi-stage Dockerfile for Generative Resume Co-Writer
# Based on Context7 research findings for Docker best practices

# Build stage - Use UV for fast Python package management
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Set work directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY requirements.txt ./

# Install Python dependencies using UV (faster than pip)
RUN uv pip install --system -r requirements.txt

# Production stage - Minimal runtime image
FROM python:3.11-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app \
    && chown -R app:app /app

# Set work directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Health check - FastAPI app should respond within 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests, os; port = os.getenv('PORT', '8000'); requests.get(f'http://localhost:{port}/health')" || exit 1

# Expose port
EXPOSE 8000

# Default command - use uvicorn for ASGI server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
