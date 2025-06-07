FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt

# Second stage: runtime image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH" \
    BTC_STACK_BUILDER_CONFIG_DIR=/app/config

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user to run the application
RUN groupadd -r botuser && useradd -r -g botuser -m -d /home/botuser botuser

# Create necessary directories with proper permissions
RUN mkdir -p /app/config /app/logs /app/data && \
    chown -R botuser:botuser /app

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Copy application code
COPY --chown=botuser:botuser . /app/

# Set working directory
WORKDIR /app

# Create volume mount points
VOLUME ["/app/config", "/app/logs", "/app/data"]

# Switch to non-root user
USER botuser

# Expose Prometheus metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# Set the entrypoint
ENTRYPOINT ["python", "-m", "btc_stack_builder.main"]

# Default command (can be overridden)
CMD ["--environment", "production"]
