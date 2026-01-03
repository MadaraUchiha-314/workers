# Multi-stage build for the workers service
# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install poetry
RUN pip install --no-cache-dir poetry>=1.8.0

# Copy dependency files
COPY pyproject.toml poetry.lock README.md ./

# Configure poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (production only, skip installing the project itself)
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY workers/ ./workers/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the service port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the service
ENTRYPOINT ["python", "-m", "uvicorn", "workers.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

