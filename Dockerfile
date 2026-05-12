# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency metadata first
COPY pyproject.toml uv.lock* ./

# Install all deps EXCEPT torch
# torch already comes from CUDA runtime stage
RUN uv sync --frozen --no-dev --no-install-package torch

# Copy project
COPY . .


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: CUDA Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --from=builder /app /app

# Default command
CMD ["python", "main.py"]