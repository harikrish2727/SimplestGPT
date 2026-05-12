# ── Stage 1: install non-torch dependencies ───────────────────────────────────
# Use slim Python for the build stage — CUDA is not needed to resolve packages.
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install everything EXCEPT torch (already in the runtime base image).
# This prevents uv from pulling a CPU-only torch wheel from PyPI.
RUN uv sync --frozen --no-dev --no-install-package torch

# ── Stage 2: GPU runtime ──────────────────────────────────────────────────────
# This image ships with: CUDA 12.4, cuDNN 9, and PyTorch 2.6 (CUDA-enabled).
# NOTE: it uses Python 3.11. Relax requires-python to >=3.11 in pyproject.toml.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS runtime

WORKDIR /app

# Copy the venv with the non-torch deps from builder
COPY --from=builder /app/.venv /app/.venv

# Prepend venv to PATH so our installed packages take precedence
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy source files
COPY attention.py block.py data_preparation.py dataset.py \
     eval.py feedforward.py gpt.py train.py main.py ./

COPY input.txt ./

# Model checkpoint will be saved here — mount a host directory to persist it
VOLUME ["/app/output"]

# Pass your .env at runtime:  docker run --env-file .env ...
# Never bake secrets into the image.

CMD ["python", "main.py"]