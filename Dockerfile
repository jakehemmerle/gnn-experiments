# =============================================================================
# GNN Experiments Docker Image
# Supports both CPU (CI) and GPU (local/production) targets
#
# Build CPU: docker build --build-arg TARGET=cpu -t gnn-experiments:cpu .
# Build GPU: docker build --build-arg TARGET=gpu -t gnn-experiments:gpu .
# =============================================================================

# Build arguments
ARG TARGET=cpu
ARG PYTHON_VERSION=3.13

# =============================================================================
# UV installer stage
# =============================================================================
FROM ghcr.io/astral-sh/uv:0.5 AS uv

# =============================================================================
# GPU Base - NVIDIA CUDA with Python 3.13
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-gpu

ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# =============================================================================
# CPU Base - Python slim image
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS base-cpu

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Builder - Install dependencies
# =============================================================================
ARG TARGET
FROM base-${TARGET} AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --extra dev

# Copy source and complete installation
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY configs/ /app/configs/
COPY pyproject.toml uv.lock README.md /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra dev

# =============================================================================
# Runtime - Final image
# =============================================================================
ARG TARGET
FROM base-${TARGET} AS runtime

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

# Copy from builder
COPY --from=builder --chown=appuser:appuser /app /app

# Create mount points for data and results
RUN mkdir -p /app/data /app/results \
    && chown -R appuser:appuser /app/data /app/results

USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Default command: run tests
CMD ["pytest", "-v"]
