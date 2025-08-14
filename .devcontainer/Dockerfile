# SpectraMind V50 — Dev Container Dockerfile
# CPU-friendly default (works in Codespaces & locally). GPU use: run with `--gpus all` if host supports NVIDIA.
ARG PYTHON_VERSION=3.11
FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=1.8.3 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps commonly needed for data science, plotting, and building Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates build-essential pkg-config \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev \
    libgl1 libglib2.0-0 libjpeg-turbo-progs zstd unzip \
    graphviz ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry (managed virtualenvs inside project)
USER vscode
ENV POETRY_HOME=/home/vscode/.local/share/pypoetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION} && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/vscode/.bashrc

# Pre-create a writable workspace and cache dirs
WORKDIR /workspaces/project
RUN mkdir -p /home/vscode/.cache/pip /home/vscode/.cache/pypoetry && \
    chmod -R 0775 /home/vscode/.cache

# Default shell
SHELL ["/bin/bash", "-lc"]
