# -------- Base: NVIDIA CUDA runtime on Ubuntu LTS --------
ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS base

# -------- Build args / metadata --------
ARG USERNAME=sm
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.10
ARG POETRY_VERSION=1.8.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_COLOR=1 \
    TERM=xterm-256color \
    # Hydra/CLI friendly
    TOKENIZERS_PARALLELISM=false \
    # Matplotlib headless default
    MPLBACKEND=Agg \
    # CUDA sanity
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    # Optional offline HF cache (flip to "1" if you pre-seed)
    HF_HUB_OFFLINE=0

# -------- OS packages --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git git-lfs tini \
    build-essential pkg-config cmake \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip python3-venv \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Default python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
 && python -m pip install --upgrade pip wheel

# -------- Create non-root user --------
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

# -------- Enable Poetry (system-wide) --------
ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION} \
 && ln -sf ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

# -------- Workdir & layer-cached dep install --------
WORKDIR /app

# Only copy lockfiles first to leverage Docker layer caching
COPY pyproject.toml poetry.lock* ./ 

# Install Python deps without building project yet
# We explicitly install a CUDA-matched PyTorch to guarantee GPU capability
ARG TORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu121"
RUN poetry run python -m pip install --upgrade pip \
 && poetry run python -m pip install --extra-index-url ${TORCH_CUDA_CHANNEL} \
      torch torchvision torchaudio \
 && poetry install --no-root --with main,dev

# (Optional) extra CLI/MLOps helpers frequently used by SpectraMind
# Keep here so they cache independently of source changes
RUN poetry run python -m pip install dvc[ssh,s3,gdrive] mlflow rich typer[all]

# -------- Copy source last (changes most frequently) --------
COPY . .

# Precreate outputs and cache dirs with correct ownership
RUN mkdir -p /app/outputs /app/.cache \
 && chown -R ${USERNAME}:${USERNAME} /app

USER ${USERNAME}

# -------- Sanity checks (fail-fast) --------
# This won’t break the build if spectramind module isn’t fully wired yet; flip to `python -m spectramind --version` once ready
RUN python -c "import sys; print('Python', sys.version)" \
 && poetry --version

# -------- Runtime defaults --------
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
# Default to help; override with `docker run ... python -m spectramind <cmd>`
CMD ["python", "-m", "spectramind", "--help"]