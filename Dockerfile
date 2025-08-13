# SpectraMind V50 — Docker (GPU-ready base)
#
# Build:
#   docker build -t spectramindv50:dev .
#
# Run (with GPU):
#   docker run --gpus all -it --rm -v $PWD:/workspace spectramindv50:dev bash

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Optional: use Poetry inside container
RUN pip3 install --no-cache-dir poetry

COPY pyproject.toml README.md ./
RUN poetry config virtualenvs.create false && poetry install --no-root

COPY . .
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
