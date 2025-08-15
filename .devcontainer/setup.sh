#!/usr/bin/env bash
set -euxo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git curl wget build-essential cmake ninja-build pkg-config \
    libopenblas-dev liblapack-dev libx11-dev libgl1-mesa-glx libglfw3-dev \
    libssl-dev libffi-dev libhdf5-dev graphviz ffmpeg

curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

poetry install --no-interaction --no-ansi
pip install pre-commit
pre-commit install

poetry add dvc[all] mlflow lakefs-client
pip install jupyterlab ipywidgets

python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

echo "✅ SpectraMindV50 dev container ready"
