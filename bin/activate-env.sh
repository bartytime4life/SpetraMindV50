#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate
echo "[ENV] Activated SpectraMind V50 virtual environment."
python --version
python -c "import torch; print(f'[CUDA] Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}')"
