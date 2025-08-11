#!/usr/bin/env bash
set -e

echo "Bootstrapping SpectraMind V50 environment..."

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
# pip install -r requirements.txt

echo "Environment setup complete."
