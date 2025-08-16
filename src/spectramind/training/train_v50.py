"""
train_v50.py

Main entrypoint for full SpectraMind V50 training.
Hydra-safe configuration, MLflow logging, symbolic loss integration,
curriculum phases, AMP support, checkpointing, and config hashing.
"""

import hydra
from omegaconf import DictConfig
import torch
import logging
from spectramind.training.utils import train_from_config

logger = logging.getLogger("spectramind.training")

@hydra.main(config_path="../../configs", config_name="config_v50.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main Hydra launcher for SpectraMind V50 training.
    """
    logger.info("Launching SpectraMind V50 training pipeline...")
    train_from_config(cfg)

if __name__ == "__main__":
    main()
