"""
train_corel.py

Spectral COREL GNN training for binwise uncertainty calibration.
Supports edge-feature-aware message passing, temporal bin encoding,
TorchScript export, and symbolic-region calibration scoring.
"""

import hydra
from omegaconf import DictConfig
import logging
from spectramind.training.utils import train_corel_from_config

logger = logging.getLogger("spectramind.training")

@hydra.main(config_path="../../configs", config_name="corel.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info("Launching Spectral COREL GNN training...")
    train_corel_from_config(cfg)

if __name__ == "__main__":
    main()
