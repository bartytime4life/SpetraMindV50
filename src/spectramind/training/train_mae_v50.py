"""
train_mae_v50.py

Masked Autoencoder (MAE) pretraining for SpectraMind V50.
Supports SNR-aware masking, symbolic-aware masking, curriculum schedules,
entropy logging, and Hydra config overrides.
"""

import hydra
from omegaconf import DictConfig
import logging
from spectramind.training.utils import train_mae_from_config

logger = logging.getLogger("spectramind.training")

@hydra.main(config_path="../../configs", config_name="pretrain_mae.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info("Launching MAE pretraining...")
    train_mae_from_config(cfg)

if __name__ == "__main__":
    main()
