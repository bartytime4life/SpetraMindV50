"""
train_contrastive_v50.py

Contrastive pretraining for SpectraMind V50.
Supports symbolic overlays, dual-view masking, entropy/FFT loss.
"""

import hydra
from omegaconf import DictConfig
import logging
from spectramind.training.utils import train_contrastive_from_config

logger = logging.getLogger("spectramind.training")

@hydra.main(config_path="../../configs", config_name="pretrain_contrastive.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info("Launching contrastive pretraining...")
    train_contrastive_from_config(cfg)

if __name__ == "__main__":
    main()
