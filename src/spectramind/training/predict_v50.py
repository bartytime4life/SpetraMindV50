"""
predict_v50.py

Inference script for SpectraMind V50.
Hydra-safe configs, μ/σ output, submission packaging.
"""

import hydra
from omegaconf import DictConfig
import logging
from spectramind.training.utils import predict_from_config

logger = logging.getLogger("spectramind.training")

@hydra.main(config_path="../../configs", config_name="config_v50.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info("Running inference with SpectraMind V50...")
    predict_from_config(cfg)

if __name__ == "__main__":
    main()
