"""
tune_temperature_v50.py

Temperature scaling tuner for uncertainty calibration.
Hydra-driven, COREL-compatible, logs symbolic-aware calibration metrics.
"""

import hydra
from omegaconf import DictConfig
import logging
from spectramind.training.utils import tune_temperature_from_config

logger = logging.getLogger("spectramind.training")

@hydra.main(config_path="../../configs", config_name="config_v50.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info("Running temperature scaling tuning...")
    tune_temperature_from_config(cfg)

if __name__ == "__main__":
    main()
