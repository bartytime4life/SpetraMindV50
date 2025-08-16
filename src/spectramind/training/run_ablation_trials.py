"""
run_ablation_trials.py

Automated ablation runner for SpectraMind V50.
Mutates Hydra configs, runs symbolic-aware trials, exports leaderboard.
"""

import hydra
from omegaconf import DictConfig
import logging
from spectramind.training.utils import run_ablation_trials_from_config

logger = logging.getLogger("spectramind.training")

@hydra.main(config_path="../../configs", config_name="ablation.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info("Running ablation trials...")
    run_ablation_trials_from_config(cfg)

if __name__ == "__main__":
    main()
