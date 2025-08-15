import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)


class PhotometryExtractor:
    """
    Photometry extractor stub for FGS1/AIRS:
    - In production, replace the random generator with real aperture/trace extraction.
    - This implementation writes deterministic pseudo-data for pipeline wiring & CI sanity.
    """

    def __init__(self, config: Dict):
        self.config = dict(config or {})
        self.random_seed = int(self.config.get("random_seed", 1337))
        self.planets: List[str] = list(
            self.config.get("planets", ["planet_001", "planet_002", "planet_003"])
        )
        self.samples_per_planet = int(self.config.get("samples_per_planet", 1024))

    def extract(self, raw_dir: str, out_dir: Path) -> Dict[str, np.ndarray]:
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.random_seed)

        light_curves: Dict[str, np.ndarray] = {}
        for pid in self.planets:
            # Generate a smooth synthetic baseline + tiny noise to emulate a light curve segment
            x = np.linspace(0, 1, self.samples_per_planet, dtype=np.float64)
            baseline = 1.0 + 0.0005 * np.sin(
                2 * np.pi * 3 * x
            )  # ppm-scale sin variation
            noise = rng.normal(0.0, 2e-4, size=self.samples_per_planet)
            lc = baseline + noise
            light_curves[p_id := pid] = lc.astype(np.float32)
            np.save(out_dir / f"{p_id}.npy", lc.astype(np.float32))

        index_path = out_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(light_curves.keys())), f, indent=2)

        log.info(
            "Photometry extraction complete",
            extra={"n_planets": len(light_curves), "out_dir": str(out_dir)},
        )
        return light_curves
