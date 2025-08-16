from __future__ import annotations

import math


class EarlyStopping:
    """
    Early stopping based on a monitored value.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        assert mode in ("min", "max")
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience
        return self.should_stop
