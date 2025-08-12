from __future__ import annotations

class FlowUncertaintyHead:
    def __init__(self, out_bins: int = 283, softplus: bool = True) -> None:
        self.out_bins = out_bins
        self.softplus = softplus

    def decode_sigma(self, latent: list[float]) -> list[float]:
        return [0.1 for _ in range(self.out_bins)]
