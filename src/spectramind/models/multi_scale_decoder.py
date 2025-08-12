from __future__ import annotations

class MultiScaleDecoder:
    def __init__(self, out_bins: int = 283, multiscale: bool = True) -> None:
        self.out_bins = out_bins
        self.multiscale = multiscale

    def decode_mu(self, latent: list[float]) -> list[float]:
        return [0.0 for _ in range(self.out_bins)]
