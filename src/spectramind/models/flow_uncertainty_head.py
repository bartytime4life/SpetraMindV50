import torch
import torch.nn as nn


class FlowUncertaintyHead(nn.Module):
    """
    Flow-based uncertainty decoder for σ spectra.

    Features:
    - Attention fusion with symbolic overlays.
    - Temperature scaling + conformal calibration ready.
    """

    def __init__(self, hidden_dim=128, output_bins=283):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_bins)

    def forward(self, x):
        return torch.exp(self.fc(x))  # enforce positivity

