import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEDecoder(nn.Module):
    """
    Mixture-of-Experts decoder for μ spectra.
    Allows SHAP overlays and attention tracing per expert.

    Features:
    - Expert selection with soft attention.
    - Symbolic overlay logging hooks.
    """

    def __init__(self, hidden_dim=128, output_bins=283, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, output_bins) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return torch.sum(outputs * gate_weights.unsqueeze(1), dim=-1), gate_weights

