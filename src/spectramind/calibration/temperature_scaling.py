import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """
    Logit temperature scaling. Included for completeness; not actively used in this minimal pipeline
    (AIRS/FGS1 regression typically does not employ classification logits).
    """

    def __init__(self, init_temp: float = 0.0):
        """
        init_temp is in log-space; actual temperature = exp(param).
        """
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp)
        return logits / temp.clamp_min(1e-6)
