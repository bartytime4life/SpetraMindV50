import torch.nn as nn


class SpectraMindModel(nn.Module):
    """
    Base class wrapper for SpectraMind V50 models.
    Provides save/load, config hash logging, and symbolic-aware hooks.
    """

    def save(self, path):
        import torch
        torch.save(self.state_dict(), path)

    def load(self, path):
        import torch
        self.load_state_dict(torch.load(path))

