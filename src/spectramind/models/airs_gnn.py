import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import logging

logger = logging.getLogger(__name__)


class AIRSSpectralGNN(nn.Module):
    """
    Ariel IR Spectrometer (AIRS) Graph Neural Network.
    Graph built with edges = wavelength proximity + molecule region + detector.

    Features:
    - Edge-feature aware attention (distance, molecule type).
    - Configurable GNN backbone (GATConv default).
    - Symbolic-aware spectral smoothing hooks.
    """

    def __init__(self, input_dim=32, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(GATConv(in_dim, hidden_dim, edge_dim=3))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        logger.info("Initialized AIRSSpectralGNN")

    def forward(self, x, edge_index, edge_attr):
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = self.dropout(torch.relu(x))
        return self.norm(x)

