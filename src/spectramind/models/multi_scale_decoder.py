import torch
import torch.nn as nn


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder producing mean spectra μ.
    Aggregates latent embeddings from FGS1 + AIRS.
    """

    def __init__(self, hidden_dim=128, output_bins=283):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_bins)

    def forward(self, fgs_latent, airs_latent):
        fgs_feat = fgs_latent.mean(dim=1)
        airs_feat = airs_latent.mean(dim=0).unsqueeze(0).expand(fgs_feat.size(0), -1)
        h = torch.cat([fgs_feat, airs_feat], dim=-1)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)

