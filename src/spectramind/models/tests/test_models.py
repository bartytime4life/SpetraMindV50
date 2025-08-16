import torch
from spectramind.models import FGS1MambaEncoder, AIRSSpectralGNN, MultiScaleDecoder


def test_fgs1_forward():
    model = FGS1MambaEncoder()
    x = torch.randn(2, 100, 32)
    out = model(x)
    assert out.shape[-1] == 128


def test_airs_forward():
    model = AIRSSpectralGNN()
    x = torch.randn(356, 32)
    edge_index = torch.randint(0, 356, (2, 500))
    edge_attr = torch.randn(500, 3)
    out = model(x, edge_index, edge_attr)
    assert out.shape[-1] == 128


def test_decoder_forward():
    fgs = torch.randn(2, 100, 128)
    airs = torch.randn(356, 128)
    dec = MultiScaleDecoder()
    out = dec(fgs, airs)
    assert out.shape[-1] == 283

