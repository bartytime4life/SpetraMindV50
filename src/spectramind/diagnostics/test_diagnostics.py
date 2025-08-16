import numpy as np

from . import symbolic_loss


def test_symbolic_loss_nonnegativity():
    mu = np.array([1.0, -0.5, 2.0])
    loss = symbolic_loss.symbolic_nonnegativity(mu)
    assert loss > 0
