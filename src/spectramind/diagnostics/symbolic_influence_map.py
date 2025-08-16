"""
Symbolic Influence Map
----------------------
Computes ∂L/∂μ gradients per symbolic rule and aggregates them.
Exports JSON maps, plots, and dashboard overlays.
"""

import json

import matplotlib.pyplot as plt
import numpy as np


def compute_symbolic_influence(
    mu, symbolic_loss_fn, rules, save_json="symbolic_influence.json"
):
    grads = {}
    for rule in rules:
        grad = np.gradient(symbolic_loss_fn(mu, rule))
        grads[rule] = grad.tolist()
    with open(save_json, "w") as f:
        json.dump(grads, f, indent=2)
    return grads


def plot_symbolic_influence(grads, save_png="symbolic_influence.png"):
    for rule, g in grads.items():
        plt.plot(g, label=f"Rule {rule}")
    plt.legend()
    plt.savefig(save_png)
    plt.close()
