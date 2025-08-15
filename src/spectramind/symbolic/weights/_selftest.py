"""Lightweight smoke test for local validation. Safe to run in CI."""

from __future__ import annotations

from .loader import compose_weights
from .validator import validate_weight_config


def run() -> None:
    w = compose_weights(profile_name="hot_jupiter")
    validate_weight_config(w)
    print("OK", len(w))


if __name__ == "__main__":  # pragma: no cover
    run()
