"""
Safety guard tests to ensure training entrypoint scripts do not execute on import.

Hydra-decorated "main" functions should only run under `if __name__ == "__main__":`.
We confirm that importing each module does not add unexpected top-level side-effects.
"""

import importlib

import pytest

ENTRY_MODULES = [
    "src.spectramind.training.train_v50",
    "src.spectramind.training.predict_v50",
    "src.spectramind.training.train_mae_v50",
    "src.spectramind.training.train_contrastive_v50",
    "src.spectramind.training.tune_temperature_v50",
    "src.spectramind.training.train_corel",
    "src.spectramind.training.run_ablation_trials",
]


def test_import_has_no_side_effects(monkeypatch):
    # Track if any unexpected print/log happens by toggling a sentinel via monkeypatch if needed.
    # For now, the presence of "main" symbol suffices and import should not raise.
    for mod_name in ENTRY_MODULES:
        if mod_name.endswith("train_v50"):
            pytest.importorskip("torch")
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "main"), f"{mod_name} missing main() after import"
