"""
Contract tests around Hydra integration without invoking full hydra.main().

We validate:
- The decorated functions exist and remain importable.
- The functions accept no runtime args (Hydra supplies cfg), so signature has zero params.
This is a lightweight guard that helps detect accidental signature regressions.
"""

import importlib
import inspect

import pytest

TARGETS = [
    "src.spectramind.training.train_v50:main",
    "src.spectramind.training.predict_v50:main",
    "src.spectramind.training.train_mae_v50:main",
    "src.spectramind.training.train_contrastive_v50:main",
    "src.spectramind.training.tune_temperature_v50:main",
    "src.spectramind.training.train_corel:main",
    "src.spectramind.training.run_ablation_trials:main",
]


def _resolve(target: str):
    mod_name, fn_name = target.split(":")
    if mod_name.endswith("train_v50"):
        pytest.importorskip("torch")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def test_hydra_decorated_functions_present_and_zero_args():
    for t in TARGETS:
        fn = _resolve(t)
        sig = inspect.signature(fn)
        # Hydra wraps the function; at import time the wrapper typically has no required parameters
        # (Hydra injects cfg at runtime). We ensure there are no required positional-only parameters.
        params = [
            p
            for p in sig.parameters.values()
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert (
            len(params) == 0
        ), f"Hydra main wrapper for {t} should have no required args; got {sig}"
