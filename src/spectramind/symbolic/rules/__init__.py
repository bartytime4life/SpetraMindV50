# SPDX-License-Identifier: Apache-2.0

"""Package initialization for SpectraMind V50 symbolic rules.

This subpackage relies heavily on PyTorch. When PyTorch is not installed we
avoid importing the rule implementations so that importing
``spectramind.symbolic.rules`` does not raise ``ModuleNotFoundError`` at module
import time. Callers that actually need these rules will still fail with a
clear error when accessing the attributes.
"""

from importlib.util import find_spec as _find_spec

_TORCH_AVAILABLE = _find_spec("torch") is not None

if _TORCH_AVAILABLE:  # pragma: no cover - exercised only when torch present
    from .asymmetry_rule import AsymmetryRule
    from .base_rule import RuleOutput, SymbolicRule
    from .composite_rule import CompositeRule
    from .fft_spectral_rule import FFTSpectralRule
    from .molecular_coherence_rule import MolecularCoherenceRule
    from .nonnegativity_rule import NonNegativityRule
    from .photonic_alignment_rule import PhotonicAlignmentRule
    from .registry import (
        RULE_REGISTRY,
        build_rule_from_config,
        get_rule,
        list_rules,
        register_rule,
    )
    from .smoothness_rule import SmoothnessRule

    __all__ = [
        "SymbolicRule",
        "RuleOutput",
        "SmoothnessRule",
        "NonNegativityRule",
        "MolecularCoherenceRule",
        "PhotonicAlignmentRule",
        "AsymmetryRule",
        "FFTSpectralRule",
        "CompositeRule",
        "RULE_REGISTRY",
        "register_rule",
        "get_rule",
        "list_rules",
        "build_rule_from_config",
    ]
else:  # torch missing
    # Provide minimal placeholders so that attribute access produces a helpful
    # error rather than failing at import time. These keep optional dependencies
    # truly optional.
    __all__ = []

    class _TorchRequired:
        def __init__(self, name: str) -> None:  # pragma: no cover - trivial
            self._name = name

        def __call__(self, *args, **kwargs):  # pragma: no cover - trivial
            raise ImportError(f"{self._name} requires PyTorch to be installed")

    AsymmetryRule = CompositeRule = FFTSpectralRule = MolecularCoherenceRule = (
        NonNegativityRule
    ) = PhotonicAlignmentRule = SmoothnessRule = _TorchRequired(
        "spectramind.symbolic.rules"
    )

    def _missing(*args, **kwargs):  # pragma: no cover - trivial
        raise ImportError("spectramind.symbolic.rules requires PyTorch")

    RULE_REGISTRY = {}
    register_rule = get_rule = list_rules = build_rule_from_config = _missing
