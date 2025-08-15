# SPDX-License-Identifier: Apache-2.0

"""Package initialization for SpectraMind V50 symbolic rules.

Exposes the public API: base classes, concrete rules, and registry helpers.
"""

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
