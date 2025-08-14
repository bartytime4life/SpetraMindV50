# Symbolic Configuration (SpectraMind V50)

This subtree contains symbolic configuration for physics-/logic-aware constraints used across the SpectraMind V50 pipeline.

- `rules/` (if present): Canonical symbolic rule packs (base truths; usually do not mutate frequently)
- `molecules/`: Molecule-centric metadata/configs and region mappings
- `overrides/`: Contextual override layers (competition, events, molecules) that are composed over the canonical packs
- `profiles/` (if present): Profile definitions selecting subsets/weights of rules for different operating modes

> Policy: **immutable base + composable overrides**. Always update overrides first for scenario-specific needs.
