# configs/profiles

This directory defines **symbolic and diagnostic profiles** for SpectraMind V50 using Hydra’s configuration grouping.

## Purpose
Profiles provide domain-specific configurations for symbolic rules, diagnostics, and reproducibility. They enable modular switching between astrophysics, chemistry, biology, cross-disciplinary, and GUI-focused modes.

## Available Profiles
- **default.yaml** – Baseline symbolic and diagnostic settings.
- **astrophysics.yaml** – Physics-informed rules (FFT smoothness, photonic alignment, cosmology overlays).
- **chemistry.yaml** – Quantum chemistry and molecular line constraints.
- **biology.yaml** – Symbolic entropy and pathway consistency checks for biological systems.
- **cross_disciplinary.yaml** – Fusion across AI, physics, chemistry, biology.
- **gui.yaml** – Diagnostic/visualization overlays for dashboards.

## Usage
To activate a profile in Hydra CLI:
```bash
python train_v50.py profiles=astrophysics
python diagnose_v50.py profiles=gui
```

## References

Profiles are informed by:

* Documentation-first protocol
* Patterns, algorithms, fractals reference
* NASA-grade scientific modeling
* Physics & astrophysics modeling
* Chemistry & biology foundations
* Engineering guide to GUI systems
* Foundational templates & glossary
* Domain module examples
* Ariel mission science context
