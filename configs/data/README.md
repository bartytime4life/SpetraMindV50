# SpectraMind V50 — `configs/data` Pack

This directory defines the Hydra‑safe, environment‑portable data configuration used across the CLI:

- `base.yaml` — canonical defaults for paths, IO, loader, splits, calibration, features, instruments, diagnostics.  
- `local.yaml` — workstation overrides (paths under `_local/`, more workers).  
- `kaggle.yaml` — Kaggle runtime guardrails (fewer workers, working paths).  
- `splits.yaml` — GroupKFold materialization controls (file output, quick‑dev slice).  
- `calibration.yaml` — tuning knobs for each calibration kill‑chain stage.  
- `fgs1.yaml` — FGS1 instrument‑specific feature & extraction settings.  
- `airs.yaml` — AIRS instrument‑specific spectral feature settings.  
- `features.yaml` — persisted NPZ key schemas and toggles for fgs1_white / airs_bins.  
- `paths.yaml` — skeleton of directories the CLI can pre‑create for you.  
- `schema.yaml` — runtime schemas for validators and integrity checks.  
- `dev_fast.yaml` — developer quick‑iterate configuration.

### Composition examples

Use via `config_v50.yaml` defaults or CLI overrides:

```bash
# Use base + local overrides
python -m spectramind selftest +data=@configs/data/base.yaml +data=@configs/data/local.yaml

# Kaggle runtime with dev-fast tweaks
python -m spectramind diagnose +data=@configs/data/base.yaml +data=@configs/data/kaggle.yaml +data=@configs/data/dev_fast.yaml

# Explicit split materialization
python -m spectramind prepare-splits +data=@configs/data/base.yaml +data=@configs/data/splits.yaml
``` 
