# SpectraMind V50 - Config Tests

This directory contains **unit and integration tests** for validating all Hydra configuration files in `src/spectramind/config`.

## Features
- ✅ Hydra YAML syntax validation  
- ✅ Reproducibility hash testing  
- ✅ CLI integration checks  
- ✅ Schema compliance for core sections (`model`, `train`, `data`, `symbolic`, `diagnostics`)  
- ✅ Symbolic config validation (loss weights, profiles)  

## Usage
Run all tests:
```bash
pytest src/spectramind/config/tests -v
```

Run via SpectraMind CLI:

```bash
spectramind test --mode deep
```

## Outputs

* `config_hashes.txt` (per-test hash log)
* Pass/fail summaries in `pytest` output

These tests ensure SpectraMind V50 configs are **valid, reproducible, symbolic-aware, and CLI-integrated**.
