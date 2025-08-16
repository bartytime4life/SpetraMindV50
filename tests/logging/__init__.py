"""
tests.logging
=============
Test suite for SpectraMind V50 logging subsystem.

Covers:
- Console handler
- Rotating file logs
- JSONL event stream
- Hydra job_logging integration
- Reproducibility log capture (ENV, git hash, configs)

All tests are written in pytest style and are intended
for automated CI/CD workflows.
"""
