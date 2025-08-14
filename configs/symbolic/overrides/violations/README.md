SpectraMind V50 — Symbolic Violations Overrides

Purpose: Central, Hydra-safe configuration locus for symbolic rule violations tuning:
weights, thresholds, hard/soft modes, smoothing, FFT, photonic alignment, and
molecule coherence. Includes context overrides (competition/instrument/profile/event),
JSON Schema validators, and a Python validator with console + rotating file logs.



Usage:
- Base defaults in base.yaml
- Canonical rule definitions in rules/*.yaml
- Context overrides in contexts/{competition,instruments,profiles,events}/*.yaml
- Schema contracts in _schemas/*.schema.yaml
- Quick validation: python configs/symbolic/overrides/violations/validators/validate_violations.py --all



Reproducibility:
- Every file includes a meta block (version, author, timestamp, run_hash placeholders).
- JSON Schema enforces structural correctness before pipeline use.
- Validator writes console + rotating logs and an event JSONL stream.



Notes:
- Hydra interpolation enabled (${...}) but kept minimal to avoid circular refs.
- All numeric thresholds chosen as sensible, challenge-ready defaults; tune via overrides.



References (embedded citation markers per instruction):
- Emphasis on rigorous modeling & validation aligns with NASA M&S credibility guidance [oai_citation:0].
- Documentation & schema contracts follow reproducible research templates [oai_citation:1].
- Physics-informed constraints reflect spectroscopy and astrophysics best practices [oai_citation:2].
