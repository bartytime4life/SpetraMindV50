# Configuration Helpers

Utility helpers for loading, validating and manipulating SpectraMind
configuration files.  These modules expose a small API that is also
integrated into the root ``spectramind`` CLI.

Functions provided include:

- ``load_config`` / ``save_config`` for basic YAML IO.
- ``cli_override_parser`` and ``apply_overrides`` for Hydra-style CLI overrides.
- ``validate_config`` for JSON schema validation.
- ``inject_symbolic_constraints`` for default symbolic weights.
- ``capture_environment`` / ``log_environment`` for environment metadata.
