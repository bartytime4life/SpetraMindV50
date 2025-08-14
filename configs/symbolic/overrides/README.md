# Symbolic Overrides

Override layers that modify or specialize canonical symbolic rules.

## Structure

- `_schemas/`: JSON Schemas for validating override files
- `events/`: Event-driven rule switching (mission phases, instrument states, pipeline triggers)
- `competition/`: Leaderboard-safe hardening (time/compute caps, determinism, risk guards)
- `molecules/`: Molecule-region attention and physics-aware emphasis (e.g., H2O / CO2 bands)

## Validation

All YAML/JSON overrides are designed to be validated with the schemas in `_schemas/`. You can run
your preferred JSON Schema validator (e.g., `ajv`, `python -m jsonschema`) during CI to enforce structure.
