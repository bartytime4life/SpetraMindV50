# configs/symbolic/molecules

Symbolic, human-friendly chemical knowledge for pipelines:
- `periodic_table.yaml` — minimal extendable element facts
- `bonds.yaml` — bond orders & canonical ranges
- `functional_groups.yaml` — compact group patterns
- `amino_acids.yaml` — biochemical reference (subset as starter)
- `spectroscopy.yaml` — exoplanet-relevant molecules + bands/lines
- `symbolic_map.yaml` — how symbols resolve to runtime assets
- `_schemas/molecule.schema.json` — JSON Schema used by validation

## Validate locally
Use any YAML->JSON + JSON Schema validator (e.g., `yajsv`, `ajv`). Example:
```bash
yq -o=json '.molecules[]' spectroscopy.yaml | ajv -s _schemas/molecule.schema.json -d /dev/stdin
```

Conventions
  • IDs: kebab_or_snake, immutable and code-facing.
  • Fields are deliberately conservative; add domain fields as needed.
  • Keep provenance in spectral.references.
