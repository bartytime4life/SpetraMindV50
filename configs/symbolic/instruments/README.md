# instruments

Concrete instrument profiles. Each file should validate against `instrument.schema.yaml`.

Conventions
- Wavelengths in microns (μm), resolutions as λ/Δλ at reference λ.
- Noise terms are RMS per exposure unless specified.
- `calibration` fields are paths (relative to repo) or identifiers resolvable by your data loader.
