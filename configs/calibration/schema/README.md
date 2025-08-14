SpectraMind V50 — Calibration Config Schemas (Hydra-safe, JSON Schema 2020-12)

Purpose:

Canonical JSON Schemas that validate all YAML calibration configs used by the pipeline.

These schemas define structure and constraints for stage configs (ADC, dark, flat,

nonlinearity, cosmic, photometry, trace extraction, normalization, alignment), uncertainty

calibration (temperature scaling, COREL), validation gates, and output products manifests.



Usage:

1) Author your Hydra YAMLs under configs/calibration/*.yaml

2) Run schema/validate.py to validate YAMLs against these JSON Schemas

   python configs/calibration/schema/validate.py --files configs/calibration/*.yaml

3) The schemas are Draft 2020-12 and composed via $ref to common_defs.schema.json



Notes:

• All numeric tolerances are in SI unless the field explicitly states detector units (ADU).

• Times are ISO-8601 or seconds since exposure start; choose one and keep consistent.

• Any stage may be toggled on/off. When off, its parameters are ignored by the runtime.

• Set "io.policy" fields in your runtime configs (outside schema scope) for DVC/lakeFS.

• These schemas are intentionally strict to keep CI honest; relax with care.



Files:

- common_defs.schema.json                 : Shared $defs (enums, tolerances, path, rng)

- frame_corrections.schema.json           : ADC, bias, dark, flat, nonlinearity, ramp

- cosmic_ray.schema.json                  : CR detection/rejection policies

- photometry_extraction.schema.json       : Aperture/annulus, centroiding, jitter feed

- trace_extraction.schema.json            : AIRS trace model and extraction params

- normalization_alignment.schema.json     : Detrend, normalization, temporal alignment

- uncertainty_calibration.schema.json     : Temperature scaling + COREL graph settings

- validation_checks.schema.json           : Calib QA thresholds and gating rules

- products_manifest.schema.json           : Declares expected artifacts/paths

- calibration_pipeline.schema.json        : Top-level object composing all stages

- validate.py                             : CLI to validate YAMLs against schemas



Conventions:

- snake_case keys

- booleans prefer explicit "enabled: true/false"

- arrays use fixed-length where shapes are critical (e.g., center [x,y])

- $anchor provided for selective reuse by tooling



Example (YAML to be validated against calibration_pipeline.schema.json):



version: v50

stages:

  adc:

    enabled: true

    bits: 16

    saturation_adu: 65535

  dark:

    enabled: true

    reference_map: "refs/dark_map_v3.fits"

    scale_mode: "exptime"

  flat:

    enabled: true

    reference_map: "refs/flat_airsv2.fits"

    normalize: "median"

  nonlinearity:

    enabled: true

    model: "poly3"

    coeffs: [0.0, 1.0, 2.3e-6, -7.1e-12]

    valid_range_adu: [0, 60000]

  ramp:

    enabled: false

  cosmic:

    enabled: true

    method: "lacosmic"

    sigma_clip: 5.0

    max_frac_flagged: 0.02

  photometry:

    enabled: true

    method: "aperture"

    aperture_radius_px: 5.0

    annulus_inner_px: 8.0

    annulus_outer_px: 12.0

    centroid:

      mode: "2d-moments"

      refine_iterations: 2

  trace:

    enabled: true

    model: "poly2"

    polynomial_x_to_row: [12.4, 0.0031, -1.1e-6]

    width_px: 7

  normalize_align:

    enabled: true

    normalization: "median"

    detrend:

      enabled: true

      regressors: ["jitter_x", "jitter_y", "centroid_x", "centroid_y"]

      method: "ridge"

      alpha: 0.1

    resample:

      enabled: true

      method: "cubic"

      target_dt_s: 1.0

  uncertainty:

    enabled: true

    temperature_scaling:

      enabled: true

      initial_tau: 1.0

      bounds: [0.25, 4.0]

    corel:

      enabled: true

      graph:

        edge_type: "wavelength+region"

        add_self_loops: true

      training:

        epochs: 30

        lr: 1e-3

        weight_decay: 1e-4

  validate:

    enabled: true

    thresholds:

      max_bad_pixels_frac: 0.005

      max_rms_e2e: 0.02

  products:

    enabled: true

    out_dir: "artifacts/calibrated"

    write_intermediate: true



End README

