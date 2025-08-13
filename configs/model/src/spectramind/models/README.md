# SpectraMind V50 — `configs/model/src/spectramind/models`

Hydra configs that bind model components to classes in `src/spectramind/models`.

## Components
- `fgs1_mamba.yaml` → `spectramind.models.fgs1_mamba.FGS1MambaEncoder`
- `airs_gnn.yaml` → `spectramind.models.airs_gnn.AIRSGNNEncoder`
- `multi_scale_decoder.yaml` → `spectramind.models.multi_scale_decoder.MultiScaleMuDecoder`
- `flow_uncertainty_head.yaml` → `spectramind.models.flow_uncertainty_head.FlowUncertaintyHead`
- `fusion.yaml` → `spectramind.models.fusion.FusionBlock`
- `registry.yaml` → name→class mapping
- `pipeline_active.yaml` → choose active components
- `__init__.yaml` → group defaults

See the SpectraMind V50 architecture docs for scientific & engineering rationale.
