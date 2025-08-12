/configs/model/README.md

/configs/model

Hydra YAML configs for the SpectraMind V50 modeling stack (FGS1 Mamba SSM, AIRS physics‑informed GNN, fusion, μ/σ decoders, symbolic losses). Compose these from configs/config_v50.yaml or via CLI overrides:
	•	python -m spectramind train model=base
	•	python -m spectramind train model=small
	•	python -m spectramind train model=medium
	•	python -m spectramind train model=large

Files:
	•	base.yaml — canonical model graph and options (encoders, fusion, decoders, losses).
	•	small.yaml|medium.yaml|large.yaml — presets scaling depth/width and regularization.
	•	symbolic_profiles.yaml — molecule windows, seam index, loss weights defaults.
	•	fusion_presets.yaml — common fusion choices (concat, cross‑attend, gated).
	•	decoder_presets.yaml — μ multi‑scale head and σ head presets (flow/quantile).
	•	explainability.yaml — SHAP/attention export and hooks.
Use with Hydra: model@model.encoder.fgs1=base style overrides are supported if you split files later; here we keep one cohesive model/base.yaml with nested blocks for simplicity.

/configs/model/base.yaml

Base model configuration for SpectraMind V50

Encoders: FGS1 Mamba SSM (temporal), AIRS GNN (spectral graph with physics edges)

Decoders: multi‑scale μ head, flow σ head; optional quantile head

Losses: Gaussian Log‑Likelihood + Symbolic pack (smoothness, non‑negativity, molecule coherence, seam, ratios, quantile monotonicity)

Explainability hooks: attention/SHAP exports

TorchScript/JIT options for production inference

name: spectramind_v50
seed: ${oc.env:SM_SEED,1337}
precision: ${oc.env:SM_DTYPE,“fp16”}         # [fp32, fp16, bf16]
amp: true
deterministic: false
cudnn_benchmark: true
torchscript:
enable: false
strict: false

encoders:
fgs1_mamba:
enabled: true
in_features: 6                     # [flux, time_norm, centroid_x, centroid_y, jitter_x, jitter_y]
d_model: 256
d_state: 64
d_conv: 4
expand: 2
n_layers: 8
bidirectional: true
dropout: 0.1
layer_norm: “rms”                  # [layernorm, rms]
proj_out: 256
export_step_latents: false         # for UMAP/t‑SNE diagnostics
airs_gnn:
enabled: true
node_features: 8                   # per‑bin features or pooled time features
hidden_dim: 256
out_dim: 256
n_layers: 4
backend: “gat”                     # [gat, rgcn, nnconv, mpnn]
heads: 4
residual: true
dropout: 0.1
edge_features:
use: true
dims: 4                          # [Δλ, mol_tag_i, mol_tag_j, seam_flag]
embed:
mol_tags: true
seam_flag: true
positional_encoding:
use: true
kind: “fourier”                # [sinusoidal, fourier]
dim: 16
graph:
n_bins: 283
wavelength_csv: ${data.wavelengths}
seam_index: ${symbolic.seam_index}
edges:
wavelength_adjacency: true
molecular_links: true
detector_seams: true
checks:
validate_edge_attr_sizes: true
allow_self_loops: false

fusion:
kind: “concat”                       # [concat, cross_attend, gated]
dim: 512
hidden: 256
dropout: 0.1
cross_attend:
n_heads: 4
n_layers: 1

decoders:
mu_head:
type: “multiscale”                 # [single, multiscale]
hidden: 512
dropout: 0.1
scales:
- coarse
- mid
- fine
aux_losses: true
sigma_head:
type: “flow”                       # [flow, quantile]
hidden: 256
dropout: 0.1
activation: “softplus”
sigma_min: 1.0e-4
temperature_scaling_compatible: true
quantile:
q_list: [0.1, 0.5, 0.9]
monotonic_penalty: 1.0

losses:
gll_weight: 1.0
symbolic:
enable: true
smoothness:
lambda: 0.1
window: 2
mask_strong_lines: true
nonnegativity:
lambda: 0.05
power: 2
molecular_coherence:
lambda: 0.1
use_templates: true
normalize_within_window: true
seam_continuity:
lambda: 0.05
ratio_envelopes:
lambda: 0.05
pairs: [[“CH4”,“H2O”],[“CO2”,“H2O”]]
quantile_monotonicity:
lambda: 0.05
regularization:
weight_decay: 0.01
dropout: 0.1

explainability:
export_attention: true
export_node_importance: true
shap:
enable: true
topk_bins: 24
overlay_symbolic: true

calibration:
enable: true
temperature:
enable: true
corel:
enable: true
coverage_target: 0.9
loss: “mse”                        # coverage penalty + mse on z‑scores
gnn_backend: “gat”
hidden_dim: 128
n_layers: 2
dropout: 0.1
edge_features: true

jit_export:
enable: false
output_dir: “outputs/jit”

/configs/model/small.yaml

Lightweight preset for fast iteration / CI smoke runs

defaults:
	•	base

encoders:
fgs1_mamba:
d_model: 128
d_state: 32
n_layers: 4
airs_gnn:
hidden_dim: 128
out_dim: 128
n_layers: 2
heads: 2
fusion:
dim: 256
hidden: 128
decoders:
mu_head:
hidden: 256
sigma_head:
hidden: 128

losses:
symbolic:
smoothness:
lambda: 0.05
nonnegativity:
lambda: 0.02

/calibration:
enable: true

/configs/model/medium.yaml

Mid‑size preset for development on a single GPU

defaults:
	•	base

encoders:
fgs1_mamba:
d_model: 192
d_state: 48
n_layers: 6
airs_gnn:
hidden_dim: 192
out_dim: 192
n_layers: 3
heads: 4
fusion:
dim: 384
hidden: 192
decoders:
mu_head:
hidden: 384
sigma_head:
hidden: 192

/configs/model/large.yaml

Larger preset for leaderboard training (ensure GPU VRAM headroom)

defaults:
	•	base

encoders:
fgs1_mamba:
d_model: 320
d_state: 96
n_layers: 10
dropout: 0.1
airs_gnn:
hidden_dim: 320
out_dim: 320
n_layers: 5
heads: 6
dropout: 0.1
fusion:
dim: 640
hidden: 320
dropout: 0.15
decoders:
mu_head:
hidden: 640
sigma_head:
hidden: 320
losses:
regularization:
weight_decay: 0.02

/configs/model/symbolic_profiles.yaml

Symbolic windows and detector seam index used by symbolic losses and graph edges

Windows are index ranges (inclusive) in wavelength‑sorted bin order

seam_index: 178
molecule_windows:
H2O:
ranges: [[40, 85], [210, 240]]
CO2:
ranges: [[95, 130], [245, 265]]
CH4:
ranges: [[135, 175], [270, 282]]
ratio_bounds:
CH4_H2O: [0.2, 3.0]
CO2_H2O: [0.1, 2.5]

/configs/model/fusion_presets.yaml

Common fusion strategies usable via override of model.fusion

concat:
kind: “concat”
dim: 512
hidden: 256
dropout: 0.1
cross_attend:
kind: “cross_attend”
dim: 512
hidden: 256
dropout: 0.1
cross_attend:
n_heads: 4
n_layers: 1
gated:
kind: “gated”
dim: 512
hidden: 256
dropout: 0.1

/configs/model/decoder_presets.yaml

μ/σ decoder presets

multiscale_mu_default:
type: “multiscale”
hidden: 512
dropout: 0.1
scales: [coarse, mid, fine]
aux_losses: true
sigma_flow_default:
type: “flow”
hidden: 256
dropout: 0.1
activation: “softplus”
sigma_min: 1.0e-4

/configs/model/explainability.yaml

Controls for explainability hooks during forward/eval

export_dir: “outputs/explainability”
capture_attention: true
capture_node_importance: true
capture_latents:
fgs1_steps: false
airs_nodes: true
shap:
enable: true
topk_bins: 24
save_png: true
save_json: true

/configs/train/README.md

/configs/train

Hydra YAML configs for optimization, schedules, curriculum, data loaders, logging, checkpoints, calibration, and runtime discipline. Compose with CLI:
	•	python -m spectramind train train=base
	•	python -m spectramind train train=local
	•	python -m spectramind train train=kaggle
	•	python -m spectramind train train=fast_smoke

Files:
	•	base.yaml — canonical training regime with curriculum (MAE → contrastive → supervised GLL+symbolic), cosine LR, AMP, grad‑accum, checkpoints, MLflow hooks.
	•	local.yaml — overrides for workstation (bigger workers, paths).
	•	kaggle.yaml — overrides for Kaggle runtime (≤9h guardrail; tuned loader & batch sizes).
	•	fast_smoke.yaml — ultra‑fast CI sanity config.
	•	ablation.yaml — defaults for spectramind ablate runs (sweeps, seeds, logging).
	•	calibration.yaml — post‑hoc temperature scaling + COREL options and schedules.

/configs/train/base.yaml

Canonical training configuration for SpectraMind V50

Three‑phase curriculum + calibration: MAE → (optional) contrastive → supervised (GLL + symbolic) → temp/corel

Logs: console + rotating file + JSONL events + optional MLflow

Checkpoints: best‑GLL, last; resume supported

device: “cuda”
seed: ${model.seed}
epochs:
mae: 20
contrastive: 10
supervised: 40
early_stopping:
enable: true
patience: 8
monitor: “val/gll”
mode: “min”
resume:
enable: true
ckpt_path: null

optimizer:
name: “adamw”
lr: 2.0e-4
betas: [0.9, 0.95]
eps: 1.0e-8
weight_decay: ${model.losses.regularization.weight_decay}
scheduler:
name: “cosine”
warmup_steps: 2000
min_lr: 1.0e-6
t_max: null

grad:
clip_norm: 1.0
accumulate_steps: 2
amp: ${model.amp}
precision: ${model.precision}

batch:
train: 32
eval: 64
workers: 8
pin_memory: true
prefetch_factor: 2
persistent_workers: true
drop_last: true

data_augmentation:
fgs1:
jitter_augment: true
jitter_std: 0.25           # pixels; used to generate synthetic centroid/jitter coupling
time_dropout_prob: 0.0
airs:
spectral_smooth_noise: 0.0

curriculum:
enable: true
phases: [“mae”, “contrastive”, “supervised”]
mae:
mask_kind: “snr_aware”     # [random, block, fixed, snr_aware, molecule_masked]
mask_ratio: 0.35
snr_floor_db: 10.0
decoder_loss_weight: 0.1
optimizer:
lr: 3.0e-4
contrastive:
enable: true
temperature: 0.07
projector_dim: 128
views: 2
supervised:
symbolic_weight_scale: 1.0

supervised_objective:
gll_weight: ${model.losses.gll_weight}
symbolic: ${model.losses.symbolic}

logging:
level: “INFO”
rich_console: true
file_log:
enable: true
path: “v50_debug_log.md”
rotate_mb: 8
backups: 5
jsonl:
enable: true
path: “events.jsonl”
mlflow:
enable: true
tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,””}
experiment: “spectramind_v50”
tags:
repo_sha: ${oc.env:GIT_COMMIT,“unknown”}
config_hash: ${oc.env:SM_CONFIG_HASH,“unknown”}

checkpoints:
dir: “outputs/checkpoints”
save_best_on: “val/gll”
save_last: true
interval_epochs: 1
keep: 3

validation:
interval_steps: 1000
metrics:
- “gll”
- “rmse”
- “mae”
- “coverage_90”            # after calibration stage (eval‑only)
save_preds: false

submission:
pack_on_finish: false
out_csv: “outputs/submission.csv”

calibration:
run_after_supervised: true
temperature:
enable: true
max_iters: 500
lr: 1.0e-2
tol: 1.0e-6
corel:
enable: true
coverage_target: 0.9
holdout_fraction: 0.2
epochs: 20
batch: 256
optimizer:
name: “adamw”
lr: 1.0e-3
weight_decay: 0.0

runtime_guardrails:
target_total_hours: 9.0
fast_kaggle_mode: false
assert_on_overrun: true
telemetry_interval_s: 30

reproducibility:
save_env_snapshot: true
capture:
git_sha: ${oc.env:GIT_COMMIT,“unknown”}
cuda_version: ${oc.env:CUDA_VERSION,“unknown”}
driver_version: ${oc.env:NVIDIA_DRIVER_VERSION,“unknown”}
python: ${oc.env:PYTHON_VERSION,“unknown”}

CLI wiring hints (used by spectramind.py)

cli:
register_commands: true
expose_flags:
- “epochs.supervised”
- “optimizer.lr”
- “batch.train”
- “runtime_guardrails.fast_kaggle_mode”

/configs/train/local.yaml

Local workstation overrides (bigger workers, stable batch sizes)

defaults:
	•	base

batch:
train: 32
eval: 64
workers: 12
prefetch_factor: 3
runtime_guardrails:
fast_kaggle_mode: false

logging:
level: “INFO”
mlflow:
enable: ${oc.env:MLFLOW_ON,false}

/configs/train/kaggle.yaml

Kaggle runtime discipline — tuned for ≤9h across ~1,100 planets

defaults:
	•	base

batch:
train: 24
eval: 48
workers: 4
prefetch_factor: 2
runtime_guardrails:
fast_kaggle_mode: true
assert_on_overrun: true
scheduler:
warmup_steps: 1500
optimizer:
lr: 1.6e-4
checkpoints:
keep: 2
logging:
level: “WARNING”
mlflow:
enable: false

/calibration:
run_after_supervised: true

/configs/train/fast_smoke.yaml

Ultra‑fast CI smoke test (minutes)

defaults:
	•	base

epochs:
mae: 1
contrastive: 0
supervised: 2
batch:
train: 8
eval: 16
workers: 2
curriculum:
enable: true
phases: [“mae”, “supervised”]
logging:
level: “WARNING”
checkpoints:
keep: 1
validation:
interval_steps: 2000

/configs/train/ablation.yaml

Defaults for ablation/sweep runs (used by spectramind ablate / auto_ablate_v50.py)

defaults:
	•	base

ablate:
enable: true
max_runs: 32
top_n: 5
search_space:
optimizer.lr: [1.0e-4, 2.0e-4, 3.0e-4]
model.encoders.fgs1_mamba.n_layers: [4, 6, 8]
model.encoders.airs_gnn.n_layers: [2, 3, 4]
model.fusion.kind: [“concat”, “gated”]
model.decoders.sigma_head.type: [“flow”]
model.losses.symbolic.smoothness.lambda: [0.05, 0.1, 0.2]
export:
md: “outputs/ablate/leaderboard.md”
html: “outputs/ablate/leaderboard.html”
zip_top_n: true
zip_dir: “outputs/ablate/topn_bundles”
logging:
level: “INFO”
mlflow:
enable: true
experiment: “spectramind_v50_ablate”

/configs/train/calibration.yaml

Standalone calibration runs (post‑hoc only) via spectramind calibrate-*

temperature:
enable: true
max_iters: 800
lr: 1.0e-2
tol: 1.0e-6
corel:
enable: true
coverage_target: 0.9
holdout_fraction: 0.2
epochs: 30
batch: 256
optimizer:
name: “adamw”
lr: 1.0e-3
weight_decay: 0.0