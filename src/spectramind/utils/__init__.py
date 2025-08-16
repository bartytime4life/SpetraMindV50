"""
SpectraMind V50 Utilities Package.

This package centralizes mission-grade utilities used across the SpectraMind V50
pipeline (training, inference, diagnostics, CLI, CI). Every module emphasizes
reproducibility, robust logging, symbolic-aware workflows, and clean interfaces.

Modules re-exported here (import-once, everywhere):

* logging_utils: Console + rotating-file logger with JSONL event stream support.
* jsonl: Safe, atomic JSONL writing/reading utilities.
* event_stream: Structured JSONL event emitter for CLI, training, diagnostics.
* paths: Repository-aware path helpers (root detection, artifacts, logs).
* hashing: Stable hashing for configs/artifacts (blake2, canonical JSON).
* reproducibility: Git/ENV capture, seed setting, run snapshotting.
* hydra_utils: OmegaConf helpers, resolve/save configs, flatten/unflatten.
* io_utils: Atomic save/load for JSON/YAML/CSV/NPZ/Torch.
* timers: High-precision timing utilities (Timer, Stopwatch).
* parallel: Simple multiprocessing helpers with robust error handling.
* distributed: Torch distributed helpers (init, barrier, reduce dicts).
* metrics: Scientific metrics (GLL, RMSE, MAE, calibration checks).
* numpy_torch: Safe NumPy↔Torch conversions and device helpers.
* profiles: Lightweight CPU/Torch profilers with file export.
* cli_integration: Unified CLI invocation logging to v50_debug_log.md.
* mlflow_utils: Optional MLflow tracker with safe fallbacks.
* report: Lightweight HTML helpers for diagnostics dashboards.
* retry: Exponential backoff retry decorator with jitter.
* cache: Disk-cache decorator keyed by stable hashes.
* env_info: Hardware/software env discovery (CUDA, torch, drivers).
* validators: Light schema/shape/type validators for configs/data.
* exceptions: Project-wide exception types.
* lakefs_dvc: Optional DVC/lakeFS presence checks and helper affordances.
* checks: Self-test for utils package integrity.

All utilities avoid heavy optional dependencies by design and degrade gracefully.
"""

from .logging_utils import get_logger, configure_logging, add_file_handler, add_jsonl_handler
from .jsonl import jsonl_append, jsonl_iter, atomic_write_text, atomic_write_bytes
from .event_stream import EventStream
from .paths import (
    find_repo_root,
    get_default_paths,
    ensure_dir,
    repo_relpath,
    V50_DEFAULTS,
)
from .hashing import stable_hash, hash_config, hash_file, hash_bytes
from .reproducibility import (
    set_all_seeds,
    snapshot_run_info,
    collect_git_info,
    collect_env_info,
    summarize_packages,
)
from .hydra_utils import (
    is_omegaconf,
    to_resolved_dict,
    save_resolved_config,
    flatten_dict,
    unflatten_dict,
)
from .io_utils import (
    load_json,
    save_json,
    load_yaml,
    save_yaml,
    load_csv,
    save_csv,
    save_npz,
    load_npz,
    save_torch,
    load_torch,
    path_exists,
    safe_mkdirs,
)
from .timers import Timer, Stopwatch, time_block
from .parallel import run_parallel, cpu_count_safe, chunked
from .distributed import (
    ddp_available,
    init_distributed,
    ddp_rank,
    ddp_world_size,
    ddp_is_master,
    ddp_barrier,
    ddp_reduce_dict,
)
from .metrics import (
    gll_loss,
    rmse,
    mae,
    calibration_error,
    binwise_gll,
    zscore_histogram,
)
from .numpy_torch import (
    to_numpy,
    to_tensor,
    move_to_device,
    autocast_if_available,
    amp_dtype_from_str,
)
from .profiles import cpu_profile, torch_profile
from .cli_integration import log_cli_invocation, load_run_hash_summary, append_debug_log
from .mlflow_utils import MlflowClientSafe, mlflow_safe_log_params, mlflow_safe_log_metrics
from .report import html_img_base64, html_section, save_html_fragment
from .retry import retry
from .cache import disk_cache
from .env_info import detect_env
from .validators import require_keys, ensure_type, ensure_shape, validate_numeric_range
from .exceptions import (
    SpectraMindError,
    ConfigValidationError,
    ReproducibilityError,
    DistributedError,
    IOValidationError,
)
from .lakefs_dvc import have_dvc, have_lakefs, dvc_status, lakefs_status
from .checks import utils_selftest

__all__ = [
    # logging & events
    "get_logger",
    "configure_logging",
    "add_file_handler",
    "add_jsonl_handler",
    "EventStream",
    # filesystem/paths
    "find_repo_root",
    "get_default_paths",
    "ensure_dir",
    "repo_relpath",
    "V50_DEFAULTS",
    # hashing
    "stable_hash",
    "hash_config",
    "hash_file",
    "hash_bytes",
    # reproducibility
    "set_all_seeds",
    "snapshot_run_info",
    "collect_git_info",
    "collect_env_info",
    "summarize_packages",
    # hydra
    "is_omegaconf",
    "to_resolved_dict",
    "save_resolved_config",
    "flatten_dict",
    "unflatten_dict",
    # io utils
    "load_json",
    "save_json",
    "load_yaml",
    "save_yaml",
    "load_csv",
    "save_csv",
    "save_npz",
    "load_npz",
    "save_torch",
    "load_torch",
    "path_exists",
    "safe_mkdirs",
    # timers
    "Timer",
    "Stopwatch",
    "time_block",
    # parallel
    "run_parallel",
    "cpu_count_safe",
    "chunked",
    # distributed
    "ddp_available",
    "init_distributed",
    "ddp_rank",
    "ddp_world_size",
    "ddp_is_master",
    "ddp_barrier",
    "ddp_reduce_dict",
    # metrics
    "gll_loss",
    "rmse",
    "mae",
    "calibration_error",
    "binwise_gll",
    "zscore_histogram",
    # numpy/torch
    "to_numpy",
    "to_tensor",
    "move_to_device",
    "autocast_if_available",
    "amp_dtype_from_str",
    # profiling
    "cpu_profile",
    "torch_profile",
    # CLI integration
    "log_cli_invocation",
    "load_run_hash_summary",
    "append_debug_log",
    # mlflow
    "MlflowClientSafe",
    "mlflow_safe_log_params",
    "mlflow_safe_log_metrics",
    # report
    "html_img_base64",
    "html_section",
    "save_html_fragment",
    # retry/cache/env
    "retry",
    "disk_cache",
    "detect_env",
    # validators/exceptions
    "require_keys",
    "ensure_type",
    "ensure_shape",
    "validate_numeric_range",
    "SpectraMindError",
    "ConfigValidationError",
    "ReproducibilityError",
    "DistributedError",
    "IOValidationError",
    # dvc/lakefs
    "have_dvc",
    "have_lakefs",
    "dvc_status",
    "lakefs_status",
    # self-test
    "utils_selftest",
]
