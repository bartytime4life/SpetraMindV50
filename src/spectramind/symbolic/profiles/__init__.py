"""Package initializer for the SpectraMind V50 neuro-symbolic profiles subsystem.

Exposes core loader/registry/diagnostics surface and version metadata helpers.
"""
from __future__ import annotations

from .utils import (
    find_repo_root,
    safe_load_yaml,
    safe_dump_yaml,
    deep_merge,
    sha256_of_text,
    load_all_yaml_files_in_dir,
)
from .logging_utils import (
    setup_logging,
    write_jsonl_event,
    append_v50_debug_log,
    git_snapshot,
    env_snapshot,
)
from .profile_loader import (
    SymbolicRuleRef,
    SymbolicProfile,
    ProfileLoadResult,
    load_profiles_with_overrides,
    validate_profile_dict,
)
from .profile_registry import (
    ProfileRegistry,
    get_registry,
)

__all__ = [
    "find_repo_root",
    "safe_load_yaml",
    "safe_dump_yaml",
    "deep_merge",
    "sha256_of_text",
    "load_all_yaml_files_in_dir",
    "setup_logging",
    "write_jsonl_event",
    "append_v50_debug_log",
    "git_snapshot",
    "env_snapshot",
    "SymbolicRuleRef",
    "SymbolicProfile",
    "ProfileLoadResult",
    "load_profiles_with_overrides",
    "validate_profile_dict",
    "ProfileRegistry",
    "get_registry",
]
