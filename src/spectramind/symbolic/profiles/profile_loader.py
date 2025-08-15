"""Loader for Symbolic Profiles:

- Validates profile dicts against a lightweight schema.
- Merges defaults with overrides, supporting extends: semantics and deep merge.
- Returns dataclass objects + run hashes for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logging_utils import setup_logging
from .utils import (
    deep_merge,
    find_repo_root,
    load_all_yaml_files_in_dir,
    safe_dump_yaml,
    safe_load_yaml,
    sha256_of_text,
)


@dataclass
class SymbolicRuleRef:
    id: str
    weight: float = 1.0
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolicProfile:
    id: str
    name: str
    description: str
    rules: List[SymbolicRuleRef]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileLoadResult:
    profiles: Dict[str, SymbolicProfile]             # id -> profile
    hash_by_id: Dict[str, str]                       # id -> sha256 over normalized YAML
    combined_hash: str                               # hash over all concatenated normalized profiles
    source_maps: Dict[str, List[Path]]               # id -> list of contributing file paths (default + overrides)


def _validate_str(node: Any, path: str) -> None:
    if not isinstance(node, str):
        raise ValueError(f"Schema violation at {path}: expected str, got {type(node).__name__}")


def _validate_number(node: Any, path: str) -> None:
    if not isinstance(node, (int, float)):
        raise ValueError(f"Schema violation at {path}: expected number, got {type(node).__name__}")


def _validate_int(node: Any, path: str) -> None:
    if not isinstance(node, int):
        raise ValueError(f"Schema violation at {path}: expected int, got {type(node).__name__}")


def _validate_dict(node: Any, path: str) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"Schema violation at {path}: expected dict, got {type(node).__name__}")


def _validate_list(node: Any, path: str, min_len: int = 0) -> None:
    if not isinstance(node, list):
        raise ValueError(f"Schema violation at {path}: expected list, got {type(node).__name__}")
    if len(node) < min_len:
        raise ValueError(
            f"Schema violation at {path}: expected list with length >= {min_len}, got {len(node)}"
        )


def validate_profile_dict(doc: Dict[str, Any]) -> None:
    """Validate a loaded YAML document against the schema definitions."""
    _validate_dict(doc, "doc")
    if "profiles" not in doc:
        raise ValueError("Schema violation: top-level 'profiles' missing")
    profiles = doc["profiles"]
    _validate_list(profiles, "profiles", min_len=1)
    for i, p in enumerate(profiles):
        p_path = f"profiles[{i}]"
        _validate_dict(p, p_path)
        for req in ("id", "name", "description", "rules"):
            if req not in p:
                raise ValueError(f"Schema violation at {p_path}: missing required key '{req}'")
        _validate_str(p["id"], f"{p_path}.id")
        _validate_str(p["name"], f"{p_path}.name")
        _validate_str(p["description"], f"{p_path}.description")
        if "tags" in p:
            _validate_list(p["tags"], f"{p_path}.tags")
            for j, t in enumerate(p["tags"]):
                _validate_str(t, f"{p_path}.tags[{j}]")
        if "metadata" in p:
            _validate_dict(p["metadata"], f"{p_path}.metadata")
        rules = p["rules"]
        _validate_list(rules, f"{p_path}.rules", min_len=1)
        for j, r in enumerate(rules):
            r_path = f"{p_path}.rules[{j}]"
            _validate_dict(r, r_path)
            if "id" not in r:
                raise ValueError(f"Schema violation at {r_path}: missing required key 'id'")
            _validate_str(r["id"], f"{r_path}.id")
            if "weight" in r:
                _validate_number(r["weight"], f"{r_path}.weight")
            if "priority" in r:
                _validate_int(r["priority"], f"{r_path}.priority")
            if "conditions" in r:
                _validate_dict(r["conditions"], f"{r_path}.conditions")


def _normalize_profile_for_hash(p: Dict[str, Any]) -> str:
    """Produce a deterministic YAML string for hashing (sorted keys where possible)."""
    p2 = dict(p)
    rules = p2.get("rules", [])
    if isinstance(rules, list):
        rules_sorted = sorted(
            rules,
            key=lambda r: (
                str(r.get("id", "")),
                int(r.get("priority", 0)),
                float(r.get("weight", 1.0)),
            ),
        )
        p2["rules"] = rules_sorted
    return safe_dump_yaml(p2)


def _dict_to_profile(p: Dict[str, Any]) -> SymbolicProfile:
    rules = [
        SymbolicRuleRef(
            id=str(r["id"]),
            weight=float(r.get("weight", 1.0)),
            priority=int(r.get("priority", 0)),
            conditions=dict(r.get("conditions", {})),
        )
        for r in p["rules"]
    ]
    return SymbolicProfile(
        id=str(p["id"]),
        name=str(p["name"]),
        description=str(p["description"]),
        rules=rules,
        tags=list(p.get("tags", [])),
        metadata=dict(p.get("metadata", {})),
    )


def _apply_extends_and_merge(
    base_profiles_by_id: Dict[str, Dict[str, Any]],
    override_doc: Dict[str, Any],
    source: Path,
    source_map: Dict[str, List[Path]],
) -> Dict[str, Dict[str, Any]]:
    """Apply an override document to base_profiles_by_id with extends+merge semantics."""
    result = dict(base_profiles_by_id)
    if "profiles" not in override_doc:
        return result
    for p in override_doc["profiles"]:
        pid = p.get("id")
        if not pid:
            continue
        source_map.setdefault(pid, [])
        if "extends" in p:
            base_id = p["extends"]
            base = result.get(base_id)
            if base is None:
                merged = p.copy()
                merged.pop("extends", None)
            else:
                add = p.copy()
                add.pop("extends", None)
                merged = deep_merge(base, add)
        else:
            merged = p
        result[pid] = merged
        source_map[pid].append(source)
    return result


def load_profiles_with_overrides(
    repo_root: Optional[Path] = None,
    defaults_rel: str = "src/spectramind/symbolic/profiles/default_profiles.yaml",
    schema_rel: str = "src/spectramind/symbolic/profiles/profile_schema.yaml",
    overrides_rel: str = "configs/symbolic/overrides/profiles",
    extra_sources: Optional[List[Path]] = None,
    logger_name: str = "profiles.loader",
) -> ProfileLoadResult:
    """Load default profiles, then apply overrides and optional extra sources."""
    logger = setup_logging(logger_name)
    repo_root = find_repo_root(repo_root)
    defaults_path = repo_root / defaults_rel
    schema_path = repo_root / schema_rel
    overrides_dir = repo_root / overrides_rel

    if not defaults_path.exists():
        raise FileNotFoundError(f"Default profiles not found: {defaults_path}")

    schema_doc = safe_load_yaml(schema_path)
    defaults_doc = safe_load_yaml(defaults_path)
    validate_profile_dict(defaults_doc)

    base_by_id: Dict[str, Dict[str, Any]] = {}
    source_map: Dict[str, List[Path]] = {}

    for p in defaults_doc["profiles"]:
        pid = p["id"]
        base_by_id[pid] = p
        source_map[pid] = [defaults_path]

    if extra_sources:
        for extra in extra_sources:
            if extra.is_dir():
                docs = load_all_yaml_files_in_dir(extra)
                for p_path, doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    try:
                        validate_profile_dict(doc)
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"Skipping invalid profile doc {p_path}: {e}")
                        continue
                    base_by_id = _apply_extends_and_merge(base_by_id, doc, p_path, source_map)
            else:
                doc = safe_load_yaml(extra)
                if isinstance(doc, dict):
                    try:
                        validate_profile_dict(doc)
                        base_by_id = _apply_extends_and_merge(base_by_id, doc, extra, source_map)
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"Skipping invalid profile doc {extra}: {e}")

    for ov_path, ov_doc in load_all_yaml_files_in_dir(overrides_dir):
        if not isinstance(ov_doc, dict):
            continue
        try:
            validate_profile_dict(ov_doc)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Skipping invalid override {ov_path}: {e}")
            continue
        base_by_id = _apply_extends_and_merge(base_by_id, ov_doc, ov_path, source_map)

    profiles: Dict[str, SymbolicProfile] = {}
    hash_by_id: Dict[str, str] = {}
    concat_norm: List[str] = []
    for pid, pdict in sorted(base_by_id.items(), key=lambda kv: kv[0]):
        pdict = dict(pdict)
        pdict.pop("extends", None)
        try:
            validate_profile_dict({"profiles": [pdict]})
        except Exception as e:
            raise ValueError(f"Final merged profile '{pid}' failed validation: {e}") from e
        profiles[pid] = _dict_to_profile(pdict)
        norm = _normalize_profile_for_hash(pdict)
        hash_by_id[pid] = sha256_of_text(norm)
        concat_norm.append(f"# {pid}\n{norm}")

    combined_hash = sha256_of_text("\n".join(concat_norm))
    return ProfileLoadResult(
        profiles=profiles,
        hash_by_id=hash_by_id,
        combined_hash=combined_hash,
        source_maps=source_map,
    )
