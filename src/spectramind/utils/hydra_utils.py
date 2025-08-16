from typing import Any, Dict, Tuple


def is_omegaconf(obj: Any) -> bool:
    """Return True if obj looks like an OmegaConf config (duck-typing to avoid hard dependency)."""
    try:
        from omegaconf import DictConfig, ListConfig  # type: ignore

        return isinstance(obj, (DictConfig, ListConfig))
    except Exception:
        return False


def to_resolved_dict(cfg: Any) -> Dict:
    """Convert Hydra/OmegaConf config to a resolved plain dict without interpolation nodes.
    If not an OmegaConf config, try to cast (best-effort)."""
    if is_omegaconf(cfg):
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    if isinstance(cfg, dict):
        return cfg
    # Fallback: try dataclass or object with __dict__
    try:
        import dataclasses

        if dataclasses.is_dataclass(cfg):
            return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    except Exception:
        pass
    try:
        return dict(cfg)
    except Exception:
        return {"value": str(cfg)}


def save_resolved_config(cfg: Any, path_yaml: str, path_json: str) -> Tuple[str, str]:
    """Save resolved config to YAML and JSON for audits."""
    import json

    from .io_utils import save_yaml

    resolved = to_resolved_dict(cfg)
    save_yaml(resolved, path_yaml)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)
    return path_yaml, path_json


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict using dotted keys."""
    items = {}
    for k, v in d.items():
        nk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, nk, sep=sep))
        else:
            items[nk] = v
    return items


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict:
    """Inverse of flatten_dict for dotted keys."""
    root: Dict[str, Any] = {}
    for k, v in d.items():
        cur = root
        parts = k.split(sep)
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return root
