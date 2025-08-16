import importlib
from typing import Any, Dict

def resolve_callable(target: str):
    """
    Resolve a dotted 'module:callable' or 'module.sub:callable' path to an object.
    """
    if ":" in target:
        module_name, attr = target.split(":")
    else:
        # Support "module.attr" form as well
        parts = target.split(".")
        module_name, attr = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)

def build_from_target(cfg: Dict[str, Any]) -> Any:
    """
    Instantiate a class or call a function given:
    cfg = {"target": "package.module:ClassName", "params": {...}}
    """
    target = cfg.get("target")
    if not target:
        raise ValueError("Missing 'target' in config block.")
    ctor = resolve_callable(target)
    params = cfg.get("params") or {}
    return ctor(**params)
