# SPDX-License-Identifier: MIT

"""Stable hashing utilities."""

import json
import hashlib
from typing import Any, Dict, Union
from omegaconf import OmegaConf, DictConfig


def stable_dumps(obj: Any) -> str:
    """Return canonical JSON with sorted keys and compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(config: Union[Dict[str, Any], DictConfig]) -> str:
    """Compute SHA256 hash of resolved configuration contents."""
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    data = stable_dumps(config)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
