from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Type

from pydantic import BaseModel

# Internal registry storage
_REGISTRY: Dict[str, Type[BaseModel]] = {}


@dataclass
class SchemaRegistry:
    """Container for Pydantic model registrations."""

    name_to_model: Dict[str, Type[BaseModel]] = field(default_factory=dict)

    def register(self, model: Type[BaseModel], *, name: str | None = None) -> None:
        """Register a model by name; defaults to the class name."""
        key = name or model.__name__
        if key in self.name_to_model and self.name_to_model[key] is not model:
            raise ValueError(
                f"Schema name conflict: {key} already registered to {self.name_to_model[key]}"
            )
        self.name_to_model[key] = model

    def list(self) -> List[str]:
        """List registered model names."""
        return sorted(self.name_to_model.keys())

    def get(self, name: str) -> Type[BaseModel]:
        """Get model class by name."""
        if name not in self.name_to_model:
            raise KeyError(f"Schema '{name}' not registered")
        return self.name_to_model[name]

    def export_json_schemas(self, out_dir: str) -> List[str]:
        """Export JSON Schema files for all registered models. Returns list of written file paths."""
        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        for name in self.list():
            model = self.get(name)
            # Pydantic v2 JSON schema
            schema = model.model_json_schema()
            path = os.path.join(out_dir, f"{name}.schema.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
            written.append(path)
        return written


# Global registry instance mirroring the module-level _REGISTRY for convenience
registry = SchemaRegistry(name_to_model=_REGISTRY)


def register_model(cls: Type[BaseModel]) -> Type[BaseModel]:
    """Class decorator to register Pydantic models by their class name."""
    _REGISTRY[cls.__name__] = cls
    return cls


def list_registered_models() -> List[str]:
    """List names of registered models."""
    return registry.list()


def export_json_schemas(out_dir: str) -> List[str]:
    """Export all registered model JSON Schemas to a directory."""
    return registry.export_json_schemas(out_dir)
