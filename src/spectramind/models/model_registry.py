"""
Model Registry
--------------

Keeps a mapping of model names to classes for CLI instantiation.
"""

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name):
    return MODEL_REGISTRY[name]

