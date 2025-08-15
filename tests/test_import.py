from importlib import import_module

def test_import_package():
    module = import_module("spetramind")
    assert hasattr(module, "__version__")
