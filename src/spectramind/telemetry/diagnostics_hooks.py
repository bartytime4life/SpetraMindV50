from typing import Any, Callable, Dict, List


class DiagnosticsHooks:
    """
    Registry for custom diagnostic hooks.

    Hooks are functions with no arguments that return a dict of diagnostics.
    """

    def __init__(self) -> None:
        self.hooks: List[Callable[[], Dict[str, Any]]] = []

    def register_hook(self, hook_fn: Callable[[], Dict[str, Any]]) -> None:
        """Register a new diagnostics hook."""
        self.hooks.append(hook_fn)

    def run_hooks(self) -> Dict[str, Any]:
        """Run all registered hooks and return merged diagnostics."""
        results: Dict[str, Any] = {}
        for hook in self.hooks:
            try:
                out = hook()
                if isinstance(out, dict):
                    results.update(out)
                else:
                    results[hook.__name__] = "NON_DICT_RETURN_IGNORED"
            except Exception as e:
                results[hook.__name__] = f"ERROR: {e}"
        return results
