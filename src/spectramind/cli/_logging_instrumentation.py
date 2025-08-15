"""
Typer CLI auto-instrumentation helpers.

Usage in any CLI module (BEFORE constructing the app or at top of file):
from spectramind.logging.bootstrap import init_logging_for_cli
meta = init_logging_for_cli(command="spectramind train", args={"--config":"path/to/yaml"})
# later, when executing:
from spectramind.logging import log_cli_call
log_cli_call("spectramind train", {"--config":"path/to/yaml"}, meta["config_hash"], meta["version"], extra={"duration_s": 12.3})

This module also provides a decorator to wrap main() style functions and a helper
to patch a Typer app's callback for automatic logging of command invocations.
"""

import functools
import time
from typing import Any, Callable, Dict

from spectramind.logging import get_logger, log_cli_call
from spectramind.logging.bootstrap import init_logging_for_cli


def loggable_cli(command_name: str, arg_fn: Callable[[], Dict[str, Any]]):
    """
    Decorator for main() in simple scripts without Typer.

    Example:
        @loggable_cli("spectramind selftest", lambda: {"--mode":"fast"})
        def main():
            ...
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            meta = init_logging_for_cli(command=command_name, args=arg_fn())
            logger = get_logger("spectramind.cli")
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                dt = time.time() - t0
                try:
                    log_cli_call(
                        command_name,
                        arg_fn(),
                        meta["config_hash"],
                        meta["version"],
                        extra={"duration_s": round(dt, 3)},
                    )
                    logger.info(f"Completed {command_name} in {dt:.3f}s")
                except Exception as e:
                    logger.error(f"Failed to record cli_call event: {e}")

        return wrapper

    return decorator


def instrument_typer_app(app, command_name: str, args: Dict[str, Any]):
    """
    Attach a Typer callback that logs start/end for the whole app invocation.
    Call this once, after app = Typer() is created and before app() is run.
    """
    meta = init_logging_for_cli(command=command_name, args=args)
    logger = get_logger("spectramind.cli")

    @app.callback(invoke_without_command=True)
    def _root_callback():
        # This callback runs before any subcommand
        logger.info(f"Typer app invoked: {command_name}")

    def _teardown(result_ok: bool, started_ts: float):
        try:
            log_cli_call(
                command_name,
                args,
                meta["config_hash"],
                meta["version"],
                extra={
                    "ok": result_ok,
                    "duration_s": round(time.time() - started_ts, 3),
                },
            )
        except Exception as e:
            logger.error(f"CLI teardown log failed: {e}")

    # Return a context manager-style helper for main()
    class _AppRun:
        def __enter__(self):
            self._t0 = time.time()
            return meta

        def __exit__(self, exc_type, exc, tb):
            _teardown(result_ok=(exc is None), started_ts=self._t0)
            # don't suppress exceptions
            return False

    return _AppRun(), meta
