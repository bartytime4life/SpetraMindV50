"""CLI module components."""

from .selftest import main as selftest_main
from .pipeline_consistency_checker import main as consistency_check_main

__all__ = ["selftest_main", "consistency_check_main"]