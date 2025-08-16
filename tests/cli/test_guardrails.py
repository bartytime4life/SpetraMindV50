# -*- coding: utf-8 -*-
"""Tests for guardrail utilities used by CLI commands."""
import pytest

from spectramind.cli.cli_guardrails import confirm_guard, dry_run_guard


def test_dry_run_guard(monkeypatch):
    called = {"ran": False}

    def action():
        called["ran"] = True
        return 42

    guarded = dry_run_guard(action)
    # When dry_run=True the wrapped function should not execute and return 0
    assert guarded(dry_run=True) == 0
    assert called["ran"] is False

    # With dry_run=False the action should run and return its value
    assert guarded(dry_run=False) == 42
    assert called["ran"] is True


def test_confirm_guard(monkeypatch):
    called = {"ran": False}

    def action():
        called["ran"] = True
        return 1

    guarded = confirm_guard(prompt="?")(action)

    # Auto-confirm via environment variable
    monkeypatch.setenv("SPECTRAMIND_CONFIRM", "yes")
    assert guarded(confirm=False) == 1
    assert called["ran"] is True

    # When user declines, the guard should exit before running action
    called["ran"] = False
    monkeypatch.setenv("SPECTRAMIND_CONFIRM", "")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    with pytest.raises(SystemExit):
        guarded(confirm=False)
    assert called["ran"] is False
