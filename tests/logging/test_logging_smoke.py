import importlib


def test_logging_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRAMIND_LOG_DIR", str(tmp_path / "logs"))
    logging_pkg = importlib.reload(importlib.import_module("src.spectramind.logging"))
    init_logging = logging_pkg.init_logging
    get_version_banner = logging_pkg.get_version_banner
    ensure_log_tables = logging_pkg.ensure_log_tables
    log_event = logging_pkg.log_event
    get_logger = logging_pkg.get_logger

    init_logging()
    ensure_log_tables()
    banner = get_version_banner()
    assert "version" in banner and "config_hash" in banner
    log_event("unit_test", {"ok": True})
    md = tmp_path / "logs" / "v50_debug_log.md"
    jl = tmp_path / "logs" / "events.jsonl"
    assert md.exists()
    assert jl.exists()
    get_logger("spectramind.test").info("hello")
