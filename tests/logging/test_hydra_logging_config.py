from hydra import compose, initialize

def test_hydra_job_logging_config():
    """Ensure Hydra job_logging config loads correctly."""
    with initialize(config_path="../../configs/hydra", job_name="test"):
        cfg = compose(config_name="job_logging")
    assert "handlers" in cfg
    assert "spectramind_jsonl" in cfg.handlers
    assert "root" in cfg
