# Logging Tests

This directory contains unit and integration tests for the SpectraMind V50 logging system.

## Coverage

- **Console Handler** (`test_console_handler.py`): Verifies stdout logging.
- **File Rotation** (`test_file_rotation.py`): Ensures rotating log files behave correctly.
- **JSONL Stream** (`test_jsonl_stream.py`): Validates structured JSONL event logging.
- **Integration Pipeline Logging** (`test_integration_pipeline_logging.py`): Full logging pipeline.
- **Hydra Config** (`test_hydra_logging_config.py`): Confirms Hydra `job_logging` loads correctly.
- **Reproducibility Metadata** (`test_reproducibility_log_capture.py`): Captures ENV, Git, and config hash.

## Usage

Run all logging tests with:

```bash
pytest tests/logging -v
```

These tests are included in the CI/CD pipeline to ensure the logging system is reliable, reproducible, and Hydra-safe.
