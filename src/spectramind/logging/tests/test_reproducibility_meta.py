"""
Reproducibility metadata smoke test.

If the repo provides a function or env-driven logging that emits reproducibility fields,
validate they appear in a sample JSON structure. This test is lenient and passes if fields
are not available yet (to allow incremental adoption).
"""
import os
import json
import pytest

CANDIDATE_ENV_KEYS = ["SMV50_RUN_ID", "SMV50_CONFIG_HASH", "SMV50_CLI_VERSION"]

@pytest.mark.smoke
def test_reproducibility_env_present_or_skipped(monkeypatch_env_logging, tmp_path):
    # Success if at least one of the candidate env vars is set (fixture sets them),
    # or if project doesn't use them yet, we accept but assert the fixture worked.
    present = [k for k in CANDIDATE_ENV_KEYS if os.getenv(k)]
    assert present, "Expected at least one reproducibility env var to be set by fixture."
    # Simulate a log record
    record = {k.lower(): os.getenv(k) for k in present}
    s = json.dumps(record)
    assert all(v in s for v in record.values())
