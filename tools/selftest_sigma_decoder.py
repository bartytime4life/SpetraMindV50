#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Self-test for configs/model/sigma_decoder
- Parses all YAMLs in the sigma_decoder group
- Performs sanity checks on keys/values/types
- Appends a Markdown section to v50_debug_log.md
- Appends a JSON line to logs/events.jsonl
- Exits non-zero on failure

Run:
  python tools/selftest_sigma_decoder.py
"""
import json, sys, os, hashlib, datetime, subprocess
from pathlib import Path

# Optional imports with lightweight bootstrap note
try:
    import yaml  # PyYAML
except Exception as e:
    print("[SELFTEST] Missing dependency: pyyaml. Install with: python -m pip install --user pyyaml", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "model" / "sigma_decoder"
LOG_MD = ROOT / "v50_debug_log.md"
EVENTS = ROOT / "logs" / "events.jsonl"

def load_yaml(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def assert_keys(d: dict, keys, ctx: str):
    for k in keys:
        if k not in d:
            raise AssertionError(f"[SELFTEST] Missing key '{k}' in {ctx}")

def main():
    required = [
        "_group_.yaml",
        "flow.yaml",
        "quantile.yaml",
        "ensemble.yaml",
        "calibration.yaml",
        "monitor.yaml",
        "export.yaml",
    ]
    missing = [f for f in required if not (CFG_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"[SELFTEST] Missing config files: {missing}")

    # Load all YAMLs
    yg = load_yaml(CFG_DIR / "_group_.yaml")
    yflow = load_yaml(CFG_DIR / "flow.yaml")
    yq = load_yaml(CFG_DIR / "quantile.yaml")
    yens = load_yaml(CFG_DIR / "ensemble.yaml")
    ycal = load_yaml(CFG_DIR / "calibration.yaml")
    ymon = load_yaml(CFG_DIR / "monitor.yaml")
    yexp = load_yaml(CFG_DIR / "export.yaml")

    # Basic structure checks
    assert "defaults" in yg and isinstance(yg["defaults"], list), "[SELFTEST] _group_.yaml must have list 'defaults'"
    assert any("flow" == d or (isinstance(d, dict) and "flow" in d.values()) for d in yg["defaults"]), "[SELFTEST] defaults must include flow"
    assert "model" in yflow and "sigma_decoder" in yflow["model"], "[SELFTEST] flow.yaml shape invalid"
    assert "model" in yq and "sigma_decoder" in yq["model"], "[SELFTEST] quantile.yaml shape invalid"
    assert "model" in yens and "sigma_decoder" in yens["model"], "[SELFTEST] ensemble.yaml shape invalid"
    assert "calibration" in ycal, "[SELFTEST] calibration.yaml must contain 'calibration'"
    assert "monitor" in ymon, "[SELFTEST] monitor.yaml must contain 'monitor'"
    assert "export" in yexp, "[SELFTEST] export.yaml must contain 'export'"

    # Flow specifics
    f = yflow["model"]["sigma_decoder"]
    assert f.get("name") == "flow", "[SELFTEST] flow.yaml model.sigma_decoder.name must be 'flow'"
    assert isinstance(f.get("hidden_dims"), list) and all(isinstance(x, int) for x in f["hidden_dims"]), "[SELFTEST] hidden_dims must be a list[int]"
    sm = f.get("sigma_min_ppm", {})
    assert sm.get("fgs1", None) is not None and sm.get("airs", None) is not None, "[SELFTEST] sigma_min_ppm must define fgs1 and airs"
    assert 0.0 < float(sm["fgs1"]) < 100.0, "[SELFTEST] fgs1 sigma_min_ppm unreasonable"
    assert 0.0 < float(sm["airs"]) < 1000.0, "[SELFTEST] airs sigma_min_ppm unreasonable"

    # Quantile specifics
    q = yq["model"]["sigma_decoder"]
    assert q.get("name") == "quantile", "[SELFTEST] quantile.yaml model.sigma_decoder.name must be 'quantile'"
    qs = q.get("quantiles", [])
    assert qs == [0.10, 0.50, 0.90], "[SELFTEST] quantiles must be [0.10, 0.50, 0.90]"
    mono = q.get("monotonicity", {})
    assert mono.get("enabled", False) is True, "[SELFTEST] quantile monotonicity must be enabled"

    # Ensemble specifics
    e = yens["model"]["sigma_decoder"]
    assert e.get("name") == "ensemble", "[SELFTEST] ensemble.yaml model.sigma_decoder.name must be 'ensemble'"
    blend = e.get("blend", {})
    assert abs(float(blend.get("flow_w", -1)) + float(blend.get("quantile_w", -1)) - 1.0) < 1e-6, "[SELFTEST] ensemble blend weights must sum to 1.0"

    # Calibration specifics
    c = ycal["calibration"]
    assert c.get("temperature", {}).get("enabled", None) is not None, "[SELFTEST] calibration.temperature.enabled must exist"
    assert c.get("corel", {}).get("enabled", None) is not None, "[SELFTEST] calibration.corel.enabled must exist"

    # Monitor specifics
    m = ymon["monitor"]["coverage"]
    assert isinstance(m.get("targets", []), list) and 0.0 < min(m["targets"]) < 1.0, "[SELFTEST] monitor.coverage.targets invalid"

    # Export specifics
    ex = yexp["export"]["artifacts"]
    assert ex.get("write_summary_json", False) is True, "[SELFTEST] export.write_summary_json must be true"

    # Hash summary for provenance
    hashes = {name: sha256_of_file(CFG_DIR / name) for name in required}
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Append Markdown log
    LOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with LOG_MD.open("a", encoding="utf-8") as md:
        md.write("\n\n---\n")
        md.write(f"### {now} — sigma_decoder self-test\n")
        md.write(f"- Result: **PASS**\n")
        md.write(f"- Files checked: {len(required)}\n")
        for k, v in hashes.items():
            md.write(f"  - `{k}`: `{v}`\n")
        md.write("- Notes: configs present; keys validated; ensemble weights sane; coverage targets sane.\n")

    # Append JSONL event
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    evt = {
        "ts": now,
        "event": "sigma_decoder_selftest",
        "result": "PASS",
        "files": required,
        "hashes": hashes,
        "repo_git_sha": None,
    }
    # Best-effort git sha
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
        evt["repo_git_sha"] = sha
    except Exception:
        pass
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt) + "\n")

    print("[SELFTEST] sigma_decoder: PASS")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Log failure path too
        try:
            now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            with open("v50_debug_log.md", "a", encoding="utf-8") as md:
                md.write("\n\n---\n")
                md.write(f"### {now} — sigma_decoder self-test\n")
                md.write(f"- Result: **FAIL**\n")
                md.write(f"- Error: {e}\n")
            with open("logs/events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": now, "event": "sigma_decoder_selftest", "result": "FAIL", "error": str(e)}) + "\n")
        except Exception:
            pass
        print(f"[SELFTEST] FAILURE: {e}", file=sys.stderr)
        sys.exit(1)
