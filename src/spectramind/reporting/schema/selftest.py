from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

from .cli_log import CLILogEntry, CLILogBatch
from .diagnostic_summary import BinStat, DiagnosticSummary, GlobalMetricSummary
from .symbolic_violation import SymbolicRuleHit, SymbolicViolationSummary
from .calibration import BinCalibration, CalibrationSummary
from .html_report import HTMLAsset, HTMLReportManifest
from .report_manifest import (
    Artifact,
    DatasetMeta,
    GeneratorMeta,
    ReportPackageManifest,
)
from .schema_registry import list_registered_models, export_json_schemas


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_selftest() -> None:
    # Build small instances
    entry = CLILogEntry(
        timestamp=_iso_now(),
        cli="spectramind",
        subcommand="diagnose dashboard",
        args=["--html-out", "report.html"],
        exit_code=0,
        duration_s=12.34,
        user="andy",
        host="local",
        metrics={"gll": -1234.5},
    )
    batch = CLILogBatch(
        entries=[entry],
        by_subcommand_count={"diagnose dashboard": 1},
        by_config_hash_count={},
        total_calls=1,
    )

    gms = GlobalMetricSummary(
        num_planets=1,
        num_bins=283,
        mean_gll=-4.56,
        median_gll=-4.5,
        mean_rmse=0.12,
        mean_mae=0.09,
    )
    per_bin = [BinStat(bin_index=i, gll=-0.01 * i) for i in range(3)]
    diag = DiagnosticSummary(planets=["PlanetX"], global_metrics=gms, per_bin=per_bin)

    sym = SymbolicViolationSummary(
        planet="PlanetX",
        top_rules=[SymbolicRuleHit(rule_id="nonnegativity", score=0.0, affected_bins=[])],
        total_violation_norm=0.0,
    )

    cal = CalibrationSummary(
        target_coverage=0.9,
        observed_coverage=0.88,
        num_bins=3,
        per_bin=[
            BinCalibration(bin_index=0, nominal_sigma=0.1, calibrated_sigma=0.11, residual_abs=0.09, covered=True),
            BinCalibration(bin_index=1, nominal_sigma=0.1, calibrated_sigma=0.10, residual_abs=0.12, covered=False),
            BinCalibration(bin_index=2, nominal_sigma=0.1, calibrated_sigma=0.09, residual_abs=0.08, covered=True),
        ],
    )

    html = HTMLReportManifest(
        title="Diagnostics Report",
        assets=[
            HTMLAsset(kind="iframe", title="UMAP", path="umap.html", mime="text/html", width=1200, height=900),
            HTMLAsset(kind="png", title="GLL Heatmap", path="gll_heatmap.png", mime="image/png"),
        ],
    )

    pkg = ReportPackageManifest(
        dataset=DatasetMeta(dataset_name="ariel-neurips-2025", split="val", num_planets=1, num_bins=283),
        generator=GeneratorMeta(cli="spectramind", version="v50.0", environment={"PYTHONHASHSEED": "0"}),
        diagnostics=diag,
        calibration=cal,
        symbolic=[sym],
        html_report=html,
        artifacts=[
            Artifact(name="umap", path="umap.html", kind="html", description="Interactive UMAP plot"),
            Artifact(name="gll_heatmap", path="gll_heatmap.png", kind="png"),
        ],
    )

    # Basic serialization checks
    _ = entry.to_json()
    _ = batch.to_json()
    _ = diag.to_json()
    _ = sym.to_json()
    _ = cal.to_json()
    _ = html.to_json()
    js = pkg.to_json()
    d = json.loads(js)
    assert d["dataset"]["dataset_name"] == "ariel-neurips-2025"

    # Registry listing and schema export
    names = list_registered_models()
    assert "CLILogEntry" in names
    assert "ReportPackageManifest" in names

    with tempfile.TemporaryDirectory() as td:
        written = export_json_schemas(td)
        assert any(p.endswith("ReportPackageManifest.schema.json") for p in written)
        # write a combined JSON to verify no errors
        with open(os.path.join(td, "package_example.json"), "w", encoding="utf-8") as f:
            f.write(js)

    print("OK: reporting.schema selftest passed.")


if __name__ == "__main__":
    run_selftest()
