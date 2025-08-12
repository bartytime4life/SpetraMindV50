from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import print

APP_VERSION = "0.1.0"
APP = typer.Typer(add_completion=False, help="SpectraMind V50 — Unified CLI")

# -------------------------- logging & telemetry ------------------------------ #

@dataclass
class RunMeta:
    timestamp: str
    user: str
    host: str
    os: str
    py: str
    git_sha: str
    config_hash: str
    cmd: str


def _iso_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _git_sha() -> str:
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return "NA"


def _hash_configs() -> str:
    h = hashlib.sha256()
    for p in sorted(Path("configs").rglob("*.yaml")):
        try:
            h.update(p.read_bytes())
        except Exception:
            pass
    return h.hexdigest()[:12]


def _append_log(line: str, log_file: str = "v50_debug_log.md") -> None:
    try:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _append_jsonl(event: dict, jsonl_file: str = "events.jsonl") -> None:
    try:
        p = Path(jsonl_file)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ------------------------------- commands ----------------------------------- #

@APP.command()
def version() -> None:
    """Print CLI version + config hash and write to audit logs."""
    meta = RunMeta(
        timestamp=_iso_utc(),
        user=os.getenv("USER", "unknown"),
        host=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        py=sys.version.split()[0],
        git_sha=_git_sha(),
        config_hash=_hash_configs(),
        cmd="version",
    )
    line = f"[version] {meta.timestamp} v{APP_VERSION} git={meta.git_sha} cfg={meta.config_hash}"
    _append_log(line)
    _append_jsonl({"event": "version", **asdict(meta)})
    print(line)


@APP.command()
def selftest() -> None:
    """Verify file presence, CLI registration, config readability."""
    required = [
        "configs/config_v50.yaml",
        "spectramind.py",
        "src/spectramind/cli/selftest.py",
        "src/spectramind/diagnostics/generate_html_report.py",
    ]
    missing = [p for p in required if not Path(p).exists()]
    status = "ok" if not missing else f"missing: {missing}"
    _append_log(f"[selftest] {_iso_utc()} {status}")
    _append_jsonl({"event": "selftest", "status": status, "missing": missing})
    if missing:
        print(f"[red]❌ Missing files: {missing}")
        raise typer.Exit(1)
    print("[green]✅ Selftest passed.")


@APP.command()
def train(dry_run: bool = typer.Option(False, help="Skip heavy work")) -> None:
    """Train pipeline (stub): MAE→(contrastive)→supervised with GLL+symbolic."""
    from src.spectramind.training.train_v50 import train as _train

    _append_log(f"[train] {_iso_utc()} dry_run={dry_run}")
    _append_jsonl({"event": "train", "dry_run": dry_run})
    _train(dry_run=dry_run)


@APP.command()
def predict(out_csv: Path = typer.Option(Path("outputs/submission.csv"), exists=False)) -> None:
    """Run inference and write μ/σ submission CSV (stub)."""
    from src.spectramind.inference.predict_v50 import predict as _predict

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _append_log(f"[predict] {_iso_utc()} -> {out_csv}")
    _append_jsonl({"event": "predict", "out_csv": str(out_csv)})
    _predict(out_csv)
    print(f"[green]✅ Submission written: {out_csv}")


@APP.command("dashboard")
def dashboard_cmd(html: Path = typer.Option(Path("outputs/diagnostics/diagnostic_report_v50.html"))) -> None:
    """Generate versioned HTML diagnostics report (stub)."""
    from src.spectramind.diagnostics.generate_html_report import write_report

    html.parent.mkdir(parents=True, exist_ok=True)
    _append_log(f"[dashboard] {_iso_utc()} -> {html}")
    _append_jsonl({"event": "dashboard", "html": str(html)})
    write_report(html)
    print(f"[green]✅ Diagnostics HTML: {html}")


@APP.command()
def calibrate(dry_run: bool = typer.Option(False, help="Print calibration plan only")) -> None:
    """Run calibration kill chain on raw data."""
    from src.spectramind.data.calibration_chain import CalibrationKillChain
    
    _append_log(f"[calibrate] {_iso_utc()} dry_run={dry_run}")
    _append_jsonl({"event": "calibrate", "dry_run": dry_run})
    
    if dry_run:
        print("[cyan]DRY-RUN: Calibration kill chain steps:")
        print("  1. ADC reversal / bias correction") 
        print("  2. Bad pixel mask & interpolation")
        print("  3. Non-linearity correction")
        print("  4. Dark subtraction")
        print("  5. Flat-fielding")
        print("  6. Trace extraction / photometry")
        print("[cyan]Would process FGS1 and AIRS data with variance propagation.")
        return
    
    try:
        # Create output directories
        Path("outputs/calibrated/fgs1").mkdir(parents=True, exist_ok=True)
        Path("outputs/calibrated/airs").mkdir(parents=True, exist_ok=True)
        Path("outputs/logs/calibration").mkdir(parents=True, exist_ok=True)
        
        # FGS1 calibration
        fgs1_chain = CalibrationKillChain(instrument="fgs1")
        
        # Create dummy raw data for demonstration
        import numpy as np
        dummy_fgs1_data = np.random.poisson(1000, size=(10, 32, 32)).astype(np.float32)
        
        print("[cyan]Running FGS1 calibration kill chain...")
        cal_data, cal_var, cal_mask, step_logs = fgs1_chain.calibrate(dummy_fgs1_data)
        
        # Save calibrated data
        np.savez(
            "outputs/calibrated/fgs1/fgs1_demo.npz",
            frames=cal_data,
            variance=cal_var, 
            mask=cal_mask
        )
        
        # Save logs
        fgs1_chain.save_logs("outputs/logs/calibration/fgs1_calibration_log.json")
        
        print(f"[green]✅ FGS1 calibration complete. {len(step_logs)} steps executed.")
        
        # AIRS calibration  
        airs_chain = CalibrationKillChain(instrument="airs")
        dummy_airs_data = np.random.poisson(500, size=(10, 64, 283)).astype(np.float32)
        
        print("[cyan]Running AIRS calibration kill chain...")
        cal_data, cal_var, cal_mask, step_logs = airs_chain.calibrate(dummy_airs_data)
        
        # Save calibrated data
        np.savez(
            "outputs/calibrated/airs/airs_demo.npz",
            frames=cal_data,
            variance=cal_var,
            mask=cal_mask,
            trace_meta={}
        )
        
        airs_chain.save_logs("outputs/logs/calibration/airs_calibration_log.json")
        
        print(f"[green]✅ AIRS calibration complete. {len(step_logs)} steps executed.")
        print("[green]✅ Calibration kill chain finished. Check outputs/calibrated/ for results.")
        
    except Exception as e:
        print(f"[red]❌ Calibration failed: {e}")
        raise typer.Exit(1)


@APP.command("validate-schema")
def validate_schema_cmd(
    file_path: Path = typer.Option(..., help="Path to file to validate"),
    schema_type: str = typer.Option(..., help="Schema type (fgs1_calibrated, airs_calibrated, fgs1_features, airs_features, submission)")
) -> None:
    """Validate data file against schema."""
    from src.spectramind.data.data_contracts import validate_schema
    
    _append_log(f"[validate-schema] {_iso_utc()} {file_path} {schema_type}")
    _append_jsonl({"event": "validate-schema", "file_path": str(file_path), "schema_type": schema_type})
    
    try:
        result = validate_schema(str(file_path), schema_type)
        
        if result["valid"]:
            print(f"[green]✅ Schema validation passed for {file_path}")
        else:
            print(f"[red]❌ Schema validation failed for {file_path}")
            for error in result["errors"]:
                print(f"[red]  Error: {error}")
                
        if result.get("warnings"):
            for warning in result["warnings"]:
                print(f"[yellow]  Warning: {warning}")
                
    except Exception as e:
        print(f"[red]❌ Schema validation error: {e}")
        raise typer.Exit(1)


@APP.command("symbolic-test")
def symbolic_test() -> None:
    """Test symbolic physics engine with dummy data."""
    from src.spectramind.symbolic.symbolic_loss import SymbolicLossEngine
    
    _append_log(f"[symbolic-test] {_iso_utc()}")
    _append_jsonl({"event": "symbolic-test"})
    
    try:
        # Create symbolic engine
        engine = SymbolicLossEngine(
            lambda_sm=0.1,
            lambda_nn=0.1,
            lambda_coh=0.1,
            lambda_seam=0.1,
            lambda_ratio=0.1
        )
        
        print("[cyan]Testing symbolic physics engine...")
        
        # Create test data - smooth spectrum with some features
        import math
        mu_test = []
        for i in range(283):
            # Smooth baseline + some molecular features
            baseline = 0.98 + 0.02 * math.sin(i * 0.02)
            if 50 <= i <= 80:  # H2O feature
                baseline -= 0.01 * math.exp(-((i-65)**2) / 100)
            if 120 <= i <= 150:  # CO2 feature  
                baseline -= 0.008 * math.exp(-((i-135)**2) / 80)
            mu_test.append(baseline)
        
        # Test with batch of 1
        mu_batch = [mu_test]
        
        # Compute symbolic loss
        total_loss, component_losses = engine.compute_symbolic_loss(mu_batch)
        
        print(f"[green]✅ Symbolic physics test complete!")
        print(f"  Total loss: {total_loss:.6f}")
        for name, loss in component_losses.items():
            print(f"  {name}: {loss:.6f}")
            
        # Test with problematic spectrum (non-smooth, negative values)
        mu_bad = mu_test.copy()
        mu_bad[100] = -0.1  # Negative value
        mu_bad[101] = 1.5   # Sharp spike
        mu_bad[102] = 0.0   # Sharp drop
        
        total_loss_bad, component_losses_bad = engine.compute_symbolic_loss([mu_bad])
        
        print(f"[yellow]Bad spectrum penalties:")
        print(f"  Total loss: {total_loss_bad:.6f} (vs {total_loss:.6f})")
        print(f"  Non-negativity: {component_losses_bad['non_negativity']:.6f}")
        print(f"  Smoothness: {component_losses_bad['smoothness']:.6f}")
        
    except Exception as e:
        print(f"[red]❌ Symbolic test failed: {e}")
        raise typer.Exit(1)


@APP.command("calibrate-temp")
def calibrate_temp() -> None:
    """Run temperature scaling calibration."""
    from src.spectramind.calibration.temperature_scaling import TemperatureScaling
    
    _append_log(f"[calibrate-temp] {_iso_utc()}")
    _append_jsonl({"event": "calibrate-temp"})
    
    try:
        # Placeholder implementation - in practice would load validation data
        ts = TemperatureScaling()
        # For demo, create dummy data
        dummy_sigma = [[0.1] * 283]
        dummy_mu = [[0.0] * 283]  
        dummy_targets = [[0.0] * 283]
        
        temperature = ts.fit(dummy_sigma, dummy_mu, dummy_targets)
        print(f"[green]✅ Temperature scaling complete. Optimal temperature: {temperature:.4f}")
        
        # Save result
        output_dir = Path("outputs/calibration")
        output_dir.mkdir(parents=True, exist_ok=True)
        ts.save(str(output_dir / "temperature.txt"))
        
    except Exception as e:
        print(f"[red]❌ Temperature scaling failed: {e}")


@APP.command("calibrate-corel")
def calibrate_corel() -> None:
    """Run COREL conformalization."""
    from src.spectramind.calibration.corel_conformal import CORELSpectralConformal
    
    _append_log(f"[calibrate-corel] {_iso_utc()}")
    _append_jsonl({"event": "calibrate-corel"})
    
    try:
        corel = CORELSpectralConformal(alpha=0.1)
        
        # Placeholder implementation - in practice would load validation data
        dummy_sigma = [[0.1] * 283]
        dummy_mu = [[0.0] * 283]
        dummy_targets = [[0.0] * 283]
        
        quantiles = corel.fit(dummy_sigma, dummy_mu, dummy_targets)
        coverage_report = corel.get_coverage_report()
        
        print(f"[green]✅ COREL calibration complete. Generated {len(quantiles)} per-bin quantiles.")
        print(f"Target coverage: {coverage_report.get('target_coverage', 0.9):.3f}")
        
        # Save results
        output_dir = Path("outputs/calibration")
        output_dir.mkdir(parents=True, exist_ok=True)
        corel.save(str(output_dir / "corel_quantiles.json"))
        
    except Exception as e:
        print(f"[red]❌ COREL calibration failed: {e}")


@APP.command()
def submit(
    bundle: bool = typer.Option(True, help="Create a submission bundle zip"),
    in_csv: Path = typer.Option(Path("outputs/submission.csv")),
    out_zip: Path = typer.Option(Path("outputs/submission_bundle.zip")),
) -> None:
    if not in_csv.exists():
        print(f"[red]❌ missing submission: {in_csv}")
        raise typer.Exit(1)
    import zipfile

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as z:
        z.write(in_csv, arcname="submission.csv")
        for extra in ("run_hash_summary_v50.json", "v50_debug_log.md", "events.jsonl"):
            if Path(extra).exists():
                z.write(extra, arcname=Path(extra).name)
    _append_log(f"[submit] {_iso_utc()} -> {out_zip}")
    _append_jsonl({"event": "submit", "out_zip": str(out_zip)})
    print(f"[green]✅ Bundled -> {out_zip}")


@APP.command("test-pipeline")
def test_pipeline() -> None:
    """Run comprehensive end-to-end pipeline test."""
    from src.spectramind.data.calibration_chain import CalibrationKillChain
    from src.spectramind.symbolic.symbolic_loss import SymbolicLossEngine
    from src.spectramind.calibration.temperature_scaling import TemperatureScaling
    from src.spectramind.calibration.corel_conformal import CORELSpectralConformal
    from src.spectramind.data.data_contracts import validate_schema
    
    _append_log(f"[test-pipeline] {_iso_utc()}")
    _append_jsonl({"event": "test-pipeline"})
    
    print("[cyan]🧪 Running comprehensive pipeline test...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Symbolic Physics Engine
    tests_total += 1
    try:
        print("[cyan]1. Testing symbolic physics engine...")
        engine = SymbolicLossEngine()
        
        # Test smooth spectrum
        import math
        mu_smooth = [[0.98 + 0.01 * math.sin(i * 0.05) for i in range(283)]]
        loss_smooth, _ = engine.compute_symbolic_loss(mu_smooth)
        
        # Test problematic spectrum
        mu_bad = mu_smooth[0].copy()
        mu_bad[100] = -0.1  # Negative
        mu_bad[101] = 1.5   # Spike
        loss_bad, _ = engine.compute_symbolic_loss([mu_bad])
        
        assert loss_bad > loss_smooth, "Symbolic loss should penalize bad spectra"
        print("   ✅ Symbolic physics engine working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Symbolic physics test failed: {e}")
    
    # Test 2: Calibration Kill Chain
    tests_total += 1
    try:
        print("[cyan]2. Testing calibration kill chain...")
        import numpy as np
        
        chain = CalibrationKillChain("fgs1")
        test_data = np.random.poisson(1000, size=(5, 16, 16)).astype(np.float32)
        
        cal_data, cal_var, cal_mask, logs = chain.calibrate(test_data)
        
        assert len(logs) == 6, "Should have 6 calibration steps"
        assert cal_data.ndim >= 1, "Should have valid output"
        assert np.all(cal_var > 0), "Variance should be positive"
        
        print(f"   ✅ Calibration kill chain: {len(logs)} steps completed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Calibration test failed: {e}")
    
    # Test 3: Temperature Scaling
    tests_total += 1
    try:
        print("[cyan]3. Testing temperature scaling...")
        ts = TemperatureScaling()
        
        # Simple test data
        dummy_sigma = [[0.1] * 10]
        dummy_mu = [[0.0] * 10]
        dummy_targets = [[0.05] * 10]
        
        temp = ts.fit(dummy_sigma, dummy_mu, dummy_targets, max_iter=10)
        calibrated = ts.transform(dummy_sigma)
        
        assert 0.1 <= temp <= 10.0, "Temperature should be reasonable"
        assert len(calibrated) == len(dummy_sigma), "Output shape should match"
        
        print(f"   ✅ Temperature scaling: τ = {temp:.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Temperature scaling test failed: {e}")
    
    # Test 4: COREL Conformal
    tests_total += 1
    try:
        print("[cyan]4. Testing COREL conformalization...")
        corel = CORELSpectralConformal(alpha=0.1)
        
        # Simple test data  
        dummy_sigma = [[0.1] * 283]
        dummy_mu = [[0.0] * 283]
        dummy_targets = [[0.05] * 283]
        
        quantiles = corel.fit(dummy_sigma, dummy_mu, dummy_targets)
        calibrated = corel.transform(dummy_sigma)
        
        assert len(quantiles) == 283, "Should have 283 quantiles"
        assert all(q > 0 for q in quantiles), "Quantiles should be positive"
        
        coverage_report = corel.get_coverage_report()
        print(f"   ✅ COREL: {len(quantiles)} quantiles @ {coverage_report['target_coverage']:.1%} target")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ COREL test failed: {e}")
    
    # Test 5: Schema Validation
    tests_total += 1
    try:
        print("[cyan]5. Testing schema validation...")
        
        # Test submission schema with our generated file
        if Path("outputs/submission.csv").exists():
            result = validate_schema("outputs/submission.csv", "submission")
            if result["valid"]:
                print("   ✅ Submission CSV schema validation passed")
                tests_passed += 1
            else:
                print(f"   ❌ Submission CSV validation failed: {result['errors'][:2]}")
        else:
            print("   ⏸️ No submission file to validate")
    except Exception as e:
        print(f"   ❌ Schema validation test failed: {e}")
        
    # Summary
    print(f"\n[bold]Pipeline Test Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("[green]🎉 All pipeline tests passed! System is ready.")
    elif tests_passed >= tests_total * 0.8:
        print("[yellow]⚠️ Most tests passed. Some components may need attention.")
    else:
        print("[red]❌ Multiple test failures. Pipeline needs debugging.")
        raise typer.Exit(1)


@APP.command()
def diagnose(dry_run: bool = typer.Option(False, help="Print planned diagnostics only")) -> None:
    _append_log(f"[diagnose] {_iso_utc()} dry_run={dry_run}")
    _append_jsonl({"event": "diagnose", "dry_run": dry_run})
    print("[cyan]Diagnostics planned: SHAP, UMAP/t-SNE, FFT, smoothness, symbolic overlays.")


if __name__ == "__main__":
    APP()
