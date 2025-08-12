"""
Enhanced HTML diagnostics report generator.

Generates comprehensive diagnostics HTML report with:
- Run metadata and provenance
- Symbolic physics violation summaries
- Model performance metrics
- Coverage plots (placeholder)
- SHAP/UMAP visualizations (placeholder)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


def write_report(path: Path) -> None:
    """
    Write comprehensive diagnostics HTML report.
    
    Args:
        path: Output path for HTML report
    """
    # Collect diagnostics data
    diagnostics_data = collect_diagnostics_data()
    
    # Generate HTML content
    html_content = generate_html_report(diagnostics_data)
    
    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")


def collect_diagnostics_data() -> Dict[str, Any]:
    """Collect all available diagnostics data."""
    from src.spectramind.utils.hash_utils import git_sha, hash_configs
    
    data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "git_sha": git_sha(),
            "config_hash": hash_configs(),
            "version": "0.1.0"
        },
        "calibration": collect_calibration_data(),
        "symbolic": collect_symbolic_data(),
        "performance": collect_performance_data(),
        "files": collect_file_data()
    }
    
    return data


def collect_calibration_data() -> Dict[str, Any]:
    """Collect calibration diagnostics."""
    calibration_data = {
        "temperature_scaling": None,
        "corel_quantiles": None,
        "calibration_logs": []
    }
    
    # Check for temperature scaling results
    temp_file = Path("outputs/calibration/temperature.txt")
    if temp_file.exists():
        try:
            temperature = float(temp_file.read_text().strip().split('\n')[0])
            calibration_data["temperature_scaling"] = {
                "temperature": temperature,
                "status": "fitted" if temperature > 0 else "error"
            }
        except (ValueError, IndexError):
            calibration_data["temperature_scaling"] = {"status": "error"}
    
    # Check for COREL results
    corel_file = Path("outputs/calibration/corel_quantiles.json")
    if corel_file.exists():
        try:
            with open(corel_file) as f:
                corel_data = json.load(f)
            calibration_data["corel_quantiles"] = {
                "n_quantiles": len(corel_data.get("quantiles", [])),
                "target_coverage": corel_data.get("target_coverage", 0.9),
                "status": "fitted"
            }
        except (json.JSONDecodeError, KeyError):
            calibration_data["corel_quantiles"] = {"status": "error"}
    
    # Check for calibration logs
    log_dir = Path("outputs/logs/calibration")
    if log_dir.exists():
        for log_file in log_dir.glob("*.json"):
            try:
                with open(log_file) as f:
                    log_data = json.load(f)
                calibration_data["calibration_logs"].append({
                    "instrument": log_data.get("instrument", "unknown"),
                    "total_steps": log_data.get("total_steps", 0),
                    "total_time_ms": log_data.get("total_time_ms", 0),
                    "file": log_file.name
                })
            except (json.JSONDecodeError, KeyError):
                continue
    
    return calibration_data


def collect_symbolic_data() -> Dict[str, Any]:
    """Collect symbolic physics diagnostics."""
    # In practice, would load from actual run logs
    return {
        "engine_status": "available",
        "constraint_types": [
            "smoothness", "non_negativity", "molecular_coherence",
            "seam_continuity", "chemistry_ratios", "quantile_monotonicity"
        ],
        "molecule_windows": {
            "h2o": {"bins": "50-80, 200-230", "n_bins": 60},
            "co2": {"bins": "120-150", "n_bins": 30},
            "ch4": {"bins": "90-120", "n_bins": 30}
        },
        "default_weights": {
            "lambda_sm": 0.1,
            "lambda_nn": 0.1,
            "lambda_coh": 0.1,
            "lambda_seam": 0.1,
            "lambda_ratio": 0.1
        }
    }


def collect_performance_data() -> Dict[str, Any]:
    """Collect performance metrics."""
    return {
        "submission_file": Path("outputs/submission.csv").exists(),
        "calibrated_files": {
            "fgs1": len(list(Path("outputs/calibrated/fgs1").glob("*.npz"))) if Path("outputs/calibrated/fgs1").exists() else 0,
            "airs": len(list(Path("outputs/calibrated/airs").glob("*.npz"))) if Path("outputs/calibrated/airs").exists() else 0
        },
        "runtime_budget": {
            "target_per_planet": "≤30s",
            "total_budget": "≤9h for ~1,100 planets"
        }
    }


def collect_file_data() -> Dict[str, Any]:
    """Collect file system diagnostics."""
    from src.spectramind.data.data_contracts import validate_directory_structure
    
    base_path = "."
    dir_validation = validate_directory_structure(base_path)
    
    return {
        "directory_structure": dir_validation,
        "output_files": list(str(p) for p in Path("outputs").rglob("*") if p.is_file()) if Path("outputs").exists() else []
    }


def generate_html_report(data: Dict[str, Any]) -> str:
    """Generate HTML report from diagnostics data."""
    
    metadata = data["metadata"]
    calibration = data["calibration"]
    symbolic = data["symbolic"]
    performance = data["performance"]
    files = data["files"]
    
    # Format calibration summary
    cal_status = []
    if calibration["temperature_scaling"]:
        temp_data = calibration["temperature_scaling"]
        if temp_data["status"] == "fitted":
            cal_status.append(f"✅ Temperature scaling: τ = {temp_data['temperature']:.4f}")
        else:
            cal_status.append("❌ Temperature scaling: error")
    else:
        cal_status.append("⏸️ Temperature scaling: not run")
        
    if calibration["corel_quantiles"]:
        corel_data = calibration["corel_quantiles"]
        if corel_data["status"] == "fitted":
            cal_status.append(f"✅ COREL: {corel_data['n_quantiles']} quantiles @ {corel_data['target_coverage']:.1%} coverage")
        else:
            cal_status.append("❌ COREL: error")
    else:
        cal_status.append("⏸️ COREL: not run")
    
    # Format calibration logs
    cal_logs_html = ""
    for log in calibration["calibration_logs"]:
        cal_logs_html += f"""
        <tr>
            <td>{log['instrument'].upper()}</td>
            <td>{log['total_steps']}</td>
            <td>{log['total_time_ms']} ms</td>
            <td><code>{log['file']}</code></td>
        </tr>"""
    
    # Format molecule windows
    mol_windows_html = ""
    for mol, info in symbolic["molecule_windows"].items():
        mol_windows_html += f"""
        <tr>
            <td>{mol.upper()}</td>
            <td>{info['bins']}</td>
            <td>{info['n_bins']}</td>
        </tr>"""
    
    # Format directory structure
    dir_status_html = ""
    for dir_path, exists in files["directory_structure"].items():
        status_icon = "✅" if exists else "❌"
        dir_status_html += f"""
        <tr>
            <td>{status_icon}</td>
            <td><code>{dir_path}</code></td>
        </tr>"""
    
    html = f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SpectraMind V50 — Diagnostics Report</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, Arial, sans-serif;
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
            line-height: 1.5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .card {{
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        code {{
            background: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9em;
        }}
        .status-list {{
            list-style: none;
            padding: 0;
        }}
        .status-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }}
        .footer {{
            margin-top: 2rem;
            padding: 1rem;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .metadata {{
            font-family: monospace;
            font-size: 0.9em;
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SpectraMind V50 — Diagnostics Report</h1>
        <p>Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge</p>
        <div class="metadata">
            Generated: {metadata['timestamp']}<br>
            Git SHA: <code>{metadata['git_sha']}</code><br>
            Config Hash: <code>{metadata['config_hash']}</code><br>
            Version: {metadata['version']}
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>🔧 Calibration Status</h2>
            <ul class="status-list">
                {"".join(f"<li>{status}</li>" for status in cal_status)}
            </ul>
            
            {f'''
            <h3>Calibration Kill Chain Logs</h3>
            <table>
                <thead>
                    <tr>
                        <th>Instrument</th>
                        <th>Steps</th>
                        <th>Time</th>
                        <th>Log File</th>
                    </tr>
                </thead>
                <tbody>
                    {cal_logs_html}
                </tbody>
            </table>
            ''' if calibration["calibration_logs"] else '<p><em>No calibration logs found.</em></p>'}
        </div>

        <div class="card">
            <h2>⚖️ Symbolic Physics Engine</h2>
            <p><strong>Status:</strong> {symbolic['engine_status']}</p>
            
            <h3>Constraint Types</h3>
            <ul>
                {"".join(f"<li>{constraint}</li>" for constraint in symbolic['constraint_types'])}
            </ul>
            
            <h3>Molecule Windows</h3>
            <table>
                <thead>
                    <tr>
                        <th>Molecule</th>
                        <th>Bin Range</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {mol_windows_html}
                </tbody>
            </table>
        </div>
    </div>

    <div class="card">
        <h2>📊 Performance & Files</h2>
        <div class="grid">
            <div>
                <h3>Output Files</h3>
                <ul class="status-list">
                    <li>{'✅' if performance['submission_file'] else '❌'} Submission CSV</li>
                    <li>📁 FGS1 calibrated: {performance['calibrated_files']['fgs1']} files</li>
                    <li>📁 AIRS calibrated: {performance['calibrated_files']['airs']} files</li>
                </ul>
                
                <h3>Runtime Budget</h3>
                <ul class="status-list">
                    <li>Per planet: {performance['runtime_budget']['target_per_planet']}</li>
                    <li>Total: {performance['runtime_budget']['total_budget']}</li>
                </ul>
            </div>
            
            <div>
                <h3>Directory Structure</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Status</th>
                            <th>Directory</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dir_status_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>🔮 Future Diagnostics</h2>
        <p>The following diagnostics components are planned for implementation:</p>
        <ul>
            <li><strong>SHAP Analysis:</strong> Feature attribution for μ predictions per spectral bin</li>
            <li><strong>UMAP/t-SNE:</strong> Latent space visualization colored by symbolic violations</li>
            <li><strong>FFT/Smoothness Maps:</strong> Per-bin second-derivative heatmaps and power spectra</li>
            <li><strong>Coverage Plots:</strong> COREL calibration performance per molecule region</li>
            <li><strong>Symbolic Dashboards:</strong> Top violated rules per planet with influence maps</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated by SpectraMind V50 diagnostics engine • <a href="https://github.com/bartytime4life/SpetraMindV50">GitHub Repository</a></p>
    </div>
</body>
</html>
"""
    
    return html
