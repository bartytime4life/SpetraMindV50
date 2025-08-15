import os
import io
import json
import time
import copy
import uuid
import yaml
import math
import base64
import typing as T
import datetime as dt
from dataclasses import dataclass, field

# Logging
import logging
from logging.handlers import RotatingFileHandler

# JSON schema validation
try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False

# Plotly / Matplotlib
try:
    import plotly
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# MLflow / Weights & Biases (optional)
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from .report_data_collector import ReportDataCollector, CollectorConfig
from .export_manager import ExportManager, ExportConfig
from .charts import Charts
from .tables import Tables


# ------------------------------
# Logging Utility (Hydra-safe)
# ------------------------------

def _ensure_logging(
    log_dir: str = "logs",
    log_name: str = "v50_reporting",
    max_bytes: int = 8_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create a Hydra-safe logger with console + rotating file + JSONL writer.
    Does not call basicConfig; uses an isolated logger name to avoid interfering with Hydra/root.

    Returns:
        logging.Logger: configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"spectramind.reporting.{log_name}")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if this function is called multiple times
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers) and any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # Rotating file handler (plain text)
    fh = RotatingFileHandler(
        os.path.join(log_dir, f"{log_name}.log"), maxBytes=max_bytes, backupCount=backup_count
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(fh)

    # JSONL event stream handler (simple custom writer)
    class JsonlHandler(logging.Handler):
        def __init__(self, path: str):
            super().__init__(level=logging.INFO)
            self.path = path

        def emit(self, record: logging.LogRecord):
            try:
                payload = {
                    "ts": dt.datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception:
                pass

    logger.addHandler(JsonlHandler(os.path.join(log_dir, f"{log_name}_events.jsonl")))
    return logger


# ------------------------------
# Git / ENV snapshot
# ------------------------------

def _capture_git_info(repo_root: str = ".") -> dict:
    """
    Capture basic Git information without raising errors if git is unavailable.
    """
    import subprocess

    info = {"commit": None, "branch": None, "is_dirty": None, "remote": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
        info["commit"] = commit
    except Exception:
        pass
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).decode().strip()
        info["branch"] = branch
    except Exception:
        pass
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).decode().strip()
        info["is_dirty"] = bool(status)
    except Exception:
        pass
    try:
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=repo_root).decode().strip()
        info["remote"] = remote
    except Exception:
        pass
    return info


def _capture_env_info() -> dict:
    """
    Capture a snapshot of key environment variables and Python/runtime info.
    """
    import platform

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "WANDB_PROJECT",
                "MLFLOW_TRACKING_URI",
                "HYDRA_FULL_ERROR",
                "KAGGLE_KERNEL_RUN_TYPE",
            ]
        },
        "utc_time": dt.datetime.utcnow().isoformat() + "Z",
    }


# ------------------------------
# Configs
# ------------------------------


@dataclass
class ReportConfig:
    """
    High-level configuration for the reporting generator.
    """

    diagnostics_dir: str = "artifacts/diagnostics"
    outputs_dir: str = "reports"
    templates_dir: str = None  # if None, auto-resolve internal templates
    schema_dir: str = None  # if None, auto-resolve internal schema
    title: str = "SpectraMind V50 — Diagnostics Report"
    subtitle: str = "NeurIPS Ariel Data Challenge 2025"
    report_version: str = "v1"
    include_sections: T.List[str] = field(
        default_factory=lambda: [
            "summary",
            "metrics",
            "plots",
            "symbolic",
            "calibration",
            "umap_tsne",
            "cli_log",
            "appendix",
        ]
    )
    enable_pdf: bool = False
    open_after: bool = False
    # Optional integrations
    mlflow_log: bool = False
    wandb_log: bool = False


class ReportGenerator:
    """
    Orchestrates the collection, validation, rendering, and export of SpectraMind V50 diagnostics reports.
    """

    def __init__(
        self,
        report_cfg: ReportConfig = ReportConfig(),
        collector_cfg: CollectorConfig = CollectorConfig(),
        export_cfg: ExportConfig = ExportConfig(),
        logger: logging.Logger = None,
    ):
        self.cfg = report_cfg
        self.collector_cfg = collector_cfg
        self.export_cfg = export_cfg
        self.logger = logger or _ensure_logging()

        # Resolve internal resource directories if not provided
        pkg_root = os.path.dirname(os.path.abspath(__file__))
        self.templates_dir = self.cfg.templates_dir or os.path.join(pkg_root, "report_templates")
        self.schema_dir = self.cfg.schema_dir or os.path.join(pkg_root, "schema")

        # Initialize helpers
        self.collector = ReportDataCollector(self.collector_cfg, self.logger)
        self.exporter = ExportManager(self.export_cfg, self.logger)
        self.tables = Tables(logger=self.logger)
        self.charts = Charts(logger=self.logger)

        # Metadata snapshot
        self.metadata = {
            "report_id": str(uuid.uuid4()),
            "title": self.cfg.title,
            "subtitle": self.cfg.subtitle,
            "version": self.cfg.report_version,
            "time_utc": dt.datetime.utcnow().isoformat() + "Z",
            "git": _capture_git_info(),
            "env": _capture_env_info(),
        }

        # Ensure outputs dir
        os.makedirs(self.cfg.outputs_dir, exist_ok=True)

    # ------------------------------
    # Schema Validation
    # ------------------------------
    def _load_schema(self) -> dict:
        """
        Try to load JSON schema from schema/report_schema.json. If missing, return a minimal fallback.
        """
        schema_path = os.path.join(self.schema_dir, "report_schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Fallback minimal schema
        self.logger.warning("report_schema.json not found. Using minimal fallback schema.")
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "object"},
                "metrics": {"type": "object"},
                "plots": {"type": "object"},
                "tables": {"type": "object"},
                "artifacts": {"type": "object"},
            },
            "required": ["summary"],
            "additionalProperties": True,
        }

    def _validate(self, data: dict) -> None:
        """
        Validate data against the JSON schema if available.
        """
        if not _HAS_JSONSCHEMA:
            self.logger.warning("jsonschema not installed; skipping schema validation.")
            return
        schema = self._load_schema()
        try:
            jsonschema.validate(instance=data, schema=schema)
            self.logger.info("Report data validated against schema.")
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
            # Proceed but log; do not raise to keep robustness
            pass

    # ------------------------------
    # Template Loading
    # ------------------------------
    def _load_template(self, kind: str = "html") -> str:
        """
        Load base template content (html or md).
        """
        if kind == "html":
            path = os.path.join(self.templates_dir, "base_template.html")
        elif kind == "md":
            path = os.path.join(self.templates_dir, "base_template.md")
        else:
            raise ValueError("Unsupported template kind.")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------
    # Building Sections
    # ------------------------------
    def _build_summary_section(self, collected: dict) -> str:
        """
        Build the summary (HTML) block including key text and small metrics table.
        """
        head = f"<h2>Summary</h2>"
        bullets = []
        # Pull a few key fields if available
        diag = collected.get("diagnostic_summary", {})
        n_planets = diag.get("n_planets", "—")
        gll_mean = diag.get("gll_mean", "—")
        entropy_mean = diag.get("entropy_mean", "—")
        build = [
            f"<li>Planets analyzed: <b>{n_planets}</b></li>",
            f"<li>Mean GLL (↓ better): <b>{gll_mean}</b></li>",
            f"<li>Mean entropy: <b>{entropy_mean}</b></li>",
            f"<li>Report ID: <code>{self.metadata['report_id']}</code></li>",
            f"<li>Git: <code>{self.metadata['git']}</code></li>",
        ]
        bullets.extend(build)
        return f"{head}<ul>{''.join(bullets)}</ul>"

    def _build_tables(self, collected: dict) -> dict:
        """
        Build key HTML tables used in template replacement.
        Returns dict of table placeholders to HTML content.
        """
        out = {}

        # Metrics table
        metrics_table = self.tables.metrics_table(
            collected.get("diagnostic_summary", {}),
            caption="Core Metrics (GLL, RMSE, MAE, Entropy)",
        )
        out["{{TABLE_METRICS}}"] = metrics_table

        # Config hash x performance matrix
        cfg_hash_table = self.tables.config_hash_performance_table(
            collected.get("cli_log_summary", {}),
            caption="Config Hash × Performance Matrix",
        )
        out["{{TABLE_CONFIG_HASH}}"] = cfg_hash_table

        # Symbolic rule leaderboard
        sym_rule_table = self.tables.symbolic_rule_leaderboard_table(
            collected.get("symbolic_summary", {}),
            caption="Top Symbolic Rule Violations (per-rule aggregate)",
        )
        out["{{TABLE_SYMBOLIC}}"] = sym_rule_table

        # CLI recent calls
        cli_table = self.tables.cli_history_table(
            collected.get("cli_log_summary", {}),
            caption="Recent CLI Calls",
        )
        out["{{TABLE_CLI}}"] = cli_table
        return out

    def _build_plots(self, collected: dict) -> dict:
        """
        Build interactive charts and return mapping of placeholders to HTML (divs) or inline images.
        """
        out = {}
        # GLL Heatmap
        gll_heatmap_html = self.charts.gll_heatmap(collected.get("diagnostic_summary", {}))
        out["{{PLOT_GLL_HEATMAP}}"] = gll_heatmap_html

        # FFT Spectrum Comparison
        fft_html = self.charts.fft_power(collected.get("fft_summary", {}))
        out["{{PLOT_FFT}}"] = fft_html

        # Calibration plot
        calib_html = self.charts.calibration(collected.get("calibration_summary", {}))
        out["{{PLOT_CALIB}}"] = calib_html

        # UMAP/t-SNE
        umap_html = self.charts.umap(collected.get("projection_summary", {}))
        tsne_html = self.charts.tsne(collected.get("projection_summary", {}))
        out["{{PLOT_UMAP}}"] = umap_html
        out["{{PLOT_TSNE}}"] = tsne_html

        # Symbolic heatmap
        symbolic_html = self.charts.symbolic_heatmap(collected.get("symbolic_summary", {}))
        out["{{PLOT_SYMBOLIC}}"] = symbolic_html

        return out

    # ------------------------------
    # Render
    # ------------------------------
    def render(self) -> dict:
        """
        Collect data, validate schema, render HTML/MD, export files.
        Returns:
            dict: paths to generated artifacts.
        """
        self.logger.info("Collecting reporting inputs...")
        collected = self.collector.collect(self.cfg.diagnostics_dir)

        self.logger.info("Validating against schema...")
        # Assemble a minimal data object for schema validation
        data_for_schema = {
            "summary": {"title": self.cfg.title, "subtitle": self.cfg.subtitle},
            "metrics": collected.get("diagnostic_summary", {}),
            "plots": {
                "have_fft": bool(collected.get("fft_summary")),
                "have_calibration": bool(collected.get("calibration_summary")),
                "have_projection": bool(collected.get("projection_summary")),
            },
            "tables": {
                "have_cli": bool(collected.get("cli_log_summary")),
                "have_symbolic": bool(collected.get("symbolic_summary")),
            },
            "artifacts": {
                "diagnostics_dir": self.cfg.diagnostics_dir,
            },
        }
        self._validate(data_for_schema)

        # Build content blocks
        self.logger.info("Building tables...")
        table_map = self._build_tables(collected)

        self.logger.info("Building plots...")
        plot_map = self._build_plots(collected)

        self.logger.info("Building summary...")
        summary_html = self._build_summary_section(collected)

        # HTML template replacement
        self.logger.info("Loading HTML template...")
        html_tpl = self._load_template("html")
        replacements = {
            "{{TITLE}}": self.cfg.title,
            "{{SUBTITLE}}": self.cfg.subtitle,
            "{{REPORT_VERSION}}": self.cfg.report_version,
            "{{UTC_TIME}}": self.metadata["time_utc"],
            "{{SUMMARY}}": summary_html,
        }
        replacements.update(table_map)
        replacements.update(plot_map)
        html_out = html_tpl
        for k, v in replacements.items():
            html_out = html_out.replace(k, v if v is not None else "")

        # MD template replacement
        self.logger.info("Loading Markdown template...")
        md_tpl = self._load_template("md")
        md_replacements = {
            "{{TITLE}}": self.cfg.title,
            "{{SUBTITLE}}": self.cfg.subtitle,
            "{{REPORT_VERSION}}": self.cfg.report_version,
            "{{UTC_TIME}}": self.metadata["time_utc"],
            "{{SUMMARY_TEXT}}": f"- Report ID: {self.metadata['report_id']}\n- Git commit: {self.metadata['git'].get('commit')}\n- Time: {self.metadata['time_utc']}",
            "{{TABLE_METRICS_MD}}": self.tables.metrics_table_md(collected.get("diagnostic_summary", {})),
        }
        md_out = md_tpl
        for k, v in md_replacements.items():
            md_out = md_out.replace(k, v if v is not None else "")

        # Export artifacts
        base_name = f"diagnostic_report_{self.cfg.report_version}"
        html_path = os.path.join(self.cfg.outputs_dir, f"{base_name}.html")
        md_path = os.path.join(self.cfg.outputs_dir, f"{base_name}.md")
        self.logger.info("Exporting artifacts...")
        self.exporter.save_html(html_path, html_out)
        self.exporter.save_markdown(md_path, md_out)

        pdf_path = None
        if self.cfg.enable_pdf:
            pdf_path = os.path.join(self.cfg.outputs_dir, f"{base_name}.pdf")
            self.exporter.save_pdf(pdf_path, html_out)  # best effort; falls back if engine missing

        # Optional integrations
        if self.cfg.mlflow_log and _HAS_MLFLOW:
            try:
                mlflow.log_artifact(html_path)
                mlflow.log_artifact(md_path)
                if pdf_path and os.path.exists(pdf_path):
                    mlflow.log_artifact(pdf_path)
            except Exception as e:
                self.logger.warning(f"MLflow log failed: {e}")

        if self.cfg.wandb_log and _HAS_WANDB:
            try:
                wandb.log({"report_html": wandb.Html(open(html_path, "r", encoding="utf-8").read(), inject=False)})
                wandb.save(html_path, base_path=self.cfg.outputs_dir)
                wandb.save(md_path, base_path=self.cfg.outputs_dir)
                if pdf_path and os.path.exists(pdf_path):
                    wandb.save(pdf_path, base_path=self.cfg.outputs_dir)
            except Exception as e:
                self.logger.warning(f"W&B log failed: {e}")

        # Optionally open in browser
        if self.cfg.open_after:
            try:
                import webbrowser

                webbrowser.open(f"file://{os.path.abspath(html_path)}")
            except Exception:
                pass

        # Write a small manifest
        manifest = {
            "report_id": self.metadata["report_id"],
            "version": self.cfg.report_version,
            "generated": self.metadata["time_utc"],
            "artifacts": {"html": html_path, "md": md_path, "pdf": pdf_path},
            "git": self.metadata["git"],
            "env": self.metadata["env"],
        }
        manifest_path = os.path.join(self.cfg.outputs_dir, f"{base_name}.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Report generated: {html_path}")
        return {"html": html_path, "md": md_path, "pdf": pdf_path, "manifest": manifest_path}
