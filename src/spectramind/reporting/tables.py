import html
import json
import typing as T
import logging


class Tables:
    """Table generators (HTML & Markdown)."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    # -----------------
    # HTML Tables
    # -----------------
    def metrics_table(self, diag: dict, caption: str = None) -> str:
        """Build a metrics summary table from diagnostic_summary.
        Keys used (best-effort): gll_mean, rmse_mean, mae_mean, entropy_mean"""
        rows = [
            ("GLL (mean, ↓)", diag.get("gll_mean", "—")),
            ("RMSE (mean, ↓)", diag.get("rmse_mean", "—")),
            ("MAE (mean, ↓)", diag.get("mae_mean", "—")),
            ("Entropy (mean)", diag.get("entropy_mean", "—")),
            ("Planets", diag.get("n_planets", "—")),
        ]
        cap = f"<div class='muted' style='margin-bottom:6px'>{html.escape(caption)}</div>" if caption else ""
        parts = [cap, "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"]
        for k, v in rows:
            parts.append(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    def config_hash_performance_table(self, cli_log_summary: dict, caption: str = None) -> str:
        """Build a config hash × performance table, best-effort from CLI log summary.
        Expects {"entries": [{"hash": "...", "gll": ..., "rmse": ..., "time": "..."}]}"""
        entries = cli_log_summary.get("entries", [])
        cap = f"<div class='muted' style='margin-bottom:6px'>{html.escape(caption)}</div>" if caption else ""
        header = (
            "<table><thead><tr><th>Config Hash</th><th>GLL</th><th>RMSE</th><th>Ran At (UTC)</th>"
            "</tr></thead><tbody>"
        )
        parts = [cap, header]
        for e in entries:
            parts.append(
                f"<tr><td><code>{html.escape(str(e.get('hash','—')))}</code></td>"
                f"<td>{html.escape(str(e.get('gll','—')))}</td>"
                f"<td>{html.escape(str(e.get('rmse','—')))}</td>"
                f"<td class='muted'>{html.escape(str(e.get('time','—')))}</td></tr>"
            )
        parts.append("</tbody></table>")
        return "".join(parts)

    def symbolic_rule_leaderboard_table(self, symbolic_summary: dict, caption: str = None) -> str:
        """Table of top symbolic rule violations.
        Expects {"top_rules": [{"rule":"...", "score":..., "coverage":...}, ...]}"""
        items = symbolic_summary.get("top_rules", [])
        cap = f"<div class='muted' style='margin-bottom:6px'>{html.escape(caption)}</div>" if caption else ""
        header = (
            "<table><thead><tr><th>Rule</th><th>Violation Score</th><th>Coverage</th>"
            "</tr></thead><tbody>"
        )
        parts = [cap, header]
        for r in items:
            parts.append(
                f"<tr><td>{html.escape(str(r.get('rule','—')))}</td>"
                f"<td>{html.escape(str(r.get('score','—')))}</td>"
                f"<td>{html.escape(str(r.get('coverage','—')))}</td></tr>"
            )
        parts.append("</tbody></table>")
        return "".join(parts)

    def cli_history_table(self, cli_log_summary: dict, caption: str = None) -> str:
        """Recent CLI calls table.
        Expects {"history":[{"cmd":"...", "status":"OK/FAIL", "ts":"...","duration_s":...}]}"""
        hist = cli_log_summary.get("history", [])
        cap = f"<div class='muted' style='margin-bottom:6px'>{html.escape(caption)}</div>" if caption else ""
        header = (
            "<table><thead><tr><th>Command</th><th>Status</th><th>Time (UTC)</th><th>Duration (s)</th>"
            "</tr></thead><tbody>"
        )
        parts = [cap, header]
        for h in hist:
            parts.append(
                f"<tr><td><code>{html.escape(str(h.get('cmd','—')))}</code></td>"
                f"<td>{html.escape(str(h.get('status','—')))}</td>"
                f"<td class='muted'>{html.escape(str(h.get('ts','—')))}</td>"
                f"<td>{html.escape(str(h.get('duration_s','—')))}</td></tr>"
            )
        parts.append("</tbody></table>")
        return "".join(parts)

    # -----------------
    # Markdown Tables
    # -----------------
    def metrics_table_md(self, diag: dict) -> str:
        """Markdown table for key metrics."""
        rows = [
            ("GLL (mean, ↓)", diag.get("gll_mean", "—")),
            ("RMSE (mean, ↓)", diag.get("rmse_mean", "—")),
            ("MAE (mean, ↓)", diag.get("mae_mean", "—")),
            ("Entropy (mean)", diag.get("entropy_mean", "—")),
            ("Planets", diag.get("n_planets", "—")),
        ]
        out = ["| Metric | Value |", "|---|---|"]
        for k, v in rows:
            out.append(f"| {k} | {v} |")
        return "\n".join(out)
