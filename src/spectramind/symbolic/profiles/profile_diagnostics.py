"""Diagnostics utilities for symbolic profiles."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .logging_utils import setup_logging


def _extract_records(obj: Any, key_hint: Optional[str]) -> Iterable[Dict[str, Any]]:
    """Try to extract a flat list of violation dicts from various shapes."""
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(obj, dict):
        if key_hint and key_hint in obj and isinstance(obj[key_hint], list):
            for item in obj[key_hint]:
                if isinstance(item, dict):
                    yield item
            return
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                for item in v:
                    yield item
                return
    # Fallback: nothing


def load_violations_json(path: Path, key_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(_extract_records(data, key_hint))


def aggregate_violations(
    records: Iterable[Dict[str, Any]]
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Dict[str, float]]]:
    """Aggregate per-planet, per-rule counts, sums, and means."""
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    planets_set, rules_set = set(), set()

    for rec in records:
        planet = str(rec.get("planet_id", "unknown"))
        rule = str(rec.get("rule_id", "unknown"))
        val = float(rec.get("violation", 0.0))
        planets_set.add(planet)
        rules_set.add(rule)
        key = (planet, rule)
        counts[key] += 1
        sums[key] += val

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, cnt in counts.items():
        s = sums[key]
        out[key] = {
            "count": float(cnt),
            "sum": float(s),
            "mean": float(s / cnt) if cnt else 0.0,
        }
    return sorted(planets_set), sorted(rules_set), out


def write_heatmap_csv(
    out_path: Path,
    planets: List[str],
    rules: List[str],
    table: Dict[Tuple[str, str], Dict[str, float]],
    metric: str = "mean",
) -> None:
    """Write a simple CSV heatmap with planets as rows, rules as columns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("planet_id," + ",".join(rules) + "\n")
        for p in planets:
            row = [p]
            for r in rules:
                val = table.get((p, r), {}).get(metric, 0.0)
                row.append(f"{val:.6f}")
            f.write(",".join(row) + "\n")


def write_summary_json(
    out_path: Path,
    planets: List[str],
    rules: List[str],
    table: Dict[Tuple[str, str], Dict[str, float]],
) -> None:
    """Write machine-friendly JSON with the aggregated metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serial = {
        "planets": planets,
        "rules": rules,
        "table": {f"{p}|{r}": m for (p, r), m in table.items()},
        "metrics": ["count", "sum", "mean"],
    }
    out_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def generate_heatmap(
    violations_json: Path,
    out_csv: Path,
    out_json: Path,
    key_hint: Optional[str] = None,
    metric: str = "mean",
    logger_name: str = "profiles.diagnostics",
) -> None:
    """End-to-end heatmap generation."""
    logger = setup_logging(logger_name)
    recs = load_violations_json(violations_json, key_hint=key_hint)
    planets, rules, table = aggregate_violations(recs)
    write_heatmap_csv(out_csv, planets, rules, table, metric=metric)
    write_summary_json(out_json, planets, rules, table)
    logger.info(f"Wrote heatmap CSV: {out_csv}")
    logger.info(f"Wrote summary JSON: {out_json}")
