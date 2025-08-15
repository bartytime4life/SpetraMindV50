"""Vectorised symbolic rule evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .symbolic_config_loader import load_yaml_checked
from .symbolic_utils import append_debug_log, now_utc_iso

DEBUG_LOG = "logs/v50_debug_log.md"


@dataclass
class SymbolicRule:
    name: str
    type: str
    weight: float = 1.0
    params: Optional[Dict[str, Any]] = None


@dataclass
class SymbolicEvalResult:
    rule_name: str
    magnitude: float
    mask: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


class SymbolicLogicEngine:
    """Evaluate symbolic rules on spectra."""

    def __init__(
        self,
        rules_yaml: str,
        weights_yaml: Optional[str] = None,
        soft_mode: bool = True,
    ) -> None:
        self.rules_yaml = rules_yaml
        self.weights_yaml = weights_yaml
        self.soft_mode = soft_mode
        self.rules = self._load_rules(rules_yaml, weights_yaml)
        append_debug_log(
            DEBUG_LOG,
            f"[{now_utc_iso()}] SymbolicLogicEngine: Loaded {len(self.rules)} rules.",
        )

    def _load_rules(
        self, rules_yaml: str, weights_yaml: Optional[str]
    ) -> List[SymbolicRule]:
        rules_dict = load_yaml_checked(rules_yaml)
        weights_dict = load_yaml_checked(weights_yaml) if weights_yaml else {}
        rule_list: List[SymbolicRule] = []
        for r in rules_dict.get("rules", []):
            w = float(
                weights_dict.get("weights", {}).get(
                    r.get("name", ""), r.get("weight", 1.0)
                )
            )
            rule = SymbolicRule(
                name=r.get("name", "unnamed_rule"),
                type=r.get("type", "custom"),
                weight=w,
                params=r.get("params", {}) or {},
            )
            rule_list.append(rule)
        return rule_list

    def evaluate(
        self,
        mu: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SymbolicEvalResult]:
        """Evaluate all rules on one spectrum."""
        res: List[SymbolicEvalResult] = []
        for rule in self.rules:
            if rule.type == "nonnegativity":
                r = self._r_nonnegativity(mu, rule)
            elif rule.type == "smoothness_l2":
                r = self._r_smoothness(mu, rule)
            elif rule.type == "fft_bandlimit":
                r = self._r_fft_bandlimit(mu, rule)
            elif rule.type == "molecule_window":
                r = self._r_molecule_window(mu, rule, metadata)
            else:
                r = self._r_custom(mu, rule, metadata)
            res.append(r)

        mag_sum = float(sum(x.magnitude for x in res))
        append_debug_log(
            DEBUG_LOG,
            f"[{now_utc_iso()}] SymbolicLogicEngine.evaluate: rules={len(self.rules)} total_mag={mag_sum:.6f}",
        )
        return res

    # ------------------ rule implementations ---------------------
    def _r_nonnegativity(
        self, mu: np.ndarray, rule: SymbolicRule
    ) -> SymbolicEvalResult:
        violations = -np.minimum(mu, 0.0)
        mag = float(np.sum(violations))
        if not self.soft_mode:
            mag = float(np.count_nonzero(mu < 0))
        return SymbolicEvalResult(
            rule_name=rule.name,
            magnitude=rule.weight * mag,
            mask=(mu < 0).astype(np.float32),
        )

    def _r_smoothness(self, mu: np.ndarray, rule: SymbolicRule) -> SymbolicEvalResult:
        lam = float(rule.params.get("lambda", 1.0))
        d2 = np.diff(mu, n=2)
        mag = float(lam * np.sum(d2**2))
        return SymbolicEvalResult(
            rule_name=rule.name, magnitude=rule.weight * mag, mask=None
        )

    def _r_fft_bandlimit(
        self, mu: np.ndarray, rule: SymbolicRule
    ) -> SymbolicEvalResult:
        cutoff = int(rule.params.get("cutoff_index", 32))
        spec = np.fft.rfft(mu - np.mean(mu))
        power = np.abs(spec) ** 2
        tail = power[cutoff:]
        mag = float(np.sum(tail))
        return SymbolicEvalResult(
            rule_name=rule.name, magnitude=rule.weight * mag, mask=None
        )

    def _r_molecule_window(
        self, mu: np.ndarray, rule: SymbolicRule, metadata: Optional[Dict[str, Any]]
    ) -> SymbolicEvalResult:
        wl = None if metadata is None else metadata.get("wavelengths")
        if wl is None:
            return SymbolicEvalResult(
                rule_name=rule.name,
                magnitude=0.0,
                mask=None,
                meta={"note": "no_wavelengths"},
            )
        ranges = rule.params.get("ranges", [])
        mode = rule.params.get("mode", "presence")
        mask = np.zeros_like(mu, dtype=np.float32)
        mag = 0.0
        for a, b in ranges:
            idx = (wl >= a) & (wl <= b)
            sub = mu[idx]
            if sub.size == 0:
                continue
            v = float(np.var(sub))
            if mode == "presence":
                mag += float(max(0.0, 1e-3 - v))
            else:
                m = float(np.mean(sub))
                mag += float(max(0.0, -m))
            mask[idx] = 1.0
        return SymbolicEvalResult(
            rule_name=rule.name, magnitude=rule.weight * mag, mask=mask
        )

    def _r_custom(
        self, mu: np.ndarray, rule: SymbolicRule, metadata: Optional[Dict[str, Any]]
    ) -> SymbolicEvalResult:
        return SymbolicEvalResult(
            rule_name=rule.name, magnitude=0.0, mask=None, meta={"note": "custom_noop"}
        )
