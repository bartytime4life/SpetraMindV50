# SPDX-License-Identifier: Apache-2.0

"""Core abstract base for all symbolic rules used in SpectraMind V50."""

from __future__ import annotations

import abc
import platform
import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .utils import (
    ensure_device,
    get_git_revision,
    get_logger,
    jsonl_event,
    now_utc_iso,
    summarize_tensor,
    to_tensor,
)


@dataclass
class RuleOutput:
    """Structured output of a symbolic rule evaluation."""

    violation_map: Tensor
    loss: Tensor
    aux: Dict[str, Any]
    name: str
    weight: float


class SymbolicRule(abc.ABC, torch.nn.Module):
    """Abstract base class for physics/chemistry/logic-informed symbolic rules."""

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        description: Optional[str] = None,
        enable_logging: bool = True,
        device: Optional[str] = None,
        log_jsonl_path: Optional[str] = "logs/symbolic_events.jsonl",
        log_text_path: Optional[str] = "logs/spectramind_symbolic_rules.log",
    ) -> None:
        super().__init__()
        self.name = name
        self.weight = float(weight)
        self.description = description or name
        self.enable_logging = enable_logging
        self.device = ensure_device(device)
        self.log = get_logger("spectramind.symbolic.rules", log_text_path)
        self.log_jsonl_path = log_jsonl_path
        # Environment snapshot
        self._env = {
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "git": get_git_revision(),
        }

    # --------- Abstract API ---------
    @abc.abstractmethod
    def evaluate_map(
        self,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        """Compute the nonnegative violation magnitudes per bin or sample."""

    def reduce_loss(
        self,
        violation_map: Tensor,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> Tensor:
        """Default reduction: weight * mean of violation_map."""

        return self.weight * violation_map.mean()

    # --------- Public API ---------
    @torch.no_grad()
    def log_eval(
        self,
        violation_map: Tensor,
        loss: Tensor,
        mu: Tensor,
        sigma: Optional[Tensor],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """JSONL + text logging. Avoid heavy tensors in JSON."""

        if not self.enable_logging:
            return

        try:
            payload = {
                "ts": now_utc_iso(),
                "event": "rule_evaluated",
                "rule": self.name,
                "weight": self.weight,
                "loss": float(loss.detach().cpu().item()),
                "violation_map_summary": summarize_tensor(violation_map),
                "mu_summary": summarize_tensor(mu, name="mu"),
                "sigma_summary": (
                    summarize_tensor(sigma, name="sigma") if sigma is not None else None
                ),
                "metadata_keys": (
                    sorted(list(metadata.keys()))
                    if isinstance(metadata, dict)
                    else None
                ),
                "env": self._env,
            }
            jsonl_event(self.log_jsonl_path, payload)
            self.log.info(
                "[%s] loss=%.6f shape=%s",
                self.name,
                payload["loss"],
                tuple(violation_map.shape),
            )
        except Exception as exc:  # pragma: no cover - logging should not break
            self.log.warning("Logging failed for rule %s: %s", self.name, exc)

    def forward(
        self,
        mu: Tensor,
        sigma: Optional[Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RuleOutput:
        """Standard torch entrypoint. Returns ``RuleOutput``."""

        mu = to_tensor(mu, device=self.device)
        sigma = to_tensor(sigma, device=self.device) if sigma is not None else None

        violation_map = self.evaluate_map(mu, sigma, metadata)
        loss = self.reduce_loss(violation_map, mu, sigma, metadata)

        aux = {
            "violation_mean": float(violation_map.mean().detach().cpu().item()),
            "violation_max": float(violation_map.max().detach().cpu().item()),
        }

        with torch.no_grad():
            self.log_eval(violation_map, loss, mu, sigma, metadata)

        return RuleOutput(
            violation_map=violation_map,
            loss=loss,
            aux=aux,
            name=self.name,
            weight=self.weight,
        )


__all__ = ["SymbolicRule", "RuleOutput"]
