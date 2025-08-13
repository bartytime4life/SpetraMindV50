Here you go — a single drop‑in file you can paste as-is.

# configs/model/src/spectramind/models/fusion.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Fusion Module (single-file version)

This module unifies all fusion strategies and a Hydra-friendly factory:
  - ConcatMLPFusion
  - CrossAttentionFusion (bi-directional pooled cross-attention)
  - GatedFusion (learned vector gate)
  - ResidualSumFusion (alpha/beta blend)
  - AdapterFusion (bottleneck adapters + tiny MLP)
  - MoEFusion (mixture-of-experts over concat)
  - LateBlendFusion (fixed/learned/cosine gamma blend)
  - IdentityFusion (debug passthrough)

Factory:
  create_fusion(cfg: Dict[str, Any]) -> FusionBase
    - Accepts aliases like: "concat+mlp" == "concat_mlp", "cross-attend" == "cross_attend"
    - Reads common settings from either cfg["fusion"] or cfg["model"]["fusion"]

Diagnostics (opt-in via config):
  export:
    taps: true/false           # intermediate tensors (e.g., projections, contexts)
    attn_weights: true/false   # attention weights (CrossAttention)
    gate_values: true/false    # gate distributions/scalars

Shapes:
  - Inputs are pooled latents: h_fgs1: [B, Df], h_airs: [B, Da]
  - Output fused: [B, D]
TorchScript:
  - Uses plain torch/nn ops; dict in return is optional (keep dict usage outside scripted path if needed)

MIT-style license (internal project use).
"""
from typing import Any, Dict, Optional, Tuple, TypedDict, List

import math
import torch
import torch.nn as nn


# ----------------------------- Common base & utils -----------------------------


class FusionExtras(TypedDict, total=False):
    """Optional diagnostics payload emitted by fusion modules when enabled."""
    attn_weights: torch.Tensor      # [B, H, 1, 1] for pooled attention blocks
    gate_values: torch.Tensor       # [B, D] or [B, 1]
    taps: Dict[str, torch.Tensor]   # arbitrary intermediate tensors


def _make_norm(kind: str, dim: int) -> nn.Module:
    kind = (kind or "layernorm").lower()
    if kind == "layernorm":
        return nn.LayerNorm(dim)

    if kind == "rms":
        # Lightweight RMSNorm (weight-only), TorchScript-friendly.
        class RMSNorm(nn.Module):
            def __init__(self, d: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(d))

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # [*, D]
                var = x.pow(2).mean(dim=-1, keepdim=True)
                x = x * torch.rsqrt(var + self.eps)
                return x * self.weight

        return RMSNorm(dim)

    if kind == "batch":
        # Expecting [B, D] (flattened), BatchNorm1d applies over D
        return nn.BatchNorm1d(dim)

    return nn.Identity()


class FusionBase(nn.Module):
    """Base class for all fusion modules with common config/shape checks and exports."""

    def __init__(
        self,
        *,
        dim: int = 256,
        dropout: float = 0.0,
        norm: str = "layernorm",
        jit_safe: bool = True,
        export_taps: bool = False,
        export_attn: bool = False,
        export_gate: bool = False,
        d_fgs1: int = 256,
        d_airs: int = 256,
        strict_check: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.norm = _make_norm(norm, self.dim)
        self.jit_safe = bool(jit_safe)
        self.export_taps = bool(export_taps)
        self.export_attn = bool(export_attn)
        self.export_gate = bool(export_gate)
        self.d_fgs1 = int(d_fgs1)
        self.d_airs = int(d_airs)
        self.strict_check = bool(strict_check)

    def _assert_shapes(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor) -> None:
        """Enforce pooled-latent shapes if strict_check is enabled: [B, Df] and [B, Da]."""
        if not self.strict_check:
            return
        if h_fgs1.ndim != 2:
            raise ValueError(f"FGS1 latent must be [B, Df], got {tuple(h_fgs1.shape)}")
        if h_airs.ndim != 2:
            raise ValueError(f"AIRS latent must be [B, Da], got {tuple(h_airs.shape)}")
        if h_fgs1.shape[1] != self.d_fgs1:
            raise ValueError(f"FGS1 latent D mismatch: expected {self.d_fgs1}, got {h_fgs1.shape[1]}")
        if h_airs.shape[1] != self.d_airs:
            raise ValueError(f"AIRS latent D mismatch: expected {self.d_airs}, got {h_airs.shape[1]}")

    def forward(  # type: ignore[override]
        self,
        h_fgs1: torch.Tensor,
        h_airs: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, FusionExtras]:
        raise NotImplementedError("FusionBase.forward must be implemented by subclasses")


# -------------------------------- Concat + MLP ---------------------------------


class _MLP(nn.Module):
    """Simple configurable MLP head (Linear+Act+Dropout stacks)."""

    def __init__(
        self,
        in_dim: int,
        hidden: Optional[List[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        hidden = hidden or [512, 256]
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}
        act = acts.get(activation, nn.GELU())
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h, bias=bias), act]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim, bias=bias)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConcatMLPFusion(FusionBase):
    """Concatenate pooled latents [h_fgs1 ; h_airs] then MLP → norm → dropout."""

    def __init__(self, *, mlp_cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        mlp_cfg = mlp_cfg or {}
        hidden = mlp_cfg.get("hidden", [512, 256])
        activation = mlp_cfg.get("activation", "gelu")
        mlp_dropout = float(mlp_cfg.get("dropout", 0.05))
        bias = bool(mlp_cfg.get("bias", True))

        self.proj = _MLP(
            in_dim=self.d_fgs1 + self.d_airs,
            hidden=hidden,
            activation=activation,
            dropout=mlp_dropout,
            bias=bias,
            out_dim=self.dim,
        )

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        x = torch.cat([h_fgs1, h_airs], dim=-1)
        fused = self.proj(x)
        fused = self.norm(fused)
        fused = self.drop(fused)

        extras: FusionExtras = {}
        if self.export_taps:
            extras["taps"] = {"concat_input": x.detach()}
        return fused, extras


# ------------------------------ Cross-Attention --------------------------------


class _SelfCrossBlock(nn.Module):
    """
    Lightweight cross-attention block on pooled tokens:
      q_in: [B, Dq], kv_in: [B, Dkv]  →  out: [B, Dq], attn: [B, H, 1, 1]
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_out: int,
        heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        resid_drop: float = 0.0,
        norm_kind: str = "layernorm",
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = int(max(1, heads))
        self.scale = (dim_q // self.heads) ** -0.5
        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.k = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.proj = nn.Linear(dim_q, dim_out, bias=True)
        self.norm_q = _make_norm(norm_kind, dim_q)
        self.norm_kv = _make_norm(norm_kind, dim_kv)
        self.drop_attn = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.drop_resid = nn.Dropout(resid_drop) if resid_drop > 0 else nn.Identity()
        self.register_buffer("attn_bias", torch.tensor(bias_init, dtype=torch.float32), persistent=False)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = q_in.shape[0]
        q = self.norm_q(q_in)
        kv = self.norm_kv(kv_in)

        q = self.q(q)   # [B, D]
        k = self.k(kv)  # [B, D]
        v = self.v(kv)  # [B, D]

        H = self.heads
        D = q.shape[-1]
        q = q.view(B, H, 1, D // H)
        k = k.view(B, H, 1, D // H)
        v = v.view(B, H, 1, D // H)

        attn_logits = (q * self.scale) @ k.transpose(-2, -1)  # [B, H, 1, 1]
        if self.attn_bias is not None:
            attn_logits = attn_logits + self.attn_bias
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.drop_attn(attn)

        out = attn @ v                     # [B, H, 1, D/H]
        out = out.contiguous().view(B, D)  # [B, D]
        out = self.proj(out)               # [B, dim_out]
        out = self.drop_resid(out)
        return out, attn


class CrossAttentionFusion(FusionBase):
    """
    Bi-directional cross-attention on pooled latents:
      - FGS1 attends to AIRS
      - AIRS attends to FGS1
    Optional physics-informed scalar conditioning via tiny linear adapters.
    """

    def __init__(
        self,
        *,
        attn_cfg: Optional[Dict[str, Any]] = None,
        pool_cfg: Optional[Dict[str, Any]] = None,  # reserved for future sequence support
        symbolic_injection: Optional[Dict[str, Any]] = None,
        attention_bias_init: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        attn_cfg = attn_cfg or {}
        self.layers = int(attn_cfg.get("layers", 2))
        heads = int(attn_cfg.get("heads", 4))
        attn_dropout = float(attn_cfg.get("dropout", 0.05))
        resid_dropout = float(attn_cfg.get("resid_dropout", 0.05))
        qkv_bias = bool(attn_cfg.get("qkv_bias", True))
        norm_kind = str(attn_cfg.get("norm", "layernorm"))

        inj = symbolic_injection or {}
        self.inject_molecule = bool(inj.get("include_molecule_masks", False))
        self.inject_seams = bool(inj.get("include_detector_seams", False))
        self.inject_wavepos = bool(inj.get("include_wavelengths", False))
        self.inject_snr = bool(inj.get("include_snr_weights", False))

        inj_dim = 0
        if self.inject_molecule:
            inj_dim += 1
        if self.inject_seams:
            inj_dim += 1
        if self.inject_wavepos:
            inj_dim += 1
        if self.inject_snr:
            inj_dim += 1

        if inj_dim > 0:
            self.cond_fgs1 = nn.Linear(inj_dim, self.d_fgs1, bias=True)
            self.cond_airs = nn.Linear(inj_dim, self.d_airs, bias=True)
        else:
            self.cond_fgs1 = nn.Identity()
            self.cond_airs = nn.Identity()

        self.fgs_from_airs = nn.ModuleList(
            [
                _SelfCrossBlock(
                    dim_q=self.d_fgs1,
                    dim_kv=self.d_airs,
                    dim_out=self.d_fgs1,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_dropout,
                    resid_drop=resid_dropout,
                    norm_kind=norm_kind,
                    bias_init=float(attention_bias_init),
                )
                for _ in range(self.layers)
            ]
        )
        self.airs_from_fgs = nn.ModuleList(
            [
                _SelfCrossBlock(
                    dim_q=self.d_airs,
                    dim_kv=self.d_fgs1,
                    dim_out=self.d_airs,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_dropout,
                    resid_drop=resid_dropout,
                    norm_kind=norm_kind,
                    bias_init=float(attention_bias_init),
                )
                for _ in range(self.layers)
            ]
        )
        self.out_proj = nn.Linear(self.d_fgs1 + self.d_airs, self.dim, bias=True)

    def _make_injection(self, inj_inputs: Dict[str, torch.Tensor], B: int) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        if self.inject_molecule and ("molecule" in inj_inputs):
            feats.append(inj_inputs["molecule"].view(B, 1).to(dtype=torch.float32))
        if self.inject_seams and ("seam" in inj_inputs):
            feats.append(inj_inputs["seam"].view(B, 1).to(dtype=torch.float32))
        if self.inject_wavepos and ("wavepos" in inj_inputs):
            feats.append(inj_inputs["wavepos"].view(B, 1).to(dtype=torch.float32))
        if self.inject_snr and ("snr" in inj_inputs):
            feats.append(inj_inputs["snr"].view(B, 1).to(dtype=torch.float32))
        if len(feats) == 0:
            device = inj_inputs.get("device", None)
            return torch.zeros((B, 0), dtype=torch.float32, device=device)
        return torch.cat(feats, dim=-1)

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        B = h_fgs1.shape[0]

        cond_vec = self._make_injection(kwargs, B)
        if cond_vec.shape[-1] > 0:
            h_fgs1 = h_fgs1 + self.cond_fgs1(cond_vec)
            h_airs = h_airs + self.cond_airs(cond_vec)

        last_attn: Optional[torch.Tensor] = None
        q_fgs = h_fgs1
        q_airs = h_airs
        for layer_fa, layer_af in zip(self.fgs_from_airs, self.airs_from_fgs):
            upd_fgs, attn_fa = layer_fa(q_fgs, q_airs)
            q_fgs = q_fgs + upd_fgs
            upd_airs, attn_af = layer_af(q_airs, q_fgs)
            q_airs = q_airs + upd_airs
            last_attn = attn_af  # export last

        fused = torch.cat([q_fgs, q_airs], dim=-1)
        fused = self.out_proj(fused)
        fused = self.norm(fused)
        fused = self.drop(fused)

        extras: FusionExtras = {}
        if self.export_attn and last_attn is not None:
            extras["attn_weights"] = last_attn.detach()
        if self.export_taps:
            extras["taps"] = {"fgs_ctx": q_fgs.detach(), "airs_ctx": q_airs.detach()}
        return fused, extras


# ----------------------------------- Gated ------------------------------------


class _Gate(nn.Module):
    """Vector gate producing values in [0,1] via Sigmoid."""

    def __init__(self, src_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(src_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(FusionBase):
    """
    Learn gate g to blend projections:
      h = g * P_airs(h_airs) + (1 - g) * P_fgs(h_fgs1)
    """

    def __init__(self, *, gate_cfg: Optional[Dict[str, Any]] = None, proj_cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        gate_cfg = gate_cfg or {}
        proj_cfg = proj_cfg or {}

        gate_from = str(gate_cfg.get("from", "fgs1")).lower()  # "fgs1" | "airs" | "both"
        gate_hidden = int(gate_cfg.get("hidden", 256))
        gate_dropout = float(gate_cfg.get("dropout", 0.05))

        self.p_fgs = nn.Linear(self.d_fgs1, self.dim, bias=bool(proj_cfg.get("bias", True)))
        self.p_airs = nn.Linear(self.d_airs, self.dim, bias=bool(proj_cfg.get("bias", True)))

        if gate_from == "both":
            src_dim = self.d_fgs1 + self.d_airs
        elif gate_from == "airs":
            src_dim = self.d_airs
        else:
            src_dim = self.d_fgs1
        self.gate_from = gate_from

        self.gate = _Gate(src_dim=src_dim, out_dim=self.dim, hidden=gate_hidden, dropout=gate_dropout)

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        pf = self.p_fgs(h_fgs1)
        pa = self.p_airs(h_airs)

        if self.gate_from == "both":
            src = torch.cat([h_fgs1, h_airs], dim=-1)
        elif self.gate_from == "airs":
            src = h_airs
        else:
            src = h_fgs1

        g = self.gate(src)  # [B, D] in [0,1]
        fused = g * pa + (1.0 - g) * pf
        fused = self.norm(fused)
        fused = self.drop(fused)

        extras: FusionExtras = {}
        if self.export_gate:
            extras["gate_values"] = g.detach()
        if self.export_taps:
            extras.setdefault("taps", {})
            extras["taps"]["p_fgs"] = pf.detach()
            extras["taps"]["p_airs"] = pa.detach()
        return fused, extras


# --------------------------------- Residual -----------------------------------


class ResidualSumFusion(FusionBase):
    """Ultra-light fusion: fused = alpha * P_fgs + beta * P_airs (scalar or per-feature)."""

    def __init__(self, *, proj_cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        proj_cfg = proj_cfg or {}

        self.p_fgs = nn.Linear(self.d_fgs1, self.dim, bias=bool(proj_cfg.get("bias", True)))
        self.p_airs = nn.Linear(self.d_airs, self.dim, bias=bool(proj_cfg.get("bias", True)))

        per_feature = bool(proj_cfg.get("per_feature", False))
        if per_feature:
            self.alpha = nn.Parameter(torch.full((self.dim,), float(proj_cfg.get("init_alpha", 0.5))))
            self.beta = nn.Parameter(torch.full((self.dim,), float(proj_cfg.get("init_beta", 0.5))))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(proj_cfg.get("init_alpha", 0.5))))
            self.beta = nn.Parameter(torch.tensor(float(proj_cfg.get("init_beta", 0.5))))
        self.per_feature = per_feature

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        pf = self.p_fgs(h_fgs1)
        pa = self.p_airs(h_airs)
        fused = self.alpha * pf + self.beta * pa
        fused = self.norm(fused)
        fused = self.drop(fused)
        return fused, {}


# ---------------------------------- Adapter -----------------------------------


class _Adapter(nn.Module):
    """LN → Linear(bottleneck) → Act → Drop → Linear → Residual."""

    def __init__(
        self,
        dim: int,
        bottleneck: int = 64,
        activation: str = "relu",
        dropout: float = 0.05,
        bias: bool = True,
    ) -> None:
        super().__init__()
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}
        act = acts.get(activation, nn.ReLU())
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck, bias=bias)
        self.act = act
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(bottleneck, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y


class AdapterFusion(FusionBase):
    """Adapter bottlenecks on each encoder, then concat + tiny MLP to fused dim."""

    def __init__(self, *, adapter_cfg: Optional[Dict[str, Any]] = None, mlp_cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        adapter_cfg = adapter_cfg or {}
        bottleneck = int(adapter_cfg.get("bottleneck", 64))
        activation = str(adapter_cfg.get("activation", "relu"))
        adp_dropout = float(adapter_cfg.get("dropout", 0.05))
        bias = bool(adapter_cfg.get("bias", True))

        self.adp_fgs = _Adapter(self.d_fgs1, bottleneck=bottleneck, activation=activation, dropout=adp_dropout, bias=bias)
        self.adp_airs = _Adapter(self.d_airs, bottleneck=bottleneck, activation=activation, dropout=adp_dropout, bias=bias)

        hidden = (mlp_cfg or {}).get("hidden", [256])
        self.mlp_in = nn.Linear(self.d_fgs1 + self.d_airs, self.dim if not hidden else hidden[0], bias=True)
        self.mlp_out: Optional[nn.Sequential]
        if hidden:
            self.mlp_out = nn.Sequential(
                nn.GELU(),
                nn.Dropout(float((mlp_cfg or {}).get("dropout", 0.05))) if float((mlp_cfg or {}).get("dropout", 0.05)) > 0 else nn.Identity(),
                nn.Linear(hidden[0], self.dim, bias=True),
            )
        else:
            self.mlp_out = None

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        a_f = self.adp_fgs(h_fgs1)
        a_a = self.adp_airs(h_airs)
        cat = torch.cat([a_f, a_a], dim=-1)
        x = self.mlp_in(cat)
        if self.mlp_out is not None:
            x = self.mlp_out(x)
        fused = self.norm(x)
        fused = self.drop(fused)
        extras: FusionExtras = {}
        if self.export_taps:
            extras["taps"] = {"adp_fgs": a_f.detach(), "adp_airs": a_a.detach()}
        return fused, extras


# ------------------------------------ MoE -------------------------------------


class _Expert(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, activation: str = "silu", dropout: float = 0.0) -> None:
        super().__init__()
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}
        act = acts.get(activation, nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            act,
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Gater(nn.Module):
    def __init__(self, in_dim: int, num_experts: int, hidden: int, dropout: float = 0.0, kind: str = "softmax_mlp") -> None:
        super().__init__()
        self.kind = kind
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden, num_experts, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.mlp(x)
        return torch.softmax(g, dim=-1)


class MoEFusion(FusionBase):
    """Mixture-of-experts over concatenated encoders with softmax gating."""

    def __init__(self, *, moe_cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        moe_cfg = moe_cfg or {}
        self.num_experts = int(moe_cfg.get("num_experts", 4))
        hidden = int(moe_cfg.get("expert_hidden", 256))
        activation = str(moe_cfg.get("activation", "silu"))
        edrop = float(moe_cfg.get("dropout", 0.05))

        gating = moe_cfg.get("gating", {}) or {}
        gsrc = str(gating.get("source", "concat"))
        ghidden = int(gating.get("hidden", 128))
        gdrop = float(gating.get("dropout", 0.05))

        if gsrc == "fgs1":
            gdim = self.d_fgs1
        elif gsrc == "airs":
            gdim = self.d_airs
        else:
            gdim = self.d_fgs1 + self.d_airs

        self.experts = nn.ModuleList(
            [
                _Expert(in_dim=self.d_fgs1 + self.d_airs, hidden=hidden, out_dim=self.dim, activation=activation, dropout=edrop)
                for _ in range(self.num_experts)
            ]
        )
        self.gater = _Gater(in_dim=gdim, num_experts=self.num_experts, hidden=ghidden, dropout=gdrop, kind="softmax_mlp")

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        cat = torch.cat([h_fgs1, h_airs], dim=-1)

        # Determine gating source by inferred input dimension to gater
        g_in_features = self.gater.mlp[0].in_features
        if g_in_features == self.d_fgs1:
            gsrc = h_fgs1
        elif g_in_features == self.d_airs:
            gsrc = h_airs
        else:
            gsrc = cat

        weights = self.gater(gsrc)  # [B, E]

        outs = [expert(cat) for expert in self.experts]  # list([B, D]) for E experts
        stack = torch.stack(outs, dim=1)                 # [B, E, D]
        weights = weights.unsqueeze(-1)                  # [B, E, 1]
        fused = (weights * stack).sum(dim=1)             # [B, D]

        fused = self.norm(fused)
        fused = self.drop(fused)

        extras: FusionExtras = {}
        if self.export_gate:
            extras["gate_values"] = weights.squeeze(-1).detach()
        if self.export_taps:
            extras["taps"] = {"concat_input": cat.detach()}
        return fused, extras


# --------------------------------- Late Blend ---------------------------------


class LateBlendFusion(FusionBase):
    """
    Late-blend that learns/fixes gamma in [0,1] to blend projected encoders.
    Strategies: "learned" (default), "fixed", "cosine_schedule".
    """

    def __init__(self, *, late_cfg: Optional[Dict[str, Any]] = None, proj_cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        late_cfg = late_cfg or {}
        proj_cfg = proj_cfg or {}

        self.strategy = str(late_cfg.get("strategy", "learned"))
        self.fixed_gamma = float(late_cfg.get("fixed_gamma", 0.5))
        cos = late_cfg.get("cosine", {}) or {}
        self.cos_min = float(cos.get("min_gamma", 0.3))
        self.cos_max = float(cos.get("max_gamma", 0.7))
        self.cos_steps = int(cos.get("total_steps", 10_000))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=False)

        self.p_fgs = nn.Linear(self.d_fgs1, self.dim, bias=bool(proj_cfg.get("bias", True)))
        self.p_airs = nn.Linear(self.d_airs, self.dim, bias=bool(proj_cfg.get("bias", True)))

        if self.strategy == "learned":
            self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.gamma = None

    def _cosine_gamma(self, step: int) -> float:
        if self.cos_steps <= 0:
            return self.cos_max
        t = min(max(step, 0), self.cos_steps)
        w = 0.5 * (1 - math.cos(math.pi * t / self.cos_steps))
        return float(self.cos_min * (1 - w) + self.cos_max * w)

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        pf = self.p_fgs(h_fgs1)
        pa = self.p_airs(h_airs)

        if self.strategy == "fixed":
            gamma = self.fixed_gamma
        elif self.strategy == "cosine_schedule":
            step = int(self._step.item())
            gamma = self._cosine_gamma(step)
            self._step = torch.tensor(step + 1, dtype=torch.long, device=self._step.device)
        else:  # learned
            gamma = float(self.gamma.clamp(0.0, 1.0).item()) if self.gamma is not None else 0.5

        fused = gamma * pa + (1.0 - gamma) * pf
        fused = self.norm(fused)
        fused = self.drop(fused)

        extras: FusionExtras = {}
        if self.export_gate:
            extras["gate_values"] = torch.full((h_fgs1.shape[0], 1), gamma, dtype=torch.float32, device=h_fgs1.device)
        if self.export_taps:
            extras["taps"] = {"p_fgs": pf.detach(), "p_airs": pa.detach()}
        return fused, extras


# ---------------------------------- Identity ----------------------------------


class IdentityFusion(FusionBase):
    """Debug-only fusion that forwards one encoder latent as the fused vector."""

    def __init__(self, *, which: str = "fgs1", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        which = (which or "fgs1").lower()
        self.which = "fgs1" if which not in ("fgs1", "airs") else which

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        fused = h_fgs1 if self.which == "fgs1" else h_airs
        fused = self.norm(fused)
        fused = self.drop(fused)
        return fused, {}


# ---------------------------------- Factory -----------------------------------


_ALIAS: Dict[str, str] = {
    "concat+mlp": "concat_mlp",
    "concat-mlp": "concat_mlp",
    "concat_mlp": "concat_mlp",
    "cross-attend": "cross_attend",
    "cross_attend": "cross_attend",
    "gate": "gated",
    "gated": "gated",
    "residual_sum": "residual_sum",
    "residual-sum": "residual_sum",
    "adapter": "adapter",
    "moe": "moe",
    "identity": "identity",
    "late_blend": "late_blend",
    "late-blend": "late_blend",
    # physics_informed aliases to cross-attend (with optional injection)
    "physics_informed": "cross_attend",
}


def _get_from_cfg(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Robust dict getter for nested Hydra configs. Accepts either:
      - cfg["model"]["fusion"][key]
      - cfg["fusion"][key]
    Falls back to default if not found.
    """
    if cfg is None:
        return default
    if key in cfg:
        return cfg[key]
    model = cfg.get("model", {})
    fusion = model.get("fusion", {})
    return fusion.get(key, default)


def create_fusion(cfg: Dict[str, Any]) -> FusionBase:
    """
    Factory that instantiates the fusion module from a Hydra-style config.

    Expected config structure (typical):
      model:
        fusion:
          type: "concat+mlp" | "cross_attend" | "gated" | "residual_sum" | "adapter" | "moe" | "identity" | "late_blend"
          dim: 256
          dropout: 0.05
          norm: "layernorm" | "rms" | "batch" | "none"
          jit_safe: true
          export:
            taps: false
            attn_weights: false
            gate_values: false
          shapes:
            d_fgs1: 256
            d_airs: 256
            strict_check: true
          # variant-specific subtrees: mlp, attn, pool, gate, proj, late, moe, adapter, passthrough
    """
    # Normalize and resolve type/alias
    raw_type = _get_from_cfg(cfg, "type", default="concat_mlp")
    ftype = _ALIAS.get(str(raw_type), str(raw_type))

    # Read common settings
    dim = int(_get_from_cfg(cfg, "dim", default=256))
    dropout = float(_get_from_cfg(cfg, "dropout", default=0.0))
    norm = _get_from_cfg(cfg, "norm", default="layernorm")
    jit_safe = bool(_get_from_cfg(cfg, "jit_safe", default=True))

    export = _get_from_cfg(cfg, "export", default={}) or {}
    export_taps = bool(export.get("taps", False))
    export_attn = bool(export.get("attn_weights", False))
    export_gate = bool(export.get("gate_values", False))

    shapes = _get_from_cfg(cfg, "shapes", default={}) or {}
    d_fgs1 = int(shapes.get("d_fgs1", dim))
    d_airs = int(shapes.get("d_airs", dim))
    strict_check = bool(shapes.get("strict_check", True))

    # Variant-specific sub-configs
    mlp_cfg = _get_from_cfg(cfg, "mlp", default=None)
    attn_cfg = _get_from_cfg(cfg, "attn", default=None)
    pool_cfg = _get_from_cfg(cfg, "pool", default=None)
    gate_cfg = _get_from_cfg(cfg, "gate", default=None)
    proj_cfg = _get_from_cfg(cfg, "proj", default=None)
    late_cfg = _get_from_cfg(cfg, "late", default=None)
    moe_cfg = _get_from_cfg(cfg, "moe", default=None)
    adapter_cfg = _get_from_cfg(cfg, "adapter", default=None)
    passthrough = _get_from_cfg(cfg, "passthrough", default="fgs1")

    # Extra hints (for physics_informed configs which alias to cross_attend)
    symbolic_injection = _get_from_cfg(cfg, "symbolic_injection", default=None)
    attention_bias_init = float(_get_from_cfg(cfg, "attention_bias_init", default=0.0))

    # Build kwargs shared to all fusers
    common = dict(
        dim=dim,
        dropout=dropout,
        norm=norm,
        jit_safe=jit_safe,
        export_taps=export_taps,
        export_attn=export_attn,
        export_gate=export_gate,
        d_fgs1=d_fgs1,
        d_airs=d_airs,
        strict_check=strict_check,
    )

    if ftype == "concat_mlp":
        return ConcatMLPFusion(mlp_cfg=mlp_cfg, **common)
    elif ftype == "cross_attend":
        return CrossAttentionFusion(
            attn_cfg=attn_cfg,
            pool_cfg=pool_cfg,
            symbolic_injection=symbolic_injection,
            attention_bias_init=attention_bias_init,
            **common,
        )
    elif ftype == "gated":
        return GatedFusion(gate_cfg=gate_cfg, proj_cfg=proj_cfg, **common)
    elif ftype == "residual_sum":
        return ResidualSumFusion(proj_cfg=proj_cfg, **common)
    elif ftype == "adapter":
        return AdapterFusion(adapter_cfg=adapter_cfg, mlp_cfg=mlp_cfg, **common)
    elif ftype == "moe":
        return MoEFusion(moe_cfg=moe_cfg, **common)
    elif ftype == "identity":
        return IdentityFusion(which=passthrough, **common)
    elif ftype == "late_blend":
        return LateBlendFusion(late_cfg=late_cfg, proj_cfg=proj_cfg, **common)
    else:
        raise ValueError(f"Unknown fusion type: {raw_type} (normalized: {ftype})")


__all__ = [
    "FusionBase",
    "FusionExtras",
    "create_fusion",
    "ConcatMLPFusion",
    "CrossAttentionFusion",
    "GatedFusion",
    "ResidualSumFusion",
    "AdapterFusion",
    "MoEFusion",
    "IdentityFusion",
    "LateBlendFusion",
]