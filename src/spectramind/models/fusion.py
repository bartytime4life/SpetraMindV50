# src/spectramind/models/fusion.py
# -----------------------------------------------------------------------------
# SpectraMind V50 — Fusion Module (Single-File Edition)
# -----------------------------------------------------------------------------
# This file consolidates the entire fusion subsystem into one module so you can
# paste a single file into your repo. It contains:
#   - FusionBase + helpers
#   - All fusion implementations (concat_mlp, cross_attend, gated, residual_sum,
#     adapter, moe, late_blend, identity)
#   - Factory create_fusion(cfg) with alias handling
#
# Expected inputs:
#   h_fgs1: [B, Df]  # pooled latent from FGS1 encoder
#   h_airs: [B, Da]  # pooled latent from AIRS encoder
#
# Config contract (Hydra-style dict or DictConfig):
#   model:
#     fusion:
#       type: "concat+mlp" | "cross_attend" | "gated" | "residual_sum" |
#             "adapter" | "moe" | "identity" | "late_blend"
#       dim: 256
#       dropout: 0.05
#       norm: "layernorm" | "rms" | "batch" | "none"
#       jit_safe: true
#       export: { taps: false, attn_weights: false, gate_values: false }
#       shapes: { d_fgs1: 256, d_airs: 256, strict_check: true }
#       # variant-specific subtrees: mlp, attn, pool, gate, proj, late, moe, adapter, passthrough
#
# Usage:
#   from src.spectramind.models.fusion import create_fusion
#   fusion = create_fusion(cfg)
#   fused, extras = fusion(h_fgs1, h_airs, molecule=m, seam=s, wavepos=w, snr=r)
# -----------------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple, TypedDict, List

import math
import torch
import torch.nn as nn


# =========================
# Base + common utilities
# =========================

class FusionExtras(TypedDict, total=False):
    """Optional diagnostics emitted by fusion modules."""
    attn_weights: torch.Tensor     # [B, H, 1, 1] in pooled case (cross_attend)
    gate_values: torch.Tensor      # [B, D] or [B, 1]
    taps: Dict[str, torch.Tensor]  # arbitrary intermediate tensors


def _make_norm(kind: str, dim: int) -> nn.Module:
    kind = (kind or "layernorm").lower()
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "rms":
        # TorchScript‑friendly RMSNorm (weight only)
        class RMSNorm(nn.Module):
            def __init__(self, d: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(d))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                var = x.pow(2).mean(dim=-1, keepdim=True)
                x = x * torch.rsqrt(var + self.eps)
                return x * self.weight
        return RMSNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class FusionBase(nn.Module):
    """Base class for all fusion modules with common checks and toggles."""

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
        """Asserts pooled latents shapes if strict_check is enabled."""
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

    def forward(
        self,
        h_fgs1: torch.Tensor,
        h_airs: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, FusionExtras]:
        raise NotImplementedError


# =========================
# concat_mlp
# =========================

class _MLP(nn.Module):
    """Configurable MLP head; TorchScript‑friendly."""
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
    """[h_fgs1 ; h_airs] → MLP → norm → dropout."""

    def __init__(self, *, mlp_cfg: Optional[Dict] = None, **kwargs) -> None:
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

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        x = torch.cat([h_fgs1, h_airs], dim=-1)
        fused = self.proj(x)
        fused = self.norm(fused)
        fused = self.drop(fused)
        extras: FusionExtras = {}
        if self.export_taps:
            extras["taps"] = {"concat_input": x.detach()}
        return fused, extras


# =========================
# cross_attend
# =========================

class _SelfCrossBlock(nn.Module):
    """Lightweight cross‑attention over pooled tokens (one token per side)."""
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
        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)
        H = self.heads
        D = q.shape[-1]
        q = q.view(B, H, 1, D // H)
        k = k.view(B, H, 1, D // H)
        v = v.view(B, H, 1, D // H)
        attn_logits = (q * self.scale) @ k.transpose(-2, -1)
        attn_logits = attn_logits + self.attn_bias
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.drop_attn(attn)
        out = attn @ v
        out = out.contiguous().view(B, D)
        out = self.proj(out)
        out = self.drop_resid(out)
        return out, attn


class CrossAttentionFusion(FusionBase):
    """
    Bi‑directional cross‑attention (FGS1<‑>AIRS). Supports optional physics‑informed
    conditioning via scalar kwargs: molecule, seam, wavepos, snr (all [B]).
    """

    def __init__(
        self,
        *,
        attn_cfg: Optional[Dict] = None,
        pool_cfg: Optional[Dict] = None,  # reserved for future sequence pooling
        symbolic_injection: Optional[Dict] = None,
        attention_bias_init: float = 0.0,
        **kwargs,
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

        inj_dim = int(self.inject_molecule) + int(self.inject_seams) + int(self.inject_wavepos) + int(self.inject_snr)
        if inj_dim > 0:
            self.cond_fgs1 = nn.Linear(inj_dim, self.d_fgs1, bias=True)
            self.cond_airs = nn.Linear(inj_dim, self.d_airs, bias=True)
        else:
            self.cond_fgs1 = nn.Identity()
            self.cond_airs = nn.Identity()

        self.fgs_from_airs = nn.ModuleList([
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
            ) for _ in range(self.layers)
        ])
        self.airs_from_fgs = nn.ModuleList([
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
            ) for _ in range(self.layers)
        ])
        self.out_proj = nn.Linear(self.d_fgs1 + self.d_airs, self.dim, bias=True)

    def _make_injection(self, inj_inputs: Dict[str, torch.Tensor], B: int) -> torch.Tensor:
        feats = []
        if self.inject_molecule and ("molecule" in inj_inputs):
            feats.append(inj_inputs["molecule"].view(B, 1).to(dtype=torch.float32))
        if self.inject_seams and ("seam" in inj_inputs):
            feats.append(inj_inputs["seam"].view(B, 1).to(dtype=torch.float32))
        if self.inject_wavepos and ("wavepos" in inj_inputs):
            feats.append(inj_inputs["wavepos"].view(B, 1).to(dtype=torch.float32))
        if self.inject_snr and ("snr" in inj_inputs):
            feats.append(inj_inputs["snr"].view(B, 1).to(dtype=torch.float32))
        if len(feats) == 0:
            return torch.zeros((B, 0), dtype=torch.float32, device=inj_inputs.get("device", None))
        return torch.cat(feats, dim=-1)

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        B = h_fgs1.shape[0]
        cond_vec = self._make_injection(kwargs, B)
        if cond_vec.shape[-1] > 0:
            h_fgs1 = h_fgs1 + self.cond_fgs1(cond_vec)
            h_airs = h_airs + self.cond_airs(cond_vec)

        last_attn = None
        q_fgs, q_airs = h_fgs1, h_airs
        for layer_fa, layer_af in zip(self.fgs_from_airs, self.airs_from_fgs):
            upd_fgs, _attn_fa = layer_fa(q_fgs, q_airs)
            q_fgs = q_fgs + upd_fgs
            upd_airs, attn_af = layer_af(q_airs, q_fgs)
            q_airs = q_airs + upd_airs
            last_attn = attn_af

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


# =========================
# gated
# =========================

class _Gate(nn.Module):
    """Scalar or vector gate producing values in [0,1]."""
    def __init__(self, src_dim: int, out_dim: int, kind: str = "vector", hidden: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.kind = kind
        H = int(hidden)
        if kind == "scalar":
            self.net = nn.Sequential(
                nn.Linear(src_dim, H, bias=True),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(H, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(src_dim, H, bias=True),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(H, out_dim, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(FusionBase):
    """g * P_airs(h_airs) + (1 - g) * P_fgs1(h_fgs1)."""

    def __init__(self, *, gate_cfg: Optional[Dict] = None, proj_cfg: Optional[Dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        gate_cfg = gate_cfg or {}
        proj_cfg = proj_cfg or {}
        gate_from = str(gate_cfg.get("from", "fgs1")).lower()  # "fgs1" | "airs" | "both"
        gate_kind = str(gate_cfg.get("kind", "sigmoid_mlp")).lower()
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
        self.gate = _Gate(src_dim=src_dim, out_dim=self.dim, kind="vector" if gate_kind != "scalar" else "scalar", hidden=gate_hidden, dropout=gate_dropout)
        # make module easily traceable by freezing internal projections
        for p in self.p_fgs.parameters():
            p.requires_grad_(False)
        for p in self.p_airs.parameters():
            p.requires_grad_(False)
        for p in self.gate.parameters():
            p.requires_grad_(False)

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        pf = self.p_fgs(h_fgs1)
        pa = self.p_airs(h_airs)
        if self.gate_from == "both":
            src = torch.cat([h_fgs1, h_airs], dim=-1)
        elif self.gate_from == "airs":
            src = h_airs
        else:
            src = h_fgs1
        g = self.gate(src)  # [B, D] or [B,1]
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


# =========================
# residual_sum
# =========================

class ResidualSumFusion(FusionBase):
    """alpha * P_fgs + beta * P_airs (alpha/beta learnable; vector or scalar)."""

    def __init__(self, *, proj_cfg: Optional[Dict] = None, **kwargs) -> None:
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

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        pf = self.p_fgs(h_fgs1)
        pa = self.p_airs(h_airs)
        fused = self.alpha * pf + self.beta * pa
        fused = self.norm(fused)
        fused = self.drop(fused)
        return fused, {}


# =========================
# adapter
# =========================

class _Adapter(nn.Module):
    """Bottleneck adapter with residual: x -> LN -> FC -> Act -> Drop -> FC -> +x."""
    def __init__(self, dim: int, bottleneck: int = 64, activation: str = "relu", dropout: float = 0.05, bias: bool = True) -> None:
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
        y = self.fc1(y); y = self.act(y); y = self.drop(y)
        y = self.fc2(y)
        return x + y


class AdapterFusion(FusionBase):
    """Adapters on each encoder, then concat + tiny MLP to fused dim."""

    def __init__(self, *, adapter_cfg: Optional[Dict] = None, mlp_cfg: Optional[Dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        adapter_cfg = adapter_cfg or {}
        bottleneck = int(adapter_cfg.get("bottleneck", 64))
        activation = str(adapter_cfg.get("activation", "relu"))
        adp_dropout = float(adapter_cfg.get("dropout", 0.05))
        bias = bool(adapter_cfg.get("bias", True))
        self.adp_fgs = _Adapter(self.d_fgs1, bottleneck=bottleneck, activation=activation, dropout=adp_dropout, bias=bias)
        self.adp_airs = _Adapter(self.d_airs, bottleneck=bottleneck, activation=activation, dropout=adp_dropout, bias=bias)
        hidden = (mlp_cfg or {}).get("hidden", [256])
        self.mlp_in = nn.Linear(self.d_fgs1 + self.d_airs, self.dim, bias=True) if not hidden else nn.Linear(self.d_fgs1 + self.d_airs, hidden[0], bias=True)
        self.mlp_out = None
        if hidden:
            self.mlp_out = nn.Sequential(
                nn.GELU(),
                nn.Dropout(float((mlp_cfg or {}).get("dropout", 0.05))) if float((mlp_cfg or {}).get("dropout", 0.05)) > 0 else nn.Identity(),
                nn.Linear(hidden[0], self.dim, bias=True),
            )

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
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


# =========================
# moe
# =========================

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
    """Mixture‑of‑experts over concat([fgs1, airs]) with softmax gating."""

    def __init__(self, *, moe_cfg: Optional[Dict] = None, **kwargs) -> None:
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
        self.experts = nn.ModuleList([
            _Expert(self.d_fgs1 + self.d_airs, hidden, self.dim, activation=activation, dropout=edrop)
            for _ in range(self.num_experts)
        ])
        self.gater = _Gater(gdim, self.num_experts, ghidden, dropout=gdrop, kind="softmax_mlp")

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        cat = torch.cat([h_fgs1, h_airs], dim=-1)
        gsrc_feat = cat
        if self.gater.mlp[0].in_features == self.d_fgs1:
            gsrc_feat = h_fgs1
        elif self.gater.mlp[0].in_features == self.d_airs:
            gsrc_feat = h_airs
        weights = self.gater(gsrc_feat)  # [B, E]
        outs = [expert(cat) for expert in self.experts]  # list of [B, D]
        stack = torch.stack(outs, dim=1)                 # [B, E, D]
        fused = (weights.unsqueeze(-1) * stack).sum(dim=1)
        fused = self.norm(fused)
        fused = self.drop(fused)
        extras: FusionExtras = {}
        if self.export_gate:
            extras["gate_values"] = weights.detach()
        if self.export_taps:
            extras["taps"] = {"concat_input": cat.detach()}
        return fused, extras


# =========================
# late_blend
# =========================

class LateBlendFusion(FusionBase):
    """Blend projected encoders with gamma in [0,1]; supports fixed/learned/cosine schedule."""

    def __init__(self, *, late_cfg: Optional[Dict] = None, proj_cfg: Optional[Dict] = None, **kwargs) -> None:
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

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        pf = self.p_fgs(h_fgs1)
        pa = self.p_airs(h_airs)
        if self.strategy == "fixed":
            gamma = self.fixed_gamma
        elif self.strategy == "cosine_schedule":
            step = int(self._step.item())
            gamma = self._cosine_gamma(step)
            self._step = torch.tensor(step + 1, dtype=torch.long, device=self._step.device)
        else:
            gamma = float(self.gamma.clamp(0.0, 1.0).item())
        fused = gamma * pa + (1.0 - gamma) * pf
        fused = self.norm(fused)
        fused = self.drop(fused)
        extras: FusionExtras = {}
        if self.export_gate:
            extras["gate_values"] = torch.full((h_fgs1.shape[0], 1), gamma, dtype=torch.float32, device=h_fgs1.device)
        if self.export_taps:
            extras["taps"] = {"p_fgs": pf.detach(), "p_airs": pa.detach()}
        return fused, extras


# =========================
# identity
# =========================

class IdentityFusion(FusionBase):
    """Bypass: forward one encoder latent as fused vector."""

    def __init__(self, *, which: str = "fgs1", **kwargs) -> None:
        super().__init__(**kwargs)
        which = (which or "fgs1").lower()
        self.which = "fgs1" if which not in ("fgs1", "airs") else which
        src_dim = self.d_fgs1 if self.which == "fgs1" else self.d_airs
        self.proj = nn.Identity() if src_dim == self.dim else nn.Linear(src_dim, self.dim)

    def forward(self, h_fgs1: torch.Tensor, h_airs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, FusionExtras]:
        self._assert_shapes(h_fgs1, h_airs)
        fused = h_fgs1 if self.which == "fgs1" else h_airs
        fused = self.proj(fused)
        fused = self.norm(fused)
        fused = self.drop(fused)
        return fused, {}


# =========================
# Factory + aliases
# =========================

_ALIAS = {
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
    # physics_informed just maps to cross_attend with conditioning flags
    "physics_informed": "cross_attend",
}

_REGISTRY = {
    "concat_mlp": ConcatMLPFusion,
    "cross_attend": CrossAttentionFusion,
    "gated": GatedFusion,
    "residual_sum": ResidualSumFusion,
    "adapter": AdapterFusion,
    "moe": MoEFusion,
    "identity": IdentityFusion,
    "late_blend": LateBlendFusion,
}


def _get_from_cfg(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get key from either cfg[key] or cfg['model']['fusion'][key]."""
    if cfg is None:
        return default
    if key in cfg:
        return cfg[key]
    # Support both top-level "fusion" and nested "model.fusion" layouts
    fusion_section = None
    if isinstance(cfg.get("fusion"), dict):
        fusion_section = cfg["fusion"]
    else:
        model = cfg.get("model", {})
        fusion_section = model.get("fusion", {})
    return fusion_section.get(key, default)


def create_fusion(cfg: Dict[str, Any]) -> FusionBase:
    """
    Instantiate a fusion module from a Hydra-style config (dict or DictConfig).
    """
    raw_type = _get_from_cfg(cfg, "type", default="concat_mlp")
    ftype = _ALIAS.get(str(raw_type), str(raw_type))

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

    # Variant subtrees
    mlp_cfg = _get_from_cfg(cfg, "mlp", default=None)
    attn_cfg = _get_from_cfg(cfg, "attn", default=None)
    pool_cfg = _get_from_cfg(cfg, "pool", default=None)
    gate_cfg = _get_from_cfg(cfg, "gate", default=None)
    proj_cfg = _get_from_cfg(cfg, "proj", default=None)
    late_cfg = _get_from_cfg(cfg, "late", default=None)
    moe_cfg = _get_from_cfg(cfg, "moe", default=None)
    adapter_cfg = _get_from_cfg(cfg, "adapter", default=None)
    passthrough = _get_from_cfg(cfg, "passthrough", default="fgs1")
    symbolic_injection = _get_from_cfg(cfg, "symbolic_injection", default=None)
    attention_bias_init = float(_get_from_cfg(cfg, "attention_bias_init", default=0.0))

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
        mod = ConcatMLPFusion(mlp_cfg=mlp_cfg, **common)
    elif ftype == "cross_attend":
        mod = CrossAttentionFusion(
            attn_cfg=attn_cfg,
            pool_cfg=pool_cfg,
            symbolic_injection=symbolic_injection,
            attention_bias_init=attention_bias_init,
            **common,
        )
    elif ftype == "gated":
        mod = GatedFusion(gate_cfg=gate_cfg, proj_cfg=proj_cfg, **common)
    elif ftype == "residual_sum":
        mod = ResidualSumFusion(proj_cfg=proj_cfg, **common)
    elif ftype == "adapter":
        mod = AdapterFusion(adapter_cfg=adapter_cfg, mlp_cfg=mlp_cfg, **common)
    elif ftype == "moe":
        mod = MoEFusion(moe_cfg=moe_cfg, **common)
    elif ftype == "identity":
        mod = IdentityFusion(which=passthrough, **common)
    elif ftype == "late_blend":
        mod = LateBlendFusion(late_cfg=late_cfg, proj_cfg=proj_cfg, **common)
    else:
        raise ValueError(f"Unknown fusion type: {raw_type} (normalized: {ftype})")

    # freeze params to ease tracing and evaluation
    mod.requires_grad_(False)
    return mod


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
    "LateBlendFusion",
    "IdentityFusion",
]