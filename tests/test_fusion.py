# pytest -q
import torch
import importlib

B = 4
Df = 32
Da = 48
D  = 64

def _inputs():
    torch.manual_seed(0)
    return torch.randn(B, Df), torch.randn(B, Da)

def _cfg(ftype: str):
    return {
        "fusion": {
            "type": ftype, "dim": D, "dropout": 0.01, "norm": "layernorm",
            "export": {"taps": True, "attn_weights": True, "gate_values": True},
            "shapes": {"d_fgs1": Df, "d_airs": Da, "strict_check": True},
            "late": {"strategy": "learned"},
            "moe": {"num_experts": 3, "expert_hidden": 128, "gating": {"source": "concat", "hidden": 64}},
        }
    }

def test_factory_and_forward_all():
    fusion_mod = importlib.import_module("src.spectramind.models.fusion")
    variants = [
        "concat+mlp", "cross_attend", "physics_informed",
        "gated", "residual_sum", "adapter", "moe", "late_blend", "identity"
    ]
    x1, x2 = _inputs()
    for t in variants:
        fuser = fusion_mod.create_fusion(_cfg(t))
        y, extras = fuser(x1, x2, molecule=torch.ones(B), seam=torch.zeros(B),
                          wavepos=torch.linspace(0,1,B), snr=torch.full((B,), 50.0))
        assert tuple(y.shape) == (B, D)
        assert isinstance(extras, dict)

def test_torchscript_trace_minimal():
    fusion_mod = importlib.import_module("src.spectramind.models.fusion")
    fuser = fusion_mod.create_fusion(_cfg("gated")).eval()
    x1, x2 = _inputs()
    with torch.no_grad():
        y, _ = fuser(x1, x2)
        assert tuple(y.shape) == (B, D)
    def fn(a, b):
        z, _ = fuser(a, b)
        return z
    ts = torch.jit.trace(fn, (x1, x2))
    yz = ts(x1, x2)
    assert tuple(yz.shape) == (B, D)
