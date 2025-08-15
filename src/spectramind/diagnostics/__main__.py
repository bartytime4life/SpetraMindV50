"""Unified CLI for SpectraMind V50 Diagnostics."""

import argparse

from ._config import load_config_from_yaml_or_defaults
from .check_calibration import run as run_cal
from .fft_power_compare import run as run_fft
from .generate_diagnostic_summary import run as run_summary
from .gll_error_localizer import run as run_gll
from .shap_overlay import run as run_shap
from .spectral_smoothness_map import run as run_smooth
from .symbolic_influence_map import run as run_symb


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="spectramind.diagnostics",
        description="SpectraMind V50 — Diagnostics CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config", type=str, default=None, help="Hydra-style YAML config"
        )

    p_sum = sub.add_parser(
        "generate-summary", help="Generate compact diagnostic summary JSON"
    )
    add_common(p_sum)

    p_gll = sub.add_parser("gll-localize", help="Compute bin-wise GLL and heatmap")
    add_common(p_gll)

    p_shap = sub.add_parser(
        "shap-overlay", help="μ × SHAP overlay (with pseudo-SHAP fallback)"
    )
    add_common(p_shap)
    p_shap.add_argument("--shap", type=str, default=None, help="Path to [N,B] SHAP npy")
    p_shap.add_argument("--top-k", type=int, default=20)

    p_fft = sub.add_parser("fft-compare", help="FFT power PCA + clustering")
    add_common(p_fft)
    p_fft.add_argument("--k", type=int, default=8)

    p_sm = sub.add_parser("smoothness-map", help="Compute spectral smoothness map")
    add_common(p_sm)

    p_cal = sub.add_parser("check-calibration", help="σ vs residual calibration checks")
    add_common(p_cal)

    p_sym = sub.add_parser("symbolic-influence", help="Compute symbolic influence maps")
    add_common(p_sym)
    p_sym.add_argument("--violation-masks", type=str, required=True)
    p_sym.add_argument("--rule-weights", type=str, default=None)
    p_sym.add_argument(
        "--mode",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "sum", "max"],
    )

    args = parser.parse_args()

    if args.cmd == "generate-summary":
        cfg = load_config_from_yaml_or_defaults(args.config)
        run_summary(cfg, config_path=args.config)
        return
    if args.cmd == "gll-localize":
        cfg = load_config_from_yaml_or_defaults(args.config)
        run_gll(cfg, config_path=args.config)
        return
    if args.cmd == "shap-overlay":
        cfg = load_config_from_yaml_or_defaults(args.config)
        run_shap(cfg, shap_path=args.shap, top_k=args.top_k)
        return
    if args.cmd == "fft-compare":
        cfg = load_config_from_yaml_or_defaults(args.config)
        run_fft(cfg, k=args.k)
        return
    if args.cmd == "smoothness-map":
        cfg = load_config_from_yaml_or_defaults(args.config)
        run_smooth(cfg)
        return
    if args.cmd == "check-calibration":
        cfg = load_config_from_yaml_or_defaults(args.config)
        run_cal(cfg)
        return
    if args.cmd == "symbolic-influence":
        cfg = load_config_from_yaml_or_defaults(args.config)
        run_symb(
            cfg,
            violation_masks_path=args.violation_masks,
            rule_weights_path=args.rule_weights,
            mode=args.mode,
        )
        return


if __name__ == "__main__":
    main()
