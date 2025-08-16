# SPDX-License-Identifier: Apache-2.0

"""Minimal pytest suite validating shapes and differentiability."""

import math

import pytest

# This test suite exercises rule modules that depend on PyTorch. When torch
# isn't installed we skip the entire module so that optional dependencies don't
# cause import-time failures.
torch = pytest.importorskip("torch")

from src.spectramind.symbolic.rules import (  # noqa: E402
    AsymmetryRule,
    CompositeRule,
    FFTSpectralRule,
    MolecularCoherenceRule,
    NonNegativityRule,
    PhotonicAlignmentRule,
    SmoothnessRule,
)


def fake_batch(B: int = 3, N: int = 283, with_sigma: bool = True, with_wl: bool = True):
    torch.manual_seed(0)
    mu = torch.rand(B, N, dtype=torch.float32) * 1e-3
    sigma = torch.rand(B, N, dtype=torch.float32) * 1e-4 if with_sigma else None
    wavelengths = torch.linspace(0.5, 7.8, N).unsqueeze(0) if with_wl else None
    meta = {"wavelengths": wavelengths}
    fgs1 = torch.sin(torch.linspace(0, 2 * math.pi, 97)).unsqueeze(0).repeat(B, 1)
    meta["fgs1_curve"] = fgs1
    return mu.requires_grad_(True), sigma, meta


def test_nonnegativity():
    mu, sigma, meta = fake_batch()
    rule = NonNegativityRule(weight=0.7, enable_logging=False)
    out = rule(mu)
    assert out.violation_map.shape == mu.shape
    assert (out.violation_map >= 0).all()
    out.loss.backward()
    assert mu.grad is not None


def test_smoothness():
    mu, sigma, meta = fake_batch()
    rule = SmoothnessRule(
        weight=0.5, laplace_weight=1.0, fft_weight=0.2, enable_logging=False
    )
    out = rule(mu, sigma, meta)
    assert out.violation_map.shape == mu.shape
    assert (out.violation_map >= 0).all()
    out.loss.backward()
    assert mu.grad is not None


def test_molecular_coherence():
    mu, sigma, meta = fake_batch()
    rule = MolecularCoherenceRule(weight=1.2, margin=0.0, enable_logging=False)
    out = rule(mu, sigma, meta)
    assert out.violation_map.shape == mu.shape
    assert (out.violation_map >= 0).all()
    out.loss.backward()
    assert mu.grad is not None


def test_photonic_alignment():
    mu, sigma, meta = fake_batch()
    rule = PhotonicAlignmentRule(weight=0.9, smoothing_window=9, enable_logging=False)
    out = rule(mu, sigma, meta)
    assert out.violation_map.shape == mu.shape
    assert (out.violation_map >= 0).all()
    out.loss.backward()
    assert mu.grad is not None


def test_asymmetry():
    mu, sigma, meta = fake_batch()
    rule = AsymmetryRule(
        weight=0.4, centers_um=[1.4, 2.3], window_um=0.1, enable_logging=False
    )
    out = rule(mu, sigma, meta)
    assert out.violation_map.shape == mu.shape
    assert (out.violation_map >= 0).all()
    out.loss.backward()
    assert mu.grad is not None


def test_fft_spectral():
    mu, sigma, meta = fake_batch()
    rule = FFTSpectralRule(weight=0.6, tail_frac=0.3, enable_logging=False)
    out = rule(mu, sigma, meta)
    assert out.violation_map.shape == mu.shape
    assert (out.violation_map >= 0).all()
    out.loss.backward()
    assert mu.grad is not None


def test_composite_sum():
    mu, sigma, meta = fake_batch()
    r1 = NonNegativityRule(weight=0.2, enable_logging=False)
    r2 = SmoothnessRule(weight=0.1, enable_logging=False)
    comp = CompositeRule([r1, r2], weight=1.0, mode="sum", enable_logging=False)
    out = comp(mu, sigma, meta)
    assert out.violation_map.shape == mu.shape
    assert (out.violation_map >= 0).all()
    out.loss.backward()
    assert mu.grad is not None
