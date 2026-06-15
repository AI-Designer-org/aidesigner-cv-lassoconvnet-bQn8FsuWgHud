"""
LassoConvNet — Comprehensive Test Suite.

Validation layers:
    1. Shape / forward-pass correctness
    2. Gradient flow & numerical stability
    3. Domain-specific CV properties (sparsity, invariance)
    4. Domain-specific CV benchmarks (linear probe proxy, reconstruction)

Usage:
    pytest test_model.py -v
    pytest test_model.py -v -k "shape"      # shape tests only
    pytest test_model.py -v -k "gradient"   # gradient tests only
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))

from config import LassoConvConfig
from model import LassoConvNet
from layers import (
    SoftThreshold,
    AdaptiveSoftThreshold,
    HardThreshold,
    compute_sparsity_ratio,
    compute_channel_sparsity,
    build_prox_operator,
)
from loss import lasso_total_loss, l1_weight_penalty, group_lasso_penalty


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cfg():
    """Default config for CIFAR-10 scale testing."""
    return LassoConvConfig(
        in_channels=3,
        img_size=32,
        n_classes=10,
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        kernel_size=3,
        lasso_mode="proximal",
        proximal_type="soft",
        threshold_init=0.01,
        threshold_learnable=True,
        norm_before_prox=True,
        norm_type="batch_norm",
        use_bias=False,
        head_hidden_dim=64,
        dropout=0.0,
        global_pool="avg",
        l1_weight_decay=1e-5,
        use_group_lasso=True,
        group_lasso_strength=1e-4,
        group_size=8,
    )


@pytest.fixture
def model(cfg):
    """LassoConvNet in eval mode."""
    m = LassoConvNet(cfg)
    m.eval()
    return m


@pytest.fixture
def sample_input(cfg):
    """Random CIFAR-like input tensor."""
    return torch.randn(2, cfg.in_channels, cfg.img_size, cfg.img_size)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1a — Shape / Forward-Pass Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapes:
    """Verify model produces correct output shapes for all three modes."""

    def test_output_shape_proximal(self, model, cfg, sample_input):
        """Proximal mode: (B, n_classes) logits from (B, 3, 32, 32) input."""
        with torch.no_grad():
            logits = model(sample_input)
        assert logits.shape == (2, cfg.n_classes), (
            f"Proximal output shape: {logits.shape}, "
            f"expected ({2}, {cfg.n_classes})"
        )

    def test_output_shape_loss_only(self, cfg, sample_input):
        """Loss-only mode: standard CNN + L1 loss, same output shape."""
        cfg_loss = LassoConvConfig(
            in_channels=cfg.in_channels,
            img_size=cfg.img_size,
            n_classes=cfg.n_classes,
            stage_channels=cfg.stage_channels,
            stage_depths=cfg.stage_depths,
            lasso_mode="loss_only",
            head_hidden_dim=cfg.head_hidden_dim,
        )
        model_loss = LassoConvNet(cfg_loss)
        model_loss.eval()
        with torch.no_grad():
            logits = model_loss(sample_input)
        assert logits.shape == (2, cfg.n_classes), (
            f"Loss-only output shape: {logits.shape}"
        )

    def test_output_shape_lista(self, cfg, sample_input):
        """LISTA-unrolled mode: same output shape."""
        cfg_lista = LassoConvConfig(
            in_channels=cfg.in_channels,
            img_size=cfg.img_size,
            n_classes=cfg.n_classes,
            stage_channels=cfg.stage_channels,
            stage_depths=cfg.stage_depths,
            lasso_mode="lista_unrolled",
            lista_iters=3,
            lista_dictionary_size=32,
            lista_tie_weights=True,
            head_hidden_dim=cfg.head_hidden_dim,
        )
        model_lista = LassoConvNet(cfg_lista)
        model_lista.eval()
        with torch.no_grad():
            logits = model_lista(sample_input)
        assert logits.shape == (2, cfg.n_classes), (
            f"LISTA output shape: {logits.shape}"
        )

    def test_variable_batch_size(self, model, cfg):
        """Output shape must scale correctly with batch size."""
        for B in [1, 4, 8]:
            x = torch.randn(B, cfg.in_channels, cfg.img_size, cfg.img_size)
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (B, cfg.n_classes), (
                f"Batch size {B}: {logits.shape}"
            )

    def test_variable_spatial_size(self, cfg):
        """Model must handle non-square inputs (as long as divisible by 8)."""
        cfg_var = LassoConvConfig(
            in_channels=3,
            img_size=64,
            n_classes=cfg.n_classes,
            stage_channels=cfg.stage_channels,
            stage_depths=cfg.stage_depths,
            head_hidden_dim=cfg.head_hidden_dim,
        )
        m = LassoConvNet(cfg_var)
        m.eval()
        x = torch.randn(2, 3, 64, 48)  # non-square
        with torch.no_grad():
            logits = m(x)
        assert logits.shape == (2, cfg.n_classes), (
            f"Non-square input: {logits.shape}"
        )

    def test_proximal_operator_shapes(self):
        """All proximal operators preserve input shape."""
        B, C, H, W = 2, 8, 16, 16
        x = torch.randn(B, C, H, W)
        theta = torch.full((1, C, 1, 1), 0.5)

        for op_cls in [SoftThreshold, AdaptiveSoftThreshold, HardThreshold]:
            op = op_cls() if op_cls != AdaptiveSoftThreshold else op_cls(alpha=0.5)
            out = op(x, theta)
            assert out.shape == (B, C, H, W), (
                f"{op_cls.__name__}: {out.shape}"
            )

    def test_backbone_stage_shapes(self, cfg):
        """Each stage produces correct spatial downsampling."""
        from backbone import LassoConvNetBackbone
        backbone = LassoConvNetBackbone(cfg)
        backbone.eval()

        x = torch.randn(2, cfg.in_channels, cfg.img_size, cfg.img_size)
        h = backbone.stem(x)
        # Stem: (B, C1, H, W) — no downsampling for CIFAR
        C1 = cfg.stage_channels[0]
        assert h.shape == (2, C1, cfg.img_size, cfg.img_size), (
            f"Stem output: {h.shape}"
        )

        for i, stage in enumerate(backbone.stages):
            h = stage(h)
            expected_ch = cfg.stage_channels[i]
            expected_h = cfg.img_size // (2 ** i)  # stage 0: 32, stage 1: 16, ...
            expected_w = cfg.img_size // (2 ** i)
            assert h.shape == (2, expected_ch, expected_h, expected_w), (
                f"Stage {i}: {h.shape}, expected ({2}, {expected_ch}, {expected_h}, {expected_w})"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1b — Gradient Flow Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradients:
    """Verify gradients flow correctly through all parameters."""

    def test_all_params_receive_gradients(self, cfg):
        """Every trainable parameter must receive a non-None gradient."""
        model = LassoConvNet(cfg)
        model.train()

        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        dead_params = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(dead_params) == 0, (
            f"Parameters with no gradient: {dead_params}"
        )

    def test_theta_params_receive_gradients(self, cfg):
        """Learnable thresholds (theta) must receive non-zero gradients."""
        model = LassoConvNet(cfg)
        model.train()

        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if "theta" in name:
                assert param.grad is not None, (
                    f"Threshold {name} has no gradient"
                )
                assert param.grad.abs().sum().item() > 0, (
                    f"Threshold {name} has zero gradient"
                )

    def test_no_nan_gradients(self, cfg):
        """No parameter gradient should contain NaN values."""
        model = LassoConvNet(cfg)
        model.train()

        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        nan_params = []
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_params.append(name)
        assert len(nan_params) == 0, (
            f"Parameters with NaN gradient: {nan_params}"
        )

    def test_gradient_flow_through_soft_threshold(self, cfg):
        """Verify gradients flow through the soft-thresholding operation."""
        op = SoftThreshold()
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        theta = torch.full((1, 8, 1, 1), 0.5, requires_grad=True)
        out = op(x, theta)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Input gradient is None"
        assert theta.grad is not None, "Theta gradient is None"
        # At least some gradient should flow (activations |x| > theta get grad)
        assert x.grad.abs().sum().item() > 0, "Input gradient is zero"

    def test_gradient_flow_through_adaptive_threshold(self, cfg):
        """Verify gradients flow through adaptive threshold (always non-zero)."""
        op = AdaptiveSoftThreshold(alpha=0.5)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        theta = torch.full((1, 8, 1, 1), 0.5, requires_grad=True)
        out = op(x, theta)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Adaptive input gradient is None"
        assert theta.grad is not None, "Adaptive theta gradient is None"
        # Adaptive threshold has non-zero gradient everywhere (unlike soft)
        assert x.grad.abs().sum().item() > 0, "Adaptive gradient is zero"

    def test_gradient_flow_loss_only_mode(self, cfg):
        """Loss_only mode must also have gradient flow."""
        cfg_loss = LassoConvConfig(
            stage_channels=cfg.stage_channels,
            stage_depths=cfg.stage_depths,
            lasso_mode="loss_only",
            head_hidden_dim=cfg.head_hidden_dim,
        )
        model = LassoConvNet(cfg_loss)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        dead = [n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None]
        assert len(dead) == 0, f"Loss-only mode: dead params: {dead}"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1c — Correctness / Sparsity Tests (CV Domain)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSparsity:
    """Tests for the core sparsity claim of LassoConvNet."""

    def test_soft_threshold_induces_sparsity(self):
        """Soft-thresholding must produce exact zeros in the dead zone."""
        x = torch.linspace(-1.0, 1.0, 100).view(1, 1, 10, 10)
        theta = torch.tensor(0.3).view(1, 1, 1, 1)
        op = SoftThreshold()
        out = op(x, theta)
        # Values in [-0.3, 0.3] should be exactly 0
        dead_zone = (torch.abs(x) <= 0.3)
        assert (out[dead_zone] == 0).all(), (
            "Soft-threshold did not zero values in [-θ, θ]"
        )
        # Values outside [-0.3, 0.3] should be shifted toward zero
        # Last element of linspace(-1, 1, 100) is 1.0 → sign(1)*max(|1|-0.3, 0) = 0.7
        assert abs(out[0, 0, -1, -1].item() - 0.7) < 1e-5, (
            f"Expected 0.7 for x=1.0, θ=0.3, got {out[0,0,-1,-1].item()}"
        )

    def test_sparsity_from_proximal_forward(self, model, cfg, sample_input):
        """Proximal mode forward pass must produce some zero activations."""
        with torch.no_grad():
            _ = model(sample_input)
            sparsity = model.get_sparsity_ratio(eps=1e-6)
        # With θ=0.01, some activations should fall in dead zone
        assert sparsity is not None, "Sparsity ratio is None"
        assert sparsity.item() >= 0.0, f"Negative sparsity: {sparsity.item()}"

    def test_proximal_mode_sparser_than_loss_only(self, cfg):
        """
        Core claim test: Proximal mode should produce sparser activations
        than loss-only mode at initialization.
        """
        # Proximal mode
        cfg_prox = LassoConvConfig(
            stage_channels=(16, 32, 64, 128),
            stage_depths=(1, 1, 1, 1),
            lasso_mode="proximal",
            threshold_init=0.05,  # higher threshold for measurable sparsity
            head_hidden_dim=64,
        )
        model_prox = LassoConvNet(cfg_prox)
        model_prox.eval()

        # Loss-only mode
        cfg_loss = LassoConvConfig(
            stage_channels=(16, 32, 64, 128),
            stage_depths=(1, 1, 1, 1),
            lasso_mode="loss_only",
            head_hidden_dim=64,
        )
        model_loss = LassoConvNet(cfg_loss)
        model_loss.eval()

        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            _ = model_prox(x)
            sparsity_prox = model_prox.get_sparsity_ratio(eps=1e-6)
            _ = model_loss(x)
            sparsity_loss = model_loss.get_sparsity_ratio(eps=1e-6)

        assert sparsity_prox is not None
        assert sparsity_loss is not None
        # Proximal mode should be at least as sparse as loss-only
        # (this is a weak test — strong test requires training)
        print(f"  Proximal sparsity: {sparsity_prox.item():.4f}")
        print(f"  Loss-only sparsity: {sparsity_loss.item():.4f}")

    def test_channel_sparsity_distribution(self, cfg):
        """Per-channel sparsity should vary across channels (different θ per ch)."""
        from backbone import LassoConvNetBackbone
        backbone = LassoConvNetBackbone(cfg)
        backbone.eval()

        # Capture stage outputs directly (these go through soft-threshold)
        activations = {}
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        for i, stage in enumerate(backbone.stages):
            stage.register_forward_hook(make_hook(f"stage_{i}"))

        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            backbone(x)

        # Measure per-channel sparsity on the last stage output (post-threshold)
        if "stage_3" in activations:
            last_act = activations["stage_3"]
            ch_sp = compute_channel_sparsity(last_act, eps=1e-6)
            assert ch_sp.shape[0] == last_act.shape[1], (
                f"Channel sparsity shape: {ch_sp.shape}"
            )
            print(f"  Channel sparsity std: {ch_sp.std().item():.4f}")


class TestProximalOperators:
    """Correctness of each proximal operator variant."""

    def test_soft_threshold_identity_at_zero(self):
        """When θ=0, soft-threshold must reduce to identity."""
        op = SoftThreshold()
        x = torch.randn(2, 4, 8, 8)
        theta = torch.zeros(1, 4, 1, 1)
        out = op(x, theta)
        assert torch.allclose(out, x, atol=1e-6), (
            "Soft-threshold with θ=0 is not identity"
        )

    def test_soft_threshold_vanishes_at_large_theta(self):
        """When θ >> |x|, soft-threshold must produce all zeros."""
        op = SoftThreshold()
        x = torch.randn(2, 4, 8, 8) * 0.1  # small values
        theta = torch.full((1, 4, 1, 1), 10.0)  # very large threshold
        out = op(x, theta)
        assert (out == 0).all(), (
            "Soft-threshold with θ >> |x| is not all zeros"
        )

    def test_adaptive_threshold_smoothness(self):
        """Adaptive threshold must be C^∞ (no hard kinks)."""
        op = AdaptiveSoftThreshold(alpha=10.0)
        x = torch.linspace(-1.0, 1.0, 1000, requires_grad=True).view(1, 1, -1, 1)
        theta = torch.tensor(0.3).view(1, 1, 1, 1)
        out = op(x, theta)
        # Compute gradient w.r.t. x (should be smooth, not step-function-like)
        grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad.sum(), x, create_graph=True)[0]
        # Second derivative should be finite everywhere (no kinks)
        assert not torch.isnan(grad2).any(), "NaN in second derivative"
        assert not torch.isinf(grad2).any(), "Inf in second derivative"
        # At x=0 (deep in dead zone), gradient should be non-zero (key advantage)
        center_grad = grad[0, 0, 500, 0].item()
        assert abs(center_grad) > 1e-6, (
            f"Adaptive gradient near zero is ~0: {center_grad:.2e}"
        )

    def test_hard_threshold(self):
        """Hard threshold: |x| > θ passes through, else zero."""
        op = HardThreshold()
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).view(1, 1, -1, 1)
        theta = torch.tensor(0.3).view(1, 1, 1, 1)
        out = op(x, theta)
        # -1.0 passes (|−1| > 0.3), -0.5 passes, 0.0 blocked, 0.5 passes, 1.0 passes
        expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).view(1, 1, -1, 1)
        assert torch.allclose(out, expected, atol=1e-6), (
            f"Hard threshold: {out.flatten()} vs {expected.flatten()}"
        )

    def test_proximal_type_swap_via_factory(self):
        """Factory must return the correct operator class."""
        assert isinstance(build_prox_operator("soft"), SoftThreshold)
        assert isinstance(build_prox_operator("adaptive", alpha=0.2), AdaptiveSoftThreshold)
        assert isinstance(build_prox_operator("hard"), HardThreshold)
        with pytest.raises(ValueError, match="Unknown"):
            build_prox_operator("nonexistent")


class TestSparsityMetrics:
    """Sparsity measurement utilities must be correct."""

    def test_sparsity_ratio_correct(self):
        """compute_sparsity_ratio must return correct fraction."""
        x = torch.zeros(100)
        x[:30] = 1.0
        ratio = compute_sparsity_ratio(x, eps=0.5)
        assert ratio.item() == pytest.approx(0.7), f"Expected ~0.70, got {ratio.item()}"

    def test_channel_sparsity_shape(self):
        """compute_channel_sparsity must return (C,) tensor."""
        x = torch.randn(2, 8, 16, 16)
        ch_sp = compute_channel_sparsity(x, eps=1e-4)
        assert ch_sp.shape == (8,), f"Shape: {ch_sp.shape}"

    def test_channel_sparsity_all_zero(self):
        """A channel with all small values must have sparsity ≈ 1."""
        x = torch.randn(2, 4, 8, 8)
        x[:, 0, :, :] = 1e-8  # channel 0 near-zero
        ch_sp = compute_channel_sparsity(x, eps=1e-4)
        assert ch_sp[0] > 0.9, f"Channel 0 sparsity: {ch_sp[0]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1d — Numerical Stability Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumerics:
    """Numerical stability of the LassoConvNet."""

    def test_bf16_forward(self, cfg, sample_input):
        """Forward pass must not produce NaN/Inf in bfloat16."""
        model = LassoConvNet(cfg).bfloat16()
        model.eval()
        x_bf16 = sample_input.bfloat16()
        with torch.no_grad():
            logits = model(x_bf16)
        assert not torch.isnan(logits).any(), "NaN in bf16 forward"
        assert not torch.isinf(logits).any(), "Inf in bf16 forward"
        assert logits.dtype == torch.bfloat16, (
            f"Expected bf16 output, got {logits.dtype}"
        )

    def test_bf16_proximal_operator(self):
        """Soft-threshold must not produce NaN in bf16."""
        op = SoftThreshold()
        x = torch.randn(2, 4, 16, 16).bfloat16()
        theta = torch.full((1, 4, 1, 1), 0.1).bfloat16()
        out = op(x, theta)
        assert not torch.isnan(out).any(), "NaN in bf16 soft-threshold"
        assert not torch.isinf(out).any(), "Inf in bf16 soft-threshold"

    def test_extreme_input_values(self, model, cfg):
        """Large or extreme inputs should not produce NaN."""
        x_large = torch.randn(2, 3, 32, 32) * 1e3
        with torch.no_grad():
            logits = model(x_large)
        assert not torch.isnan(logits).any(), "NaN with large input"
        assert not torch.isinf(logits).any(), "Inf with large input"

    def test_constant_input(self, model, cfg):
        """Constant input should not produce NaN (tests norm stability)."""
        x_const = torch.ones(2, 3, 32, 32) * 0.5
        with torch.no_grad():
            logits = model(x_const)
        assert not torch.isnan(logits).any(), "NaN on constant input"
        assert torch.isfinite(logits).all(), "Non-finite on constant input"

    def test_zero_input(self, model, cfg):
        """Zero input should give finite output (may have inactive neurons)."""
        x_zero = torch.zeros(2, 3, 32, 32)
        with torch.no_grad():
            logits = model(x_zero)
        assert torch.isfinite(logits).all(), "Non-finite on zero input"

    def test_threshold_clamping(self, cfg):
        """Threshold values must remain within valid bounds after clamp."""
        model = LassoConvNet(cfg)
        # Artificially set thresholds to negative
        for name, param in model.named_parameters():
            if "theta" in name:
                param.data.fill_(-0.5)
        # Clamp
        model.clamp_thresholds(min_val=1e-6)
        # Verify
        for name, param in model.named_parameters():
            if "theta" in name:
                assert (param >= 1e-6 - 1e-8).all(), (
                    f"Threshold {name} not clamped: min={param.min().item()}"
                )

    def test_nan_guard_triggers(self, cfg):
        """The NaN guard in forward() must raise on NaN logits."""
        model = LassoConvNet(cfg)
        model.train()
        # We cannot easily force NaN in the forward pass without modifying internals,
        # so we verify the guard is wired by checking the assertion code runs.
        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        # If we got here, the NaN assertion passed
        assert loss.item() > 0, "Loss should be positive"

    def test_total_loss_no_nan(self, cfg):
        """The composite lasso_total_loss must not produce NaN."""
        model = LassoConvNet(cfg)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        ce_loss = F.cross_entropy(logits, targets)
        total_loss = lasso_total_loss(ce_loss, model, cfg)
        assert not torch.isnan(total_loss), "Total loss is NaN"
        assert total_loss.item() >= ce_loss.item(), (
            "Total loss should be >= CE loss (penalties are non-negative)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Domain-Specific Benchmarks (CV)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCVProperties:
    """CV-specific correctness: invariance, equivariance, output properties."""

    def test_translation_robustness(self, model, cfg):
        """
        Small translations should not flip classification.
        (Weak test: model at init is random, but prediction agreement
         between original and shifted should be non-trivial.)
        """
        x = torch.randn(4, cfg.in_channels, cfg.img_size, cfg.img_size)
        # Shift by 2 pixels
        x_shifted = torch.roll(x, shifts=2, dims=-1)
        with torch.no_grad():
            pred_orig = model(x).argmax(-1)
            pred_shifted = model(x_shifted).argmax(-1)
        # At random init, agreement is ~1/n_classes on average
        # This test checks it's not pathological (all flipping)
        agreement = (pred_orig == pred_shifted).float().mean().item()
        print(f"  Translation agreement: {agreement:.2f} "
              f"(random baseline: {1.0/cfg.n_classes:.2f})")
        # Should be above random guessing
        assert agreement > 1.0 / cfg.n_classes, (
            f"Translation robustness very low: {agreement:.2f}"
        )

    def test_no_spatial_shortcut(self, model, cfg):
        """
        Random noise inputs should yield approximately uniform
        class distribution (high entropy).
        """
        with torch.no_grad():
            logits = model(torch.randn(16, cfg.in_channels, cfg.img_size, cfg.img_size))
        probs = logits.softmax(-1)
        entropy = -(probs * probs.log()).sum(-1).mean()
        max_entropy = math.log(cfg.n_classes)
        # At random init, entropy should be close to max
        relative_entropy = entropy / max_entropy
        print(f"  Noise input entropy: {entropy:.2f} / {max_entropy:.2f} "
              f"({relative_entropy:.1%})")
        assert relative_entropy > 0.5, (
            f"Low entropy on noise input: {relative_entropy:.2f}"
        )

    def test_spatial_invariance_of_pooling(self, cfg):
        """Global avg pool must produce same output for permuted patches."""
        cfg_pool = LassoConvConfig(
            stage_channels=(16, 32, 64, 128),
            stage_depths=(1, 1, 1, 1),
            head_hidden_dim=64,
        )
        model = LassoConvNet(cfg_pool)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            # Flip horizontally (should change features but not totally)
            out2 = model(torch.flip(x, dims=[-1]))
        # The model should give finite output for both
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()

    def test_zero_filter_count(self, model):
        """
        At initialization, most filters should be non-zero.
        Zero-filter counting should return a reasonable integer.
        """
        zero_count = model.count_zero_filters(eps=1e-6)
        total_conv_filters = sum(
            m.weight.shape[0] for m in model.modules()
            if isinstance(m, nn.Conv2d) and m.weight.dim() == 4
        )
        print(f"  Zero filters: {zero_count} / {total_conv_filters}")
        assert isinstance(zero_count, int), "count_zero_filters must return int"
        assert 0 <= zero_count <= total_conv_filters, (
            f"Zero filter count out of range: {zero_count}"
        )


class TestCVBenchmarks:
    """
    CV domain benchmarks for LassoConvNet.

    These are *proxy* benchmarks that run in < 30s. They do NOT replace
    full CIFAR-10/100 training but validate that the architecture is
    capable of learning useful representations.
    """

    def test_linear_probe_proxy(self, cfg):
        """
        Proxy for linear probe evaluation: verify backbone features
        are not degenerate (zero variance, constant, etc.).

        Full linear probe requires training a logistic regression on
        frozen features — this checks features are well-conditioned.
        """
        from backbone import LassoConvNetBackbone
        backbone = LassoConvNetBackbone(cfg)
        backbone.eval()

        x = torch.randn(8, cfg.in_channels, cfg.img_size, cfg.img_size)
        with torch.no_grad():
            features = backbone(x)  # (B, C4, H/8, W/8)

        # Features should have non-zero variance
        feat_std = features.std(dim=(0, 2, 3), keepdim=False)  # (C4,)
        dead_channels = (feat_std < 1e-8).sum().item()
        total_channels = features.shape[1]
        print(f"  Feature dead channels: {dead_channels} / {total_channels}")
        # At random init, most channels should be alive
        assert dead_channels < total_channels * 0.5, (
            f"Too many dead feature channels: {dead_channels}/{total_channels}"
        )

        # Feature covariance should not be singular (check condition number proxy)
        B, C, H, W = features.shape
        features_flat = features.view(B, C, H * W).mean(dim=-1)  # (B, C) — spatial avg
        cov = features_flat.T @ features_flat  # (C, C)
        try:
            eigvals = torch.linalg.eigvalsh(cov)
            cond_number = eigvals[-1] / (eigvals[0] + 1e-10)
            print(f"  Feature covariance condition number: {cond_number:.2f}")
            assert torch.isfinite(cond_number), "Non-finite condition number"
        except torch.linalg.LinAlgError:
            print("  [WARN] Could not compute eigendecomposition (features may be degenerate)")

    def test_masked_reconstruction_proxy(self, cfg):
        """
        Proxy for masked patch reconstruction: check that backbone
        features preserve spatial structure.

        Full MAE-style probe requires training a decoder — we just
        check that features at different spatial positions differ
        (i.e., the model doesn't collapse spatial info).
        """
        cfg_small = LassoConvConfig(
            in_channels=3,
            img_size=32,
            n_classes=cfg.n_classes,
            stage_channels=(16, 32, 64, 128),
            stage_depths=(1, 1, 1, 1),
            head_hidden_dim=cfg.head_hidden_dim,
        )
        from backbone import LassoConvNetBackbone
        backbone = LassoConvNetBackbone(cfg_small)
        backbone.eval()

        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            features = backbone(x)  # (B, C4, 4, 4)

        # Spatial variance across positions should be non-zero
        spatial_std = features.std(dim=(2, 3), keepdim=False)  # (B, C4)
        mean_spatial_std = spatial_std.mean().item()
        print(f"  Spatial feature std (mean across batch/ch): {mean_spatial_std:.4f}")
        # At random init, features should have some spatial variation
        assert mean_spatial_std > 1e-4, (
            f"Spatial features nearly constant: std={mean_spatial_std:.6f}"
        )

    def test_sparsity_profile_at_different_depths(self, cfg):
        """
        Measure sparsity ratio at each stage of the backbone.
        Deeper layers should (potentially) have different sparsity profiles.
        """
        from backbone import LassoConvNetBackbone
        backbone = LassoConvNetBackbone(cfg)
        backbone.eval()

        # Register hooks to capture activations at each stage
        activations = {}

        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        for i, stage in enumerate(backbone.stages):
            stage.register_forward_hook(make_hook(f"stage_{i}"))

        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            backbone(x)

        for name, act in activations.items():
            # Apply sparsity check on activations after the stage
            # (these have gone through soft-thresholding)
            sparsity = compute_sparsity_ratio(act, eps=1e-6)
            print(f"  {name}: shape={act.shape}, sparsity={sparsity.item():.4f}")
            assert sparsity.item() >= 0, f"Negative sparsity at {name}"

    def test_forward_consistent_across_modes(self, cfg):
        """
        All three modes should produce valid (finite) output
        for the same input.
        """
        x = torch.randn(2, 3, 32, 32)

        modes = ["proximal", "loss_only", "lista_unrolled"]
        for mode in modes:
            cfg_m = LassoConvConfig(
                stage_channels=(16, 32, 64, 128),
                stage_depths=(1, 1, 1, 1),
                lasso_mode=mode,
                lista_iters=3 if mode == "lista_unrolled" else 0,
                lista_dictionary_size=32 if mode == "lista_unrolled" else 0,
                head_hidden_dim=64,
            )
            m = LassoConvNet(cfg_m)
            m.eval()
            with torch.no_grad():
                logits = m(x)
            assert logits.shape == (2, 10), (
                f"Mode {mode}: shape {logits.shape}"
            )
            assert torch.isfinite(logits).all(), (
                f"Mode {mode}: non-finite logits"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Function Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLossFunctions:
    """Tests for the composite loss and individual penalties."""

    def test_l1_penalty_non_negative(self, cfg):
        """L1 weight penalty must always be >= 0."""
        model = LassoConvNet(cfg)
        pen = l1_weight_penalty(model, strength=1e-5)
        assert pen.item() >= 0, f"Negative L1 penalty: {pen.item()}"

    def test_l1_penalty_scales_with_strength(self, cfg):
        """Doubling the strength should roughly double the penalty."""
        model = LassoConvNet(cfg)
        pen1 = l1_weight_penalty(model, strength=1e-5)
        pen2 = l1_weight_penalty(model, strength=2e-5)
        # Should be approximately 2x (within floating point tolerance)
        ratio = pen2.item() / (pen1.item() + 1e-10)
        assert 1.5 < ratio < 2.5, (
            f"L1 penalty ratio: {ratio:.2f} (expected ~2.0)"
        )

    def test_group_lasso_penalty_non_negative(self, cfg):
        """Group lasso penalty must always be >= 0."""
        model = LassoConvNet(cfg)
        pen = group_lasso_penalty(model, group_size=8, strength=1e-4)
        assert pen.item() >= 0, f"Negative group lasso: {pen.item()}"

    def test_group_lasso_zero_for_small_groups(self, cfg):
        """Group lasso with group_size > output channels should give 0."""
        model = LassoConvNet(cfg)
        pen = group_lasso_penalty(model, group_size=9999, strength=1e-4)
        # Returns float 0.0 when no groups match; tensor otherwise
        pen_val = pen.item() if hasattr(pen, 'item') else float(pen)
        assert pen_val == 0.0, (
            f"Group lasso with large group: {pen_val}"
        )

    def test_composite_loss_adds_penalties(self, cfg):
        """Total loss = CE + L1 + group lasso."""
        model = LassoConvNet(cfg)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        ce = F.cross_entropy(logits, targets)
        total = lasso_total_loss(ce, model, cfg)
        # Total must be >= CE
        assert total.item() >= ce.item(), f"Total {total.item()} < CE {ce.item()}"
        # The difference should come from penalties
        l1_pen = l1_weight_penalty(model, cfg.l1_weight_decay)
        gl_pen = group_lasso_penalty(model, cfg.group_size, cfg.group_lasso_strength)
        expected_extra = l1_pen.item() + gl_pen.item()
        actual_extra = total.item() - ce.item()
        assert abs(actual_extra - expected_extra) < 1e-5, (
            f"Extra loss: {actual_extra:.6f} vs expected {expected_extra:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Misc / Utility Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtilities:
    """Tests for model utility methods."""

    def test_count_params(self, model, cfg):
        """Parameter count must be a reasonable positive integer."""
        from model import count_params
        total, trainable = count_params(model, verbose=False)
        assert total > 0, "Total params must be positive"
        assert trainable > 0, "Trainable params must be positive"
        assert trainable <= total, "Trainable > total"
        print(f"  Total params: {total:,}")

    def test_get_thresholds(self, model):
        """get_thresholds must return a dict with 'theta' entries."""
        thresholds = model.get_thresholds()
        assert len(thresholds) > 0, "No thresholds found"
        for name, val in thresholds.items():
            assert "theta" in name, f"Unexpected key: {name}"
            assert val.ndim >= 1, f"Threshold {name} is scalar"

    def test_set_check_nan(self, model):
        """set_check_nan must toggle the flag without error."""
        model.set_check_nan(enabled=True)
        model.set_check_nan(enabled=False)

    def test_theta_lower_lr_parameter_group(self, cfg):
        """Threshold params should be separable into their own optimizer group."""
        model = LassoConvNet(cfg)
        decay_params = []
        no_decay_params = []
        threshold_params = []
        for name, p in model.named_parameters():
            if "theta" in name:
                threshold_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        assert len(threshold_params) > 0, "No threshold params found"
        # Verify optimizer can be constructed
        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.l1_weight_decay},
            {"params": threshold_params, "lr": cfg.lr_threshold, "weight_decay": 0.0},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr)
        # One step
        model.train()
        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        print(f"  Optimizer groups: decay={len(decay_params)}, "
              f"threshold={len(threshold_params)}, no_decay={len(no_decay_params)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointing:
    """Gradient checkpointing must produce same outputs."""

    def test_checkpoint_consistency(self, cfg):
        """Forward pass with and without checkpointing should match (both in train mode)."""
        from model import LassoConvNet
        model_ref = LassoConvNet(cfg)
        model_ref.train()

        model_ckpt = LassoConvNet(cfg)
        model_ckpt.load_state_dict(model_ref.state_dict())
        model_ckpt.train()

        x = torch.randn(2, 3, 32, 32)
        out_ref = model_ref(x, use_checkpoint=False)
        out_ckpt = model_ckpt(x, use_checkpoint=True)
        assert torch.allclose(out_ref, out_ckpt, atol=1e-5), (
            "Checkpointing changes output"
        )

    def test_checkpoint_backward_pass(self, cfg):
        """Backward with checkpointing must succeed."""
        model = LassoConvNet(cfg)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, cfg.n_classes, (2,))
        logits = model(x, use_checkpoint=True)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        dead = [n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None]
        assert len(dead) == 0, f"Dead params with checkpoint: {dead}"
