#!/usr/bin/env python3
"""
LassoConvNet — Smoke Test.

Validates the complete model:
    1. Forward pass with shape assertions (proximal + loss_only + lista modes)
    2. Parameter count reporting
    3. Sparsity ratio measurement
    4. Group lasso penalty computation
    5. Gradient flow check (backward pass)
    6. bf16 inference (if GPU available)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import sys
import os

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LassoConvConfig
from model import LassoConvNet, count_params
from loss import lasso_total_loss, group_lasso_penalty, l1_weight_penalty
from layers import (
    SoftThreshold,
    AdaptiveSoftThreshold,
    HardThreshold,
    compute_sparsity_ratio,
    compute_channel_sparsity,
    build_prox_operator,
)


def test_config() -> None:
    """Verify the config dataclass instantiates with defaults."""
    cfg = LassoConvConfig()
    assert cfg.in_channels == 3
    assert cfg.n_classes == 10
    assert cfg.lasso_mode == "proximal"
    assert cfg.proximal_type == "soft"
    print("[OK] LassoConvConfig defaults")


def test_proximal_operators() -> None:
    """Test all proximal operators produce correct shapes and sparsity."""
    B, C, H, W = 2, 8, 16, 16
    x = torch.randn(B, C, H, W)
    theta = torch.full((1, C, 1, 1), 0.5)

    # Soft thresholding
    op = SoftThreshold()
    out = op(x, theta)
    assert out.shape == (B, C, H, W), f"SoftThreshold shape: {out.shape}"
    sparsity = compute_sparsity_ratio(out, eps=1e-4)
    print(f"[OK] SoftThreshold        output: {out.shape}, sparsity: {sparsity.item():.3f}")

    # Adaptive soft thresholding
    op = AdaptiveSoftThreshold(alpha=0.5)
    out = op(x, theta)
    assert out.shape == (B, C, H, W), f"AdaptiveSoftThreshold shape: {out.shape}"
    print(f"[OK] AdaptiveSoftThreshold output: {out.shape}")

    # Hard thresholding
    op = HardThreshold()
    out = op(x, theta)
    assert out.shape == (B, C, H, W), f"HardThreshold shape: {out.shape}"
    print(f"[OK] HardThreshold        output: {out.shape}")

    # Factory
    op = build_prox_operator("soft")
    assert isinstance(op, SoftThreshold)
    op = build_prox_operator("adaptive", alpha=0.2)
    assert isinstance(op, AdaptiveSoftThreshold)
    print("[OK] build_prox_operator factory")


def test_lasso_prox_block() -> None:
    """Test a single LassoProxConvBlock forward pass with shape checks."""
    from blocks import LassoProxConvBlock

    cfg = LassoConvConfig(
        in_channels=3,
        kernel_size=3,
        norm_type="batch_norm",
        threshold_init=0.01,
        threshold_learnable=True,
        proximal_type="soft",
        norm_before_prox=True,
        use_bias=False,
    )

    B, C, H, W = 2, 3, 32, 32
    x = torch.randn(B, C, H, W)

    # Same channels, no stride
    block = LassoProxConvBlock(in_channels=3, out_channels=3, stride=1, config=cfg)
    out = block(x)
    assert out.shape == (B, 3, H, W), f"Same-ch block: {out.shape}"
    print(f"[OK] LassoProxConvBlock (same ch, stride=1):  {out.shape}")

    # Channel doubling, stride 2
    block2 = LassoProxConvBlock(in_channels=3, out_channels=64, stride=2, config=cfg)
    out2 = block2(x)
    assert out2.shape == (B, 64, H // 2, W // 2), f"Stride-2 block: {out2.shape}"
    print(f"[OK] LassoProxConvBlock (3→64 ch, stride=2): {out2.shape}")

    # Gradient checkpointing
    block.train()
    out3 = block(x, use_checkpoint=True)
    assert out3.shape == (B, 3, H, W)
    print(f"[OK] LassoProxConvBlock (checkpoint=True):    {out3.shape}")

    # Adaptive variant
    cfg2 = LassoConvConfig(
        in_channels=3, kernel_size=3, norm_type="batch_norm",
        threshold_init=0.01, threshold_learnable=True,
        proximal_type="adaptive", norm_before_prox=True, use_bias=False,
    )
    block_ad = LassoProxConvBlock(in_channels=3, out_channels=3, stride=1, config=cfg2)
    out_ad = block_ad(x)
    assert out_ad.shape == (B, 3, H, W)
    print(f"[OK] LassoProxConvBlock (adaptive prox):      {out_ad.shape}")


def test_backbone() -> None:
    """Test the 4-stage pyramid backbone."""
    from backbone import LassoConvNetBackbone, LassoConvStage

    cfg = LassoConvConfig(
        in_channels=3,
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        kernel_size=3,
        threshold_init=0.01,
        threshold_learnable=True,
        proximal_type="soft",
        norm_before_prox=True,
        norm_type="batch_norm",
        use_bias=False,
        lasso_mode="proximal",
    )

    B = 2
    x = torch.randn(B, 3, 32, 32)

    backbone = LassoConvNetBackbone(cfg)
    out = backbone(x)
    expected_ch = cfg.stage_channels[-1]
    expected_hw = 32 // 8  # 3 downsampling stages: 32 → 16 → 8 → 4
    assert out.shape == (B, expected_ch, expected_hw, expected_hw), \
        f"Backbone output: {out.shape}, expected ({B}, {expected_ch}, {expected_hw}, {expected_hw})"
    print(f"[OK] LassoConvNetBackbone (proximal):        {out.shape}")


def test_full_model_proximal() -> None:
    """Test the full LassoConvNet in proximal mode."""
    cfg = LassoConvConfig(
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
    )

    model = LassoConvNet(cfg)
    model.eval()

    B, H, W = 2, 32, 32
    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (B, cfg.n_classes), \
        f"Proximal mode: {logits.shape}, expected ({B}, {cfg.n_classes})"
    print(f"[OK] LassoConvNet (proximal): logits = {logits.shape}")

    # Sparsity tracking
    sparsity = model.get_sparsity_ratio()
    assert sparsity is not None
    print(f"[OK] Activation sparsity ratio: {sparsity.item():.4f}")


def test_full_model_loss_only() -> None:
    """Test model in loss_only mode (no architectural sparsity)."""
    cfg = LassoConvConfig(
        in_channels=3,
        img_size=32,
        n_classes=10,
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        lasso_mode="loss_only",  # ReLU not replaced by soft-threshold
        norm_type="batch_norm",
        use_bias=False,
        head_hidden_dim=64,
        l1_weight_decay=1e-5,
    )

    model = LassoConvNet(cfg)
    model.eval()

    B = 2
    x = torch.randn(B, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (B, cfg.n_classes)
    print(f"[OK] LassoConvNet (loss_only): logits = {logits.shape}")


def test_full_model_lista() -> None:
    """Test model in lista_unrolled mode."""
    cfg = LassoConvConfig(
        in_channels=3,
        img_size=32,
        n_classes=10,
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        lasso_mode="lista_unrolled",
        lista_iters=3,
        lista_dictionary_size=32,
        lista_tie_weights=True,
        norm_type="batch_norm",
        use_bias=False,
        head_hidden_dim=64,
    )

    model = LassoConvNet(cfg)
    model.eval()

    B = 2
    x = torch.randn(B, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (B, cfg.n_classes), \
        f"LISTA mode: {logits.shape}, expected ({B}, {cfg.n_classes})"
    print(f"[OK] LassoConvNet (lista_unrolled): logits = {logits.shape}")


def test_loss_functions() -> None:
    """Test composite loss and penalty functions."""
    cfg = LassoConvConfig(
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        l1_weight_decay=1e-5,
        use_group_lasso=True,
        group_lasso_strength=1e-4,
        group_size=8,
        lasso_mode="proximal",
    )

    model = LassoConvNet(cfg)
    model.train()

    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, cfg.n_classes, (2,))

    # Forward
    logits = model(x)
    ce_loss = nn.CrossEntropyLoss()(logits, targets)

    # Composite loss
    total_loss = lasso_total_loss(ce_loss, model, cfg)
    assert total_loss.item() >= ce_loss.item(), \
        "Total loss should be >= CE loss (penalties are non-negative)"
    print(f"[OK] lasso_total_loss: CE={ce_loss.item():.4f}, "
          f"Total={total_loss.item():.4f}")

    # Individual penalties
    l1_pen = l1_weight_penalty(model, strength=cfg.l1_weight_decay)
    assert l1_pen.item() >= 0
    print(f"[OK] l1_weight_penalty: {l1_pen.item():.6f}")

    gl_pen = group_lasso_penalty(model, group_size=cfg.group_size, strength=cfg.group_lasso_strength)
    assert gl_pen.item() >= 0
    print(f"[OK] group_lasso_penalty: {gl_pen.item():.6f}")


def test_backward_pass() -> None:
    """Verify gradients flow through the entire model."""
    cfg = LassoConvConfig(
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        proximal_type="soft",
        lasso_mode="proximal",
    )

    model = LassoConvNet(cfg)
    model.train()

    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, cfg.n_classes, (2,))

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, targets)

    loss.backward()

    # Check that all parameters have gradients
    has_grad = False
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = True
            if param.grad is None:
                no_grad_params.append(name)
            elif param.grad.abs().sum().item() == 0:
                no_grad_params.append(name)

    assert has_grad, "No parameters with requires_grad=True"
    if no_grad_params:
        print(f"[WARN] Parameters with zero/no gradient: {len(no_grad_params)}")
        for p in no_grad_params[:5]:
            print(f"       {p}")
    else:
        print(f"[OK] All parameters received gradients")

    # Verify thresholds received gradients
    for name, param in model.named_parameters():
        if "theta" in name:
            assert param.grad is not None, f"Threshold {name} has no gradient"
            print(f"[OK] Threshold gradient ({name}): mean={param.grad.abs().mean().item():.6f}")

    print(f"[OK] Backward pass: loss={loss.item():.4f}")


def test_bf16_inference() -> None:
    """Test forward pass in bfloat16 (runs on CPU or GPU)."""
    cfg = LassoConvConfig(
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        proximal_type="soft",
        lasso_mode="proximal",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LassoConvNet(cfg).to(device=device, dtype=torch.bfloat16)
    model.eval()

    x = torch.randn(2, 3, 32, 32, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, cfg.n_classes), f"bf16: {logits.shape}"
    assert logits.dtype == torch.bfloat16, f"bf16 output dtype: {logits.dtype}"
    print(f"[OK] bf16 inference ({device}): logits = {logits.shape}")


def test_threshold_clamping() -> None:
    """Verify threshold values can be clamped."""
    cfg = LassoConvConfig(
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
        threshold_init=0.01,
        threshold_learnable=True,
    )

    model = LassoConvNet(cfg)

    # Set some thresholds to negative values
    for name, param in model.named_parameters():
        if "theta" in name:
            param.data.fill_(-0.1)

    # Clamp
    model.clamp_thresholds(min_val=1e-6)

    # Verify
    for name, param in model.named_parameters():
        if "theta" in name:
            assert (param >= 1e-6).all(), f"Threshold {name} not clamped"
    print(f"[OK] Threshold clamping: all thresholds ≥ 1e-6")


def test_zero_filter_counting() -> None:
    """Verify zero-filter counting works."""
    cfg = LassoConvConfig(
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
    )

    model = LassoConvNet(cfg)
    zero_filters = model.count_zero_filters(eps=1e-6)
    assert isinstance(zero_filters, int)
    print(f"[OK] Zero filter count: {zero_filters}")


def test_parameter_schedules() -> None:
    """Verify optimizer parameter groups can be created."""
    import math

    cfg = LassoConvConfig(
        stage_channels=(16, 32, 64, 128),
        stage_depths=(1, 1, 1, 1),
    )

    model = LassoConvNet(cfg)

    # Separate parameter groups:
    # Group 1: conv weights with L1 decay
    # Group 2: thresholds with lower LR
    # Group 3: biases and norm params — no weight decay
    decay_params = []
    no_decay_params = []
    threshold_params = []

    for name, param in model.named_parameters():
        if "theta" in name:
            threshold_params.append(param)
        elif param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.l1_weight_decay},
        {"params": threshold_params, "lr": cfg.lr_threshold, "weight_decay": 0.0},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr)

    # One step to verify it works
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, cfg.n_classes, (2,))
    model.train()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    optimizer.step()

    print(f"[OK] Parameter groups: decay={len(decay_params)}, "
          f"threshold={len(threshold_params)}, no_decay={len(no_decay_params)}")


def test_channel_sparsity() -> None:
    """Test per-channel sparsity measurement."""
    x = torch.randn(2, 8, 16, 16)
    x[:, 0, :, :] = 1e-8  # Make channel 0 very sparse
    x[:, 1, :, :] = 0.0   # Make channel 1 zero
    sparsity = compute_channel_sparsity(x, eps=1e-4)
    assert sparsity.shape == (8,)
    assert sparsity[0] > 0.5, f"Channel 0 sparsity: {sparsity[0]}"
    assert sparsity[1] == 1.0, f"Channel 1 sparsity: {sparsity[1]}"
    print(f"[OK] compute_channel_sparsity: {sparsity.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("LassoConvNet — Smoke Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    tests = [
        ("Config", test_config),
        ("Proximal Operators", test_proximal_operators),
        ("Channel Sparsity", test_channel_sparsity),
        ("LassoProxConvBlock", test_lasso_prox_block),
        ("Backbone", test_backbone),
        ("Full Model (proximal)", test_full_model_proximal),
        ("Full Model (loss_only)", test_full_model_loss_only),
        ("Full Model (lista)", test_full_model_lista),
        ("Loss Functions", test_loss_functions),
        ("Backward Pass", test_backward_pass),
        ("Threshold Clamping", test_threshold_clamping),
        ("Zero Filter Counting", test_zero_filter_counting),
        ("Parameter Schedules", test_parameter_schedules),
        ("bf16 Inference", test_bf16_inference),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed / {len(tests)} total")
    print("=" * 60)

    # Print parameter count for a reasonable-sized model
    print()
    print("Model parameter summary (CIFAR-10 scale):")
    cfg = LassoConvConfig(
        stage_channels=(64, 128, 256, 512),
        stage_depths=(2, 2, 2, 2),
        proximal_type="soft",
        lasso_mode="proximal",
        threshold_learnable=True,
    )
    model = LassoConvNet(cfg)
    count_params(model)
    print()

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
