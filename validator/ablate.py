#!/usr/bin/env python3
"""
LassoConvNet — Ablation Runner.

Runs single-field config changes to measure the impact of each architectural
decision on accuracy, sparsity, and parameter count.

Defines named ablations from the architect's proposed A1-A6 list:

    A1: lasso_mode proximal → loss_only
    A2: threshold_learnable True → False
    A3: use_group_lasso True → False
    A4: proximal_type "soft" → "adaptive"
    A5: lista_tie_weights True → False (LISTA mode only)
    A6: norm_before_prox True → False

Usage:
    # Dry run (measure sparsity + param count, no training):
    python ablate.py --dry-run

    # Full training run (requires train.py with CIFAR-10):
    python ablate.py --data cifar10 --epochs 50
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import os
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))

from config import LassoConvConfig
from model import LassoConvNet, count_params
from layers import compute_sparsity_ratio
from loss import lasso_total_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Definitions
# ═══════════════════════════════════════════════════════════════════════════════

def _base_config() -> LassoConvConfig:
    """Base config for CIFAR-10 ablation experiments."""
    return LassoConvConfig(
        # Architecture (CIFAR-10 scale)
        in_channels=3,
        img_size=32,
        n_classes=10,
        base_channels=64,
        stage_channels=(64, 128, 256, 512),
        stage_depths=(2, 2, 2, 2),
        kernel_size=3,
        downsample="stride",
        # Lasso mode (default for comparison)
        lasso_mode="proximal",
        proximal_type="soft",
        threshold_init=0.01,
        threshold_learnable=True,
        norm_before_prox=True,
        norm_type="batch_norm",
        # Loss terms
        l1_weight_decay=1e-5,
        l2_weight_decay=0.0,
        use_group_lasso=True,
        group_lasso_strength=1e-4,
        group_size=8,
        activation_sparsity_weight=1e-4,
        l1_decay_schedule="constant",
        # Regularization
        dropout=0.0,
        use_bias=False,
        # Head
        head_hidden_dim=256,
        global_pool="avg",
        # Training
        lr=1e-3,
        lr_threshold=1e-4,
        l1_warmup_epochs=5,
        theta_min=1e-6,
    )


ABLATIONS: Dict[str, LassoConvConfig] = {}

def _ablation(name: str, cfg: LassoConvConfig, desc: str) -> LassoConvConfig:
    """Create an ablation entry with a description."""
    cfg.desc = desc  # type: ignore[attr-defined]
    return cfg

ABLATIONS["A0_baseline"] = _ablation("baseline", _base_config(),
    "Baseline: proximal mode with all features enabled")

ABLATIONS["A1_loss_only"] = _ablation("loss_only", replace(
    _base_config(),
    lasso_mode="loss_only",
    activation_sparsity_weight=1e-4,
), "Swap architectural prox → loss-only L1 penalty")

ABLATIONS["A2_fixed_threshold"] = _ablation("fixed_threshold", replace(
    _base_config(),
    threshold_learnable=False,
), "Global fixed threshold (θ=0.01, not per-channel learnable)")

ABLATIONS["A3_no_group_lasso"] = _ablation("no_group_lasso", replace(
    _base_config(),
    use_group_lasso=False,
    group_lasso_strength=0.0,
), "Remove group-lasso structured sparsity penalty")

ABLATIONS["A4_adaptive_prox"] = _ablation("adaptive_prox", replace(
    _base_config(),
    proximal_type="adaptive",
), "Adaptive sigmoid-gated threshold instead of soft-threshold")

ABLATIONS["A6_no_preprox_norm"] = _ablation("no_preprox_norm", replace(
    _base_config(),
    norm_before_prox=False,
), "Remove normalization before soft-thresholding")

# LISTA-specific ablations (requires lista_unrolled mode)
ABLATIONS_LISTA: Dict[str, LassoConvConfig] = {}

ABLATIONS_LISTA["A5_lista_tied"] = _ablation("lista_tied", replace(
    _base_config(),
    lasso_mode="lista_unrolled",
    lista_iters=6,
    lista_dictionary_size=256,
    lista_tie_weights=True,
), "LISTA with tied weights across iterations")

ABLATIONS_LISTA["A5_lista_untied"] = _ablation("lista_untied", replace(
    _base_config(),
    lasso_mode="lista_unrolled",
    lista_iters=6,
    lista_dictionary_size=256,
    lista_tie_weights=False,
), "LISTA with untied weights per iteration")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_sparsity(model: LassoConvNet, x: torch.Tensor) -> Dict[str, float]:
    """
    Measure activation sparsity across the model.

    Returns:
        dict with keys: 'overall_sparsity', 'channel_sparsity_mean',
                        'channel_sparsity_std', 'zero_filter_count'
    """
    _ = model(x)
    sparsity = model.get_sparsity_ratio(eps=1e-6)
    zero_filters = model.count_zero_filters(eps=1e-6)

    results = {
        "overall_sparsity": sparsity.item() if sparsity is not None else -1.0,
        "zero_filter_count": zero_filters,
    }

    # Channel sparsity: skip if hook captured 2D tensor (classifier head ReLU)
    try:
        ch_sparsity = model.get_channel_sparsity(eps=1e-6)
        if ch_sparsity is not None:
            results["channel_sparsity_mean"] = ch_sparsity.mean().item()
            results["channel_sparsity_std"] = ch_sparsity.std().item()
    except (ValueError, RuntimeError):
        # Hook may capture non-4D tensors; measure sparsity from backbone directly
        from backbone import LassoConvNetBackbone
        backbone = LassoConvNetBackbone(model.config)
        backbone.load_state_dict(
            {k.replace("backbone.", ""): v
             for k, v in model.state_dict().items()
             if k.startswith("backbone.")}, strict=False)
        backbone.eval()
        feats = backbone(x)
        # feats is (B, C, H, W) — compute per-channel sparsity directly
        B, C, H, W = feats.shape
        feat_flat = feats.view(B, C, -1)
        ch_sp = (torch.abs(feat_flat) < 1e-6).float().mean(dim=(0, 2))
        results["channel_sparsity_mean"] = ch_sp.mean().item()
        results["channel_sparsity_std"] = ch_sp.std().item()

    return results


@torch.no_grad()
def measure_accuracy(
    model: LassoConvNet,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute top-1 accuracy on a data loader.
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    losses = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        losses.append(loss.cpu())
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    losses = torch.cat(losses)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "loss_mean": losses.mean().item(),
        "loss_std": losses.std().item(),
    }


def compute_model_stats(model: LassoConvNet, config: LassoConvConfig) -> Dict:
    """Compute parameter count and summary statistics."""
    total_params, trainable_params = count_params(model, verbose=False)

    # Count threshold parameters
    theta_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "theta" in n
    )

    # Count conv parameters
    conv_params = sum(
        p.numel() for m in model.modules()
        for p in m.parameters(recurse=False)
        if isinstance(m, nn.Conv2d)
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "theta_params": theta_params,
        "conv_params": conv_params,
        "n_blocks": sum(
            1 for _ in model.modules()
            if isinstance(_, nn.Conv2d)
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Ablation Runner
# ═══════════════════════════════════════════════════════════════════════════════

def dry_run(device: str = "cpu") -> Dict[str, Dict]:
    """
    Dry-run ablations: instantiate each config, measure sparsity and params.

    Does NOT train — useful to verify all configs are valid and to
    get initial sparsity measurements.
    """
    results = {}

    x = torch.randn(16, 3, 32, 32, device=device)

    all_configs = {}
    all_configs.update(ABLATIONS)
    all_configs.update(ABLATIONS_LISTA)

    for name, cfg in all_configs.items():
        print(f"\n{'=' * 60}")
        print(f"  Ablation: {name}")
        print(f"  Description: {getattr(cfg, 'desc', 'N/A')}")
        print(f"{'=' * 60}")

        try:
            model = LassoConvNet(cfg).to(device)
            model.eval()

            stats = compute_model_stats(model, cfg)
            sparsity = measure_sparsity(model, x)

            entry = {
                "config": {
                    "lasso_mode": cfg.lasso_mode,
                    "proximal_type": cfg.proximal_type,
                    "threshold_learnable": cfg.threshold_learnable,
                    "norm_before_prox": cfg.norm_before_prox,
                    "use_group_lasso": cfg.use_group_lasso,
                    "l1_weight_decay": cfg.l1_weight_decay,
                },
                "stats": stats,
                "sparsity": sparsity,
            }

            print(f"  Parameters:     {stats['total_params']:>10,}")
            print(f"  θ parameters:   {stats['theta_params']:>10,}")
            print(f"  Overall sparsity: {sparsity['overall_sparsity']:.4f}")
            print(f"  Zero filters:     {sparsity['zero_filter_count']}")
            print(f"  Channel sparsity:  μ={sparsity.get('channel_sparsity_mean', 0):.4f}, "
                  f"σ={sparsity.get('channel_sparsity_std', 0):.4f}")

            results[name] = entry

        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LassoConvNet — Ablation Runner"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run ablations without training (measure params + sparsity only)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu / cuda)",
    )
    parser.add_argument(
        "--output",
        default="ablation_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--ablation",
        default=None,
        help="Run a single ablation by name (default: all)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to CIFAR-10 data (for training-mode accuracy eval)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (for training mode)",
    )

    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Select ablation set
    if args.ablation:
        all_configs = {}
        if args.ablation in ABLATIONS:
            all_configs[args.ablation] = ABLATIONS[args.ablation]
        elif args.ablation in ABLATIONS_LISTA:
            all_configs[args.ablation] = ABLATIONS_LISTA[args.ablation]
        elif args.ablation == "all_lista":
            all_configs.update(ABLATIONS_LISTA)
        else:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {list(ABLATIONS.keys()) + list(ABLATIONS_LISTA.keys())}")
            return 1
    else:
        all_configs = {}
        all_configs.update(ABLATIONS)

    if args.dry_run:
        results = dry_run(device=device)
        # Also list ALL configs (including LISTA) in output
        lista_results = {}
        for name, cfg in ABLATIONS_LISTA.items():
            print(f"\n{'=' * 60}")
            print(f"  Ablation: {name}")
            print(f"  Description: {getattr(cfg, 'desc', 'N/A')}")
            print(f"{'=' * 60}")
            try:
                model = LassoConvNet(cfg).to(device)
                model.eval()
                x = torch.randn(16, 3, 32, 32, device=device)
                stats = compute_model_stats(model, cfg)
                sparsity = measure_sparsity(model, x)
                lista_results[name] = {
                    "config": {"lasso_mode": cfg.lasso_mode, "lista_tie_weights": cfg.lista_tie_weights},
                    "stats": stats,
                    "sparsity": sparsity,
                }
                print(f"  Parameters: {stats['total_params']:>10,}")
                print(f"  Overall sparsity: {sparsity['overall_sparsity']:.4f}")
            except Exception as e:
                print(f"  [FAIL] {e}")
                lista_results[name] = {"error": str(e)}
        results["lista_ablations"] = lista_results

        # Save
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

        # Summary table
        print("\n\n" + "=" * 60)
        print("  Ablation Summary")
        print("=" * 60)
        print(f"  {'Name':<25} {'Params':<12} {'Sparsity':<10} {'Zero Filt':<10}")
        print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10}")
        for name, entry in results.items():
            if "error" in entry:
                continue
            s = entry.get("sparsity", {})
            st = entry.get("stats", {})
            print(f"  {name:<25} {st.get('total_params', 0):>10,}  "
                  f"{s.get('overall_sparsity', 0):.4f}     "
                  f"{s.get('zero_filter_count', 0):>5}")

    else:
        print("Training mode not yet implemented.")
        print("To run ablations with training, implement a train() function")
        print("that takes (model, config, data_loader) and returns a trained model.")
        print("Example training loop structure:")
        print("""
    for name, cfg in all_configs.items():
        model = LassoConvNet(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        for epoch in range(args.epochs):
            for x, y in train_loader:
                logits = model(x)
                loss = lasso_total_loss(
                    F.cross_entropy(logits, y), model, cfg
                )
                loss.backward()
                optimizer.step()
                model.clamp_thresholds()
        acc = measure_accuracy(model, test_loader, device)
        sparsity = measure_sparsity(model, x_sample)
        results[name] = {"accuracy": acc, "sparsity": sparsity}
        """)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
