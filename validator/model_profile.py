#!/usr/bin/env python3
"""
LassoConvNet — Profiling Script.

Runs torch.profiler on the model to measure:
- Memory usage (per-operator)
- Runtime (CPU/CUDA)
- Estimated FLOPs

Usage:
    python profile.py --mode forward     # inference profiling
    python profile.py --mode train       # forward + backward
    python profile.py --mode compare     # compare all three modes
    python profile.py --mode listamem    # memory profile LISTA vs proximal
"""

from __future__ import annotations

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "coder"))

from config import LassoConvConfig
from model import LassoConvNet, count_params


def build_sample_input(cfg: LassoConvConfig, device: str = "cpu"):
    """Build a sample input tensor matching config."""
    return torch.randn(2, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)


def profile_model(
    model: nn.Module,
    cfg: LassoConvConfig,
    mode: str = "forward",
    steps: int = 10,
    device: str = "cpu",
    profile_memory: bool = True,
):
    """
    Profile the model's forward (or forward+backward) pass.

    mode: "forward" — inference only (~2× params FLOPs)
          "train"   — forward + backward (~6× params FLOPs, Kaplan et al.)
    """
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sort_key = "cuda_memory_usage" if (device == "cuda" and profile_memory) else "self_cpu_time_total"
    model = model.to(device)
    model.train() if mode == "train" else model.eval()

    sample = build_sample_input(cfg, device)
    targets = torch.randint(0, cfg.n_classes, (2,), device=device)

    print(f"Profiling: mode={mode}, device={device}, steps={steps}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            with record_function(f"step_{mode}"):
                logits = model(sample)
                if mode == "train":
                    loss = nn.CrossEntropyLoss()(logits, targets)
                    loss.backward()
                    # Zero gradients for next step to avoid accumulation
                    model.zero_grad(set_to_none=True)

    print(f"\n{'=' * 80}")
    print(f"  Profile by {sort_key}")
    print(f"{'=' * 80}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    # FLOP estimate
    params = sum(p.numel() for p in model.parameters())
    mult = 6 if mode == "train" else 2
    flops_est = mult * params
    print(f"\n  Estimated {mode} FLOPs: {flops_est / 1e6:.1f}M  ({mult}× params = {params:,})")
    print(f"  Estimated {mode} FLOPs: {flops_est / 1e9:.2f}G\n")

    # Self CPU time total
    self_cpu = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5)
    print(f"\n  Top-5 by self CPU time:")
    print(self_cpu)

    return prof


def profile_all_modes(device: str = "cpu", steps: int = 5):
    """Profile all three lasso modes and compare."""
    base_cfg = LassoConvConfig(
        stage_channels=(64, 128, 256, 512),
        stage_depths=(2, 2, 2, 2),
        head_hidden_dim=256,
    )

    modes = {
        "proximal": LassoConvConfig(
            lasso_mode="proximal",
            proximal_type="soft",
            threshold_learnable=True,
            **{k: getattr(base_cfg, k) for k in
               ["stage_channels", "stage_depths", "head_hidden_dim"]},
        ),
        "loss_only": LassoConvConfig(
            lasso_mode="loss_only",
            **{k: getattr(base_cfg, k) for k in
               ["stage_channels", "stage_depths", "head_hidden_dim"]},
        ),
        "lista_unrolled": LassoConvConfig(
            lasso_mode="lista_unrolled",
            lista_iters=6,
            lista_dictionary_size=256,
            lista_tie_weights=True,
            **{k: getattr(base_cfg, k) for k in
               ["stage_channels", "stage_depths", "head_hidden_dim"]},
        ),
    }

    results = {}
    for name, cfg in modes.items():
        print(f"\n{'#' * 70}")
        print(f"# Mode: {name}")
        print(f"{'#' * 70}")
        model = LassoConvNet(cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

        prof = profile_model(model, cfg, mode="forward", steps=steps, device=device)
        results[name] = {"params": params}

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  Mode Comparison Summary")
    print(f"{'=' * 70}")
    print(f"  {'Mode':<20} {'Params':<12}")
    print(f"  {'-'*20} {'-'*12}")
    for name, r in results.items():
        print(f"  {name:<20} {r['params']:>10,}")

    return results


def profile_memory_breakdown(device: str = "cpu"):
    """Profile memory usage of key components."""
    cfg = LassoConvConfig(
        stage_channels=(64, 128, 256, 512),
        stage_depths=(2, 2, 2, 2),
        head_hidden_dim=256,
    )

    model = LassoConvNet(cfg).to(device)
    sample = build_sample_input(cfg, device)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"\n{'=' * 70}")
    print("  Memory Profile (per-operator)")
    print(f"{'=' * 70}")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            with record_function("full_forward"):
                _ = model(sample)

    # Filter to show only memory-related stats
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage" if device == "cuda" else "self_cpu_memory_usage",
        row_limit=15,
    ))

    # Total memory
    if device == "cuda":
        print(f"\n  CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="LassoConvNet — Profiling"
    )
    parser.add_argument(
        "--mode",
        default="forward",
        choices=["forward", "train", "compare", "memory"],
        help="Profiling mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of profiling steps",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu / cuda)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory profiling",
    )
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")

    if args.mode == "compare":
        profile_all_modes(device=device, steps=args.steps)
    elif args.mode == "memory":
        profile_memory_breakdown(device=device)
    else:
        cfg = LassoConvConfig(
            stage_channels=(64, 128, 256, 512),
            stage_depths=(2, 2, 2, 2),
            head_hidden_dim=256,
        )
        model = LassoConvNet(cfg)
        profile_model(
            model, cfg,
            mode=args.mode,
            steps=args.steps,
            device=device,
            profile_memory=not args.no_memory,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
