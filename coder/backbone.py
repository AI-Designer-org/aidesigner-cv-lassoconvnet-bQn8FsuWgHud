"""
LassoConvNet — Backbone Architecture.

4-stage ResNet-style pyramid where every convolutional block uses
soft-thresholding (the Lasso proximal operator) instead of (or in
addition to) the standard ReLU activation.

The backbone progressively downsamples spatial dimensions while
increasing channel dimensions, producing a multi-scale feature
pyramid suitable for classification.

Spatial dimension progression (for input H × W):
    Stage 1:  H × W      (no downsampling)
    Stage 2:  H/2 × W/2  (first block stride 2)
    Stage 3:  H/4 × W/4  (first block stride 2)
    Stage 4:  H/8 × W/8  (first block stride 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from config import LassoConvConfig
from blocks import LassoProxConvBlock, ListaEncoder
from layers import BaseProxOperator


# ═══════════════════════════════════════════════════════════════════════════
# Stage
# ═══════════════════════════════════════════════════════════════════════════

class LassoConvStage(nn.Module):
    """
    One spatial stage of the LassoConvNet pyramid.

    Contains N LassoProxConvBlocks. The first block optionally
    downsamples via stride-2 convolution (or pooling).

    Architecture:
        [LassoProxConvBlock] × depth
        first block uses stride=2 if downsample_first=True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample_first: bool,
        config: LassoConvConfig,
    ):
        super().__init__()

        blocks = []
        for i in range(depth):
            stride = 2 if (i == 0 and downsample_first) else 1
            in_ch = in_channels if i == 0 else out_channels

            block = LassoProxConvBlock(
                in_channels=in_ch,
                out_channels=out_channels,
                stride=stride,
                config=config,
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        Returns: (B, out_channels, H_out, W_out)

        H_out = H if no downsampling, else H/2
        """
        h = x
        for block in self.blocks:
            h = block(h, use_checkpoint=use_checkpoint)
        return h


# ═══════════════════════════════════════════════════════════════════════════
# Full Backbone
# ═══════════════════════════════════════════════════════════════════════════

class LassoConvNetBackbone(nn.Module):
    """
    Full 4-stage pyramid backbone with lasso-integrated convolutions.

    Architecture:
        Stem: 3×3 conv → BN → ReLU          (B, C1, H, W)
        Stage 1: depth[0] blocks, C1 ch      (B, C1, H, W)
        Stage 2: depth[1] blocks, C2 ch      (B, C2, H/2, W/2)
        Stage 3: depth[2] blocks, C3 ch      (B, C3, H/4, W/4)
        Stage 4: depth[3] blocks, C4 ch      (B, C4, H/8, W/8)

    Each block's activation function is the Lasso proximal operator
    (soft-thresholding) with per-channel learnable thresholds.
    """

    def __init__(self, config: LassoConvConfig):
        super().__init__()
        self.config = config

        C1, C2, C3, C4 = config.stage_channels

        # ── Stem: initial feature extraction ──
        # (B, in_ch, H, W) → (B, C1, H, W)  [no downsampling for CIFAR]
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                C1,
                kernel_size=3,
                stride=1,        # no stride for small inputs (CIFAR)
                padding=1,
                bias=config.use_bias,
            ),
            nn.BatchNorm2d(C1),
            nn.ReLU(),           # standard ReLU in stem; lasso blocks start after
        )  # (B, C1, H, W)

        # ── Stages ──
        self.stages = nn.ModuleList()
        in_ch = C1

        for i, (out_ch, depth) in enumerate(
            zip(config.stage_channels, config.stage_depths)
        ):
            # Downsample at start of every stage except the first
            downsample_first = (i > 0)

            stage = LassoConvStage(
                in_channels=in_ch,
                out_channels=out_ch,
                depth=depth,
                downsample_first=downsample_first,
                config=config,
            )
            self.stages.append(stage)
            in_ch = out_ch

        # ── LISTA encoder (replaces pyramid in lista_unrolled mode) ──
        self.lista_encoder = (
            ListaEncoder(config)
            if config.lasso_mode == "lista_unrolled"
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        Returns:
            proximal mode:     (B, C4, H/8, W/8)  — final feature map
            lista_unrolled mode: (B, dict_size, H, W) — sparse code
        """
        # ── Stem ──
        h = self.stem(x)                                   # (B, C1, H, W)

        # ── Feature extraction ──
        if self.lista_encoder is not None:
            # LISTA mode: iterative sparse coding refinement
            h = self.lista_encoder(h)                      # (B, dict_size, H, W)
        else:
            # Standard mode: hierarchical pyramid
            for stage in self.stages:
                h = stage(h, use_checkpoint=use_checkpoint)
                # After stage 1: (B, C1, H, W)
                # After stage 2: (B, C2, H/2, W/2)
                # After stage 3: (B, C3, H/4, W/4)
                # After stage 4: (B, C4, H/8, W/8)

        return h


# ═══════════════════════════════════════════════════════════════════════════
# Downsampling helpers (for "pool" mode)
# ═══════════════════════════════════════════════════════════════════════════

def make_downsample_layer(
    mode: str,
    in_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Module:
    """
    Create a downsampling layer.

    For "stride" mode: 1×1 conv with stride
    For "pool" mode:   2×2 avg pool + 1×1 conv (if channel change needed)
    """
    if mode == "stride":
        return nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, bias=False,
        )
    elif mode == "pool":
        layers = [nn.AvgPool2d(kernel_size=2, stride=stride)]
        if in_channels != out_channels:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        return nn.Sequential(*layers)
    else:
        raise ValueError(f"Unknown downsample mode: {mode}")
