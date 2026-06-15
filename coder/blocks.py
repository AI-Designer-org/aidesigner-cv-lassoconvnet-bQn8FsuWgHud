"""
LassoConvNet — Core Blocks: LassoProxConvBlock and LISTA Encoder.

The LassoProxConvBlock is the core novel architectural contribution:
a standard conv-bn block with soft-thresholding (the Lasso proximal
operator) integrated into the forward pass instead of (or in addition to)
a loss-based L1 penalty.

Architecture (pre-norm residual style):
    x → Conv2d → BatchNorm → SoftThreshold(θ_c) → + residual → out
                              ↑
                      Lasso proximal operator
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

from config import LassoConvConfig
from layers import (
    BaseProxOperator,
    SoftThreshold,
    build_prox_operator,
    get_normalization_layer,
)


# ═══════════════════════════════════════════════════════════════════════════
# Core Novel Block: LassoProxConvBlock
# ═══════════════════════════════════════════════════════════════════════════

class LassoProxConvBlock(nn.Module):
    """
    Convolutional block with integrated Lasso proximal operator.

    Forward pass:
        1. Conv2d(x, stride, padding)           — spatial feature extraction
        2. BatchNorm/LayerNorm                   — stabilize scale before threshold
        3. SoftThreshold(z, θ_c)                 — ← LASSO PROXIMAL OPERATOR
        4. + residual shortcut                   — gradient flow around dead zone

    The soft-thresholding operator is the proximal map for L1 regularization:
        prox_{θ·||·||₁}(x) = sign(x) · max(|x| - θ, 0)

    Key design features:
    - Per-channel learnable thresholds θ of shape (1, C, 1, 1)
    - Pre-norm: normalization before thresholding stabilizes the scale
    - Residual shortcut bypasses the threshold for gradient flow
    - Gradient checkpointing support for memory-efficient training
    - bf16/fp16 safe: casts to float32 for the thresholding operation

    Inductive bias:
        "Sparse features are more robust and interpretable; thresholding
         activations is the proximal operator for an L1-regularized
         convolutional optimization problem."
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        config: Optional[LassoConvConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.stride = stride

        # ── Convolution ──
        # (B, in_ch, H, W) → (B, out_ch, H/stride, W/stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.kernel_size if config else 3,
            stride=stride,
            padding=(config.kernel_size // 2) if config else 1,
            bias=config.use_bias if config else False,
        )

        # ── Normalization (before threshold — stabilizes scale) ──
        self.norm = get_normalization_layer(
            config.norm_type if config else "batch_norm",
            out_channels,
        )

        # ── Learnable per-channel threshold θ ──
        # Shape: (1, C, 1, 1) — one threshold per output channel
        init_val = config.threshold_init if config else 0.01
        self.theta = nn.Parameter(
            torch.full((1, out_channels, 1, 1), init_val),
            requires_grad=config.threshold_learnable if config else True,
        )

        # ── Proximal operator (lasso!) ──
        prox_type = config.proximal_type if config else "soft"
        self.prox_op = build_prox_operator(prox_type)

        # ── Shortcut (handles channel dim and/or spatial dim changes) ──
        if stride != 1 or in_channels != out_channels:
            # 1×1 conv projects to correct channel count and spatial size
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

        # ── NaN guard (training only) ──
        self._check_nan = False

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        Returns: (B, out_channels, H/stride, W/stride)
        """
        if use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        Returns: (B, out_channels, H/stride, W/stride)
        """
        # ── Identity (shortcut) path ──
        identity = self.shortcut(x)                         # (B, out_ch, H_out, W_out)

        # ── Main path ──
        h = self.conv(x)                                    # (B, out_ch, H_out, W_out)

        if self.config is not None and self.config.norm_before_prox:
            h = self.norm(h)                                # (B, out_ch, H_out, W_out)

        # ── Lasso proximal operator (core novelty) ──
        # bf16/fp16 safety handled inside prox_op.forward()
        h = self.prox_op(h, self.theta)                     # (B, out_ch, H_out, W_out)

        # ── Residual connection ──
        out = identity + h                                  # (B, out_ch, H_out, W_out)

        # ── NaN guard (training only) ──
        if self._check_nan and self.training:
            assert not torch.isnan(out).any(), \
                f"NaN detected in {self.__class__.__name__} output"

        return out

    def extra_repr(self) -> str:
        return (
            f"stride={self.stride}, "
            f"theta_init={self.config.threshold_init if self.config else 0.01:.4f}, "
            f"theta_learnable={self.config.threshold_learnable if self.config else True}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# LISTA-Unrolled Encoder
# ═══════════════════════════════════════════════════════════════════════════

class ListaConvLayer(nn.Module):
    """
    One ISTA iteration unrolled as a feedforward layer.

    Implements one step of:
        Z_{k+1} = soft_th( Z_k - η · Dᵀ · (D · Z_k - X),  θ_k )

    where:
        D     = convolutional dictionary (decoding)
        Dᵀ    = transposed convolution (encoding, approximate)
        η     = learnable step size
        θ     = learnable threshold (sparsity level)
        X     = input features
        Z     = sparse code

    Each layer solves one step of:
        argmin_Z  ||X - D * Z||² + λ||Z||₁

    with Z initialised as the encoding of X.
    """

    def __init__(
        self,
        in_channels: int,
        dict_channels: int,
        kernel_size: int = 3,
        tied: bool = False,
        shared_dict: Optional[nn.Conv2d] = None,
    ):
        super().__init__()
        self.tied = tied

        if tied:
            # Share dictionary across all iterations
            # D must be Conv2d(dict_channels, in_channels) — maps Z → X
            self.D = shared_dict
        else:
            # Independent weights per iteration
            # D maps Z (dict_channels) → X (in_channels): decoding / dictionary
            self.D = nn.Conv2d(
                dict_channels, in_channels,
                kernel_size=kernel_size, padding=kernel_size // 2,
                bias=False,
            )

        # Transposed convolution Dᵀ maps X (in_channels) → Z (dict_channels): encoding
        # This is the approximate inverse of D, used for the gradient step.
        self.Dt = nn.Conv2d(
            in_channels, dict_channels,
            kernel_size=kernel_size, padding=kernel_size // 2,
            bias=False,
        )

        # Learnable step size and threshold per iteration
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.theta = nn.Parameter(torch.tensor(0.01))

        self.prox_op = SoftThreshold()

    def forward(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Z: (B, dict_channels, H, W) — current sparse code estimate
        X: (B, in_channels, H, W)   — original input / features to reconstruct
        Returns: (B, dict_channels, H, W) — updated sparse code
        """
        # Reconstruction residual: D @ Z - X
        # D maps dict_channels → in_channels (decoding)
        residual = self.D(Z) - X                            # (B, in_ch, H, W)

        # Gradient step: Z - eta * Dᵀ(residual)
        # Dᵀ maps in_channels → dict_channels (encoding)
        grad_step = Z - self.eta * self.Dt(residual)        # (B, dict_ch, H, W)

        # Proximal operator (lasso!)
        Z_next = self.prox_op(grad_step, self.theta)        # (B, dict_ch, H, W)

        return Z_next


class ListaEncoder(nn.Module):
    """
    Full LISTA network: unrolls K ISTA iterations for convolutional sparse coding.

    The entire encoder is interpretable as solving:
        min_Z  ||X - D * Z||² + λ||Z||₁
    via K iterations of ISTA, unrolled into a feedforward network.

    Architecture:
        X → W_encode(Z₀) → ISTA₁ → ISTA₂ → ... → ISTA_K → Z_K

    Inductive bias:
        "Deep feedforward computation can be interpreted as optimizing a
         sparse coding objective; unrolled ISTA provides theoretical
         convergence guarantees for the network's forward pass."

    When tied_weights=True, the same dictionary D is shared across all
    iterations (fewer parameters, theoretically grounded).
    When tied_weights=False, each iteration has its own D_k (more capacity).
    """

    def __init__(self, config: LassoConvConfig):
        super().__init__()
        self.config = config
        self.K = config.lista_iters

        # Input channel dimension after stem
        feat_channels = config.stage_channels[0]

        # ── Initial encoding: project features to dictionary space ──
        # (B, feat_channels, H, W) → (B, dict_channels, H, W)
        self.W_encode = nn.Conv2d(
            feat_channels,
            config.lista_dictionary_size,
            kernel_size=3, padding=1, bias=False,
        )

        # ── Unrolled LISTA iterations ──
        if config.lista_tie_weights:
            # Shared dictionary across all iterations
            shared_D = nn.Conv2d(
                config.lista_dictionary_size,
                feat_channels,
                kernel_size=3, padding=1, bias=False,
            )
            self.layers = nn.ModuleList([
                ListaConvLayer(
                    feat_channels,
                    config.lista_dictionary_size,
                    kernel_size=3,
                    tied=True,
                    shared_dict=shared_D,
                ) for _ in range(self.K)
            ])
        else:
            self.layers = nn.ModuleList([
                ListaConvLayer(
                    feat_channels,
                    config.lista_dictionary_size,
                    kernel_size=3,
                    tied=False,
                ) for _ in range(self.K)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) — input features (stem output)
        Returns: (B, dict_channels, H, W) — sparse code
        """
        # Initial sparse code estimate
        Z = self.W_encode(x)                                # (B, dict_ch, H, W)

        # Unrolled ISTA iterations
        for layer in self.layers:
            Z = layer(Z, x)                                 # (B, dict_ch, H, W)

        return Z


