"""
LassoConvNet — Proximal Operators, Base Classes, and Normalization.

Core computational primitives for the architectural integration of Lasso (L1)
regression into a CNN forward pass via soft-thresholding proximal operators.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Base Class for the Novel Operator
# ═══════════════════════════════════════════════════════════════════════════

class BaseProxOperator(ABC, nn.Module):
    """
    Abstract base class for Lasso proximal operators.

    The proximal operator for the L1 norm is:
        prox_{lambda * ||·||_1}(x) = sign(x) * max(|x| - lambda, 0)

    Subclasses implement variants (standard soft-thresholding, adaptive,
    hard-thresholding) while maintaining the same interface.

    Inductive bias:
        "Sparse features are more robust and interpretable; thresholding
         activations is the proximal operator for an L1-regularized
         convolutional optimization problem."
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Apply the proximal (thresholding) operator.

        Args:
            x:     input tensor, typically (B, C, H, W)
            theta: threshold parameter, broadcastable to x shape
                   typically (1, C, 1, 1) for per-channel thresholds
        Returns:
            thresholded tensor, same shape as x
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Concrete Proximal Operators
# ═══════════════════════════════════════════════════════════════════════════

class SoftThreshold(BaseProxOperator):
    """
    Standard L1 proximal operator (soft-thresholding).

    prox_{theta * ||·||_1}(x) = sign(x) * max(|x| - theta, 0)

    When theta=0, this reduces to the identity function.
    When theta is learnable per channel, each feature map learns its
    own sparsity level.

    bf16/fp16 safety:
        Casts to float32 internally to avoid gradient underflow in the
        |x| - theta region where small values may underflow in half-precision.
    """

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, C, H, W) or any shape
        theta: (1, C, 1, 1) — broadcastable to x
        Returns: same shape as x
        """
        # bf16/fp16 safety: cast to float32 for the abs and subtraction
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            x = x.float()
            theta = theta.float()

        out = torch.sign(x) * F.relu(torch.abs(x) - theta)  # same shape as x

        return out.to(dtype)


class AdaptiveSoftThreshold(BaseProxOperator):
    """
    Smooth (differentiable everywhere) approximation of soft-thresholding.

    Uses a sigmoid gating function:
        g(x) = sigmoid(alpha * (|x| - theta))
        z = g(x) * x

    Advantages over standard soft-threshold:
    - Non-zero gradient for |x| < theta (no gradient starvation / neuron death)
    - Smooth transition, better for SGD with small learning rates
    - Avoids hard kinks at |x| = theta

    When alpha → inf, this converges to standard soft-thresholding.
    Default alpha=0.1 gives a soft transition; increase for harder thresholding.

    Reference:
        "Learning Fast Approximations of Sparse Coding" (Gregor & LeCun, 2010)
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, C, H, W) or any shape
        theta: (1, C, 1, 1) — broadcastable to x
        Returns: same shape as x
        """
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            x = x.float()
            theta = theta.float()

        magnitude = torch.abs(x)                            # (B, C, H, W)
        gate = torch.sigmoid(self.alpha * (magnitude - theta))  # (B, C, H, W)
        out = gate * x                                      # (B, C, H, W)

        return out.to(dtype)


class HardThreshold(BaseProxOperator):
    """
    Hard-thresholding operator (L0 approximation).

    H(x) = x if |x| > theta, else 0

    This is NOT the proximal operator of any convex regularizer,
    but it can be useful for aggressive sparsity.

    Warning: gradient is zero almost everywhere (only non-zero at |x|=theta
    in the weak sense). Use only with the adaptive variant or gradient
    surrogates.
    """

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            x = x.float()
            theta = theta.float()

        # bf16 safety: use float for comparison
        mask = (torch.abs(x) > theta).to(x.dtype)          # (B, C, H, W)
        out = mask * x                                      # (B, C, H, W)

        return out.to(dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def build_prox_operator(proximal_type: str, **kwargs) -> BaseProxOperator:
    """Factory function for proximal operators."""
    if proximal_type == "soft":
        return SoftThreshold()
    elif proximal_type == "adaptive":
        return AdaptiveSoftThreshold(alpha=kwargs.get("alpha", 0.1))
    elif proximal_type == "hard":
        return HardThreshold()
    else:
        raise ValueError(f"Unknown proximal_type: {proximal_type}")


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════

def get_normalization_layer(norm_type: str, num_features: int) -> nn.Module:
    """
    Get a 2D normalization layer.

    Args:
        norm_type: "batch_norm", "layer_norm" (via GroupNorm(1, C)), or "none"
        num_features: number of channels C
    Returns:
        nn.Module that normalizes a (B, C, H, W) tensor
    """
    if norm_type == "batch_norm":
        # (B, C, H, W) → learnable per-channel affine
        return nn.BatchNorm2d(num_features, track_running_stats=True)
    elif norm_type == "layer_norm":
        # GroupNorm(1, C) = LayerNorm over (C, H, W) for 2D
        return nn.GroupNorm(1, num_features)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. "
                         f"Expected 'batch_norm', 'layer_norm', or 'none'.")


# ═══════════════════════════════════════════════════════════════════════════
# Sparsity Utilities
# ═══════════════════════════════════════════════════════════════════════════

def compute_sparsity_ratio(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the fraction of elements with near-zero magnitude.

    Args:
        x:   activation tensor, any shape
        eps: threshold below which an element is considered "zero"
    Returns:
        scalar tensor — fraction of elements with |x| < eps
    """
    return (torch.abs(x) < eps).float().mean()


def compute_channel_sparsity(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-channel sparsity ratio.

    Args:
        x:   activation tensor (B, C, H, W)
        eps: threshold for "zero"
    Returns:
        (C,) tensor — sparsity ratio per channel
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)                            # (B, C, H*W)
    return (torch.abs(x_flat) < eps).float().mean(dim=(0, 2))  # (C,)
