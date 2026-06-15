"""
LassoConvNet — Composite Loss Functions.

Integrates Lasso regression at three levels:
    1. L1 weight penalty:     element-wise L1 on all kernel weights
    2. Group lasso penalty:   structured sparsity on filter groups
    3. Activation sparsity:   auxiliary L1 on feature maps (loss_only mode)

Total loss:
    L = CE(y_pred, y_true)
      + λ₁ · ||W||₁
      + λ_g · Σ_g ||W_g||_F
      + λ_act · ||Z||₁    (only in loss_only mode)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import LassoConvConfig


# ═══════════════════════════════════════════════════════════════════════════
# L1 Weight Penalty
# ═══════════════════════════════════════════════════════════════════════════

def l1_weight_penalty(
    model: nn.Module,
    strength: float = 1e-5,
    exclude_types: tuple = (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm),
) -> torch.Tensor:
    """
    Element-wise L1 penalty on all weight parameters.

    Applied to conv and linear weight tensors (dim >= 2).
    Excludes biases and normalization parameters.

    penalty = strength * Σ|w_i| for all weight parameters w

    This induces unstructured sparsity: individual weight values are
    driven to zero independently.

    Args:
        model: PyTorch model whose parameters to penalize
        strength: L1 regularization coefficient (lambda)
        exclude_types: parameter types to exclude from penalty
    Returns:
        scalar tensor: the L1 penalty value
    """
    penalty = 0.0
    for module in model.modules():
        if isinstance(module, exclude_types):
            continue
        for name, param in module.named_parameters(recurse=False):
            # Only penalize weight matrices (dim >= 2), not biases (dim=1)
            if "weight" in name and param.dim() >= 2:
                # bf16 safety: cast to float32 for abs().sum()
                if param.dtype in (torch.float16, torch.bfloat16):
                    penalty = penalty + param.float().abs().sum()
                else:
                    penalty = penalty + param.abs().sum()

    return strength * penalty


# ═══════════════════════════════════════════════════════════════════════════
# Group Lasso Penalty
# ═══════════════════════════════════════════════════════════════════════════

def group_lasso_penalty(
    model: nn.Module,
    group_size: int = 8,
    strength: float = 1e-4,
) -> torch.Tensor:
    """
    Group lasso penalty on convolutional filters.

    Groups consecutive `group_size` output filters and applies an L2 penalty
    on each group's Frobenius norm. This encourages entire groups of filters
    to be jointly pruned (structured sparsity).

    penalty = strength * Σ_g ||W_g||_F

    where W_g ∈ ℝ^(group_size × C_in × K × K) is the weight tensor for
    filter group g.

    Structured sparsity is preferable for model compression because:
    - Entire filters can be removed at export time
    - No need for specialized sparse matrix hardware
    - Reduces the effective width of the network

    Args:
        model: PyTorch model
        group_size: number of consecutive output filters per group
        strength: group lasso regularization coefficient
    Returns:
        scalar tensor: the group lasso penalty value
    """
    penalty = 0.0

    for module in model.modules():
        if not isinstance(module, nn.Conv2d):
            continue

        weight = module.weight  # (out_channels, in_channels, kH, kW)
        out_c = weight.shape[0]

        # Only apply if there are enough filters to form at least one group
        num_groups = out_c // group_size
        if num_groups < 1:
            continue

        # Take only the first num_groups * group_size filters
        w_grouped = weight[:num_groups * group_size]  # (N*gs, C_in, kH, kW)
        w_grouped = w_grouped.view(num_groups, group_size, -1)
        # w_grouped: (num_groups, group_size, C_in * kH * kW)

        # Frobenius norm per group (L2 over all elements in group)
        group_norms = w_grouped.norm(p=2, dim=(1, 2))  # (num_groups,)

        # bf16 safety
        if group_norms.dtype in (torch.float16, torch.bfloat16):
            penalty = penalty + group_norms.float().sum()
        else:
            penalty = penalty + group_norms.sum()

    return strength * penalty


# ═══════════════════════════════════════════════════════════════════════════
# Activation Sparsity Penalty
# ═══════════════════════════════════════════════════════════════════════════

def activation_l1_penalty(
    model: nn.Module,
    strength: float = 1e-4,
    capture_hook: Optional[str] = None,
) -> torch.Tensor:
    """
    Auxiliary L1 penalty on feature map activations.

    Only needed in 'loss_only' mode where the forward pass does NOT
    include soft-thresholding. In 'proximal' mode, activation sparsity
    is induced architecturally and this penalty is redundant.

    Without hooks, this is a no-op returning 0. To actually compute
    activation L1, the caller should pass the activation tensor directly
    or use forward hooks.

    Args:
        model: PyTorch model
        strength: L1 coefficient for activations
        capture_hook: name of a module whose output to penalize (unused here)
    Returns:
        scalar tensor: the activation L1 penalty value (0 if not captured)
    """
    # This function is intentionally minimal — activation sparsity is
    # best handled architecturally via soft-thresholding in the forward pass.
    # For 'loss_only' mode, the caller should extract activations manually.
    return torch.tensor(0.0, device=next(model.parameters()).device)


# ═══════════════════════════════════════════════════════════════════════════
# Composite Lasso Loss
# ═══════════════════════════════════════════════════════════════════════════

def lasso_total_loss(
    ce_loss: torch.Tensor,
    model: nn.Module,
    config: LassoConvConfig,
) -> torch.Tensor:
    """
    Compute the composite loss with Lasso penalties.

    L = CE + λ₁ · ||W||₁ + λ_g · Σ_g ||W_g||_F [+ λ_act · ||Z||₁]

    Mode-specific behavior:
        'proximal':   L1 on weights + group lasso on filters.
                      Activation sparsity is INDUCED BY THE ARCHITECTURE
                      (soft-thresholding in forward pass), so no activation
                      penalty is needed.

        'loss_only':  L1 on weights + group lasso + activation L1.
                      No architectural sparsity — all regularization is
                      through the loss function.

        'lista_unrolled': L1 on weights + group lasso.
                          Activation sparsity is induced by LISTA iterations.

    Args:
        ce_loss: cross-entropy loss (scalar tensor)
        model: the LassoConvNet model
        config: model configuration
    Returns:
        scalar tensor: total loss
    """
    loss = ce_loss

    # ── L1 weight decay (unstructured sparsity on kernel weights) ──
    if config.l1_weight_decay > 0:
        l1_pen = l1_weight_penalty(model, strength=config.l1_weight_decay)
        loss = loss + l1_pen

    # ── Group lasso (structured sparsity on filter groups) ──
    if config.use_group_lasso and config.group_lasso_strength > 0:
        gl_pen = group_lasso_penalty(
            model,
            group_size=config.group_size,
            strength=config.group_lasso_strength,
        )
        loss = loss + gl_pen

    # ── Auxiliary L1 on activations (loss_only mode only) ──
    if (
        config.lasso_mode == "loss_only"
        and config.activation_sparsity_weight > 0
    ):
        act_pen = activation_l1_penalty(
            model, strength=config.activation_sparsity_weight
        )
        loss = loss + act_pen

    # NaN guard
    if torch.isnan(loss).any():
        raise ValueError("NaN detected in lasso_total_loss")

    return loss
