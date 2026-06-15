"""
LassoConvNet — Core Building Blocks.

Contains the novel LassoProxConvBlock (soft-thresholding = lasso proximal operator
integrated into the forward pass), adaptive variants, and LISTA encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LassoConvConfig


# ═══════════════════════════════════════════════════════════════════════
# Proximal Operators
# ═══════════════════════════════════════════════════════════════════════

def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    L1 proximal operator (soft-thresholding).

    prox_{θ·||·||_1}(x) = sign(x) * max(|x| - θ, 0)

    This IS the architectural integration of lasso regression.
    When θ=0, this reduces to the identity function.

    Args:
        x: input tensor, any shape
        theta: threshold tensor, broadcastable to x shape
               (typically shape (1, C, 1, 1) for per-channel thresholds)
    Returns:
        thresholded tensor, same shape as x
    """
    return torch.sign(x) * F.relu(torch.abs(x) - theta)


def adaptive_soft_threshold(
    x: torch.Tensor,
    theta: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Smooth (differentiable everywhere) approximation of soft-thresholding.

    Uses a sigmoid gate: g(x) = sigmoid(alpha * (|x| - theta))
    Returns: g(x) * x

    Advantages over standard soft-threshold:
    - Non-zero gradient for |x| < theta (no gradient starvation)
    - Smooth transition, better for SGD with small learning rates

    When alpha → inf, this converges to standard soft-thresholding.
    """
    magnitude = torch.abs(x)
    gate = torch.sigmoid(alpha * (magnitude - theta))
    return gate * x


# ═══════════════════════════════════════════════════════════════════════
# Feature Normalization
# ═══════════════════════════════════════════════════════════════════════

def get_normalization_layer(norm_type: str, num_features: int):
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "layer_norm":
        return nn.GroupNorm(1, num_features)  # GN(1, C) = LN over (C,H,W)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# ═══════════════════════════════════════════════════════════════════════
# Core Novel Block: LassoProxConvBlock
# ═══════════════════════════════════════════════════════════════════════

class LassoProxConvBlock(nn.Module):
    """
    Convolutional block with integrated Lasso proximal operator.

    Architecture (pre-norm style):
        x → Conv2d → BatchNorm → SoftThreshold(θ_ch) → + residual → FFN

    The soft-thresholding operator is the proximal map for L1:
        prox(x) = sign(x) * max(|x| - θ, 0)

    Key design choices:
    - Pre-norm: normalization before thresholding stabilizes the scale
    - Per-channel thresholds: each output channel learns its own θ
    - Thresholds are learnable parameters, updated via backprop
    - Residual connection bypasses the threshold for gradient flow

    Inductive bias:
        "Sparse features are more robust and interpretable; thresholding
         activations is the proximal operator for an L1-regularized
         convolutional optimization problem."
    """

    def __init__(self, in_channels: int, out_channels: int, config: LassoConvConfig):
        super().__init__()
        self.config = config

        # ── Convolution ──
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
            bias=config.use_bias,
        )

        # ── Normalization (before threshold) ──
        self.norm = get_normalization_layer(config.norm_type, out_channels)

        # ── Learnable per-channel threshold (θ) ──
        # Shape: (1, C, 1, 1) — one threshold per output channel
        init_val = config.threshold_init
        self.theta = nn.Parameter(
            torch.full((1, out_channels, 1, 1), init_val),
            requires_grad=config.threshold_learnable,
        )

        # ── Shortcut (for channel dim mismatch) ──
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        # ── Optional 1×1 projection after threshold ──
        self.proj = (
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
            if config.dropout > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        # Main path
        h = self.conv(x)

        if self.config.norm_before_prox:
            h = self.norm(h)

        # ── Lasso proximal operator (core novelty) ──
        if self.config.proximal_type == "soft":
            h = soft_threshold(h, self.theta)
        elif self.config.proximal_type == "adaptive":
            h = adaptive_soft_threshold(h, self.theta)
        else:
            raise ValueError(f"Unknown proximal_type: {self.config.proximal_type}")

        # Residual connection (critical for gradient flow through dead zone)
        return identity + h


# ═══════════════════════════════════════════════════════════════════════
# LISTA-Unrolled Encoder
# ═══════════════════════════════════════════════════════════════════════

class ListaConvLayer(nn.Module):
    """
    One ISTA iteration unrolled as a feedforward layer.

    Implements:
        Z_{k+1} = soft_th(Z_k - η · D^T · (D · Z_k - X), θ_k)

    Each layer solves one step of:
        argmin_Z  ||X - DZ||² + λ||Z||₁
    """

    def __init__(self, in_channels: int, dict_channels: int, kernel_size: int,
                 tied: bool = False, shared_dict: nn.Module = None):
        super().__init__()
        self.tied = tied

        if tied:
            # Share dictionary across all iterations
            self.D = shared_dict
            self.Dt = nn.Conv2d(dict_channels, in_channels, kernel_size,
                                padding=kernel_size//2, bias=False)
            # Learnable step size and threshold per iteration
            self.eta = nn.Parameter(torch.tensor(0.1))
            self.theta = nn.Parameter(torch.tensor(0.01))
        else:
            # Independent weights per iteration
            self.D = nn.Conv2d(in_channels, dict_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
            self.Dt = nn.Conv2d(dict_channels, in_channels, kernel_size,
                                padding=kernel_size//2, bias=False)
            self.eta = nn.Parameter(torch.tensor(0.1))
            self.theta = nn.Parameter(torch.tensor(0.01))

    def forward(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Reconstruction residual: D @ Z - X
        residual = self.D(Z) - X
        # Gradient step: Z - eta * D^T(residual)
        grad_step = Z - self.eta * self.Dt(residual)
        # Proximal operator (lasso!)
        Z_next = soft_threshold(grad_step, self.theta)
        return Z_next


class ListaEncoder(nn.Module):
    """
    Full LISTA network: unrolls K ISTA iterations for convolutional sparse coding.

    The entire encoder is interpretable as solving:
        min_Z  ||X - DZ||² + λ||Z||₁
    via K iterations of ISTA, unrolled into a feedforward network.

    Inductive bias:
        "Deep feedforward computation can be interpreted as optimizing a
         sparse coding objective; unrolled ISTA provides theoretical
         convergence guarantees."
    """

    def __init__(self, config: LassoConvConfig):
        super().__init__()
        self.config = config
        self.K = config.lista_iters

        # Initial encoding: learnable linear mapping
        self.W_encode = nn.Conv2d(
            config.in_channels if config.in_channels == 3 else config.stage_channels[0],
            config.lista_dictionary_size,
            kernel_size=3, padding=1, bias=False,
        )

        # Shared dictionary (tied weights across iterations)
        if config.lista_tie_weights:
            shared_D = nn.Conv2d(
                config.lista_dictionary_size,
                config.in_channels if config.in_channels == 3 else config.stage_channels[0],
                kernel_size=3, padding=1, bias=False,
            )
            self.layers = nn.ModuleList([
                ListaConvLayer(
                    config.in_channels if config.in_channels == 3 else config.stage_channels[0],
                    config.lista_dictionary_size,
                    kernel_size=3, tied=True, shared_dict=shared_D,
                ) for _ in range(self.K)
            ])
        else:
            self.layers = nn.ModuleList([
                ListaConvLayer(
                    config.in_channels if config.in_channels == 3 else config.stage_channels[0],
                    config.lista_dictionary_size,
                    kernel_size=3, tied=False,
                ) for _ in range(self.K)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.W_encode(x)
        for layer in self.layers:
            Z = layer(Z, x)
        return Z


# ═══════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════

def group_lasso_penalty(model: nn.Module, group_size: int, strength: float) -> torch.Tensor:
    """
    Group lasso penalty on convolutional filters.

    Groups consecutive `group_size` output filters and applies L2 penalty
    on each group's Frobenius norm. This encourages entire groups of filters
    to be jointly pruned (structured sparsity).

    penalty = strength * Σ_g ||W_g||_F

    where W_g is the weight matrix for filter group g.
    """
    penalty = 0.0
    for name, param in model.named_parameters():
        if 'conv' in name and param.dim() >= 4:
            # param shape: (out_channels, in_channels, kH, kW)
            out_c = param.shape[0]
            n_groups = out_c // group_size
            if n_groups > 0:
                # Reshape to (n_groups, group_size, -1)
                w_reshaped = param[:n_groups * group_size].view(
                    n_groups, group_size, -1
                )
                # Frobenius norm per group, then sum
                penalty += w_reshaped.norm(p=2, dim=(1, 2)).sum()
    return strength * penalty


def l1_weight_penalty(model: nn.Module, strength: float) -> torch.Tensor:
    """
    Element-wise L1 penalty on all weight parameters.
    """
    penalty = 0.0
    for param in model.parameters():
        if param.dim() >= 2:  # weight matrices, not biases/norms
            penalty += param.abs().sum()
    return strength * penalty


def lasso_total_loss(
    ce_loss: torch.Tensor,
    model: nn.Module,
    config: LassoConvConfig,
) -> torch.Tensor:
    """
    Composite loss: CE + L1_weight + group_lasso.

    In 'proximal' mode, activation sparsity is handled by the forward pass
    (soft-thresholding), so activation_sparsity_weight is not added here.
    In 'loss_only' mode, activation L1 is added to the loss.
    """
    loss = ce_loss

    if config.l1_weight_decay > 0:
        loss = loss + l1_weight_penalty(model, config.l1_weight_decay)

    if config.use_group_lasso and config.group_lasso_strength > 0:
        loss = loss + group_lasso_penalty(
            model, config.group_size, config.group_lasso_strength
        )

    return loss
