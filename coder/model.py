"""
LassoConvNet — Full Model.

Combines the LassoConvNet backbone with a classification head to produce
a complete image classifier with integrated Lasso regression.

The model supports three modes of Lasso integration:
    - "loss_only":    standard CNN + L1 penalty in the loss function
    - "proximal":     soft-thresholding in the forward pass (architectural)
    - "lista_unrolled": unrolled ISTA iterations (full optimization integration)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

from config import LassoConvConfig
from backbone import LassoConvNetBackbone
from layers import compute_sparsity_ratio, compute_channel_sparsity


# ═══════════════════════════════════════════════════════════════════════════
# Classification Head
# ═══════════════════════════════════════════════════════════════════════════

class ClassifierHead(nn.Module):
    """
    Classification head with optional global pooling.

    Architecture:
        GlobalPool → Flatten → Linear → ReLU → Dropout → Linear → Logits

    Input:  (B, C, H, W) feature maps
    Output: (B, n_classes) logits
    """

    def __init__(self, config: LassoConvConfig):
        super().__init__()
        self.config = config

        # Determine input feature dimension
        if config.lasso_mode == "lista_unrolled":
            in_features = config.lista_dictionary_size
        else:
            in_features = config.stage_channels[-1]

        # ── Global pooling ──
        if config.global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        elif config.global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)  # (B, C, 1, 1)
        else:
            self.pool = nn.Identity()

        # ── MLP classifier ──
        self.head = nn.Sequential(
            nn.Flatten(),                              # (B, C)
            nn.Linear(in_features, config.head_hidden_dim),  # (B, head_dim)
            nn.ReLU(),                                 # (B, head_dim)
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.head_hidden_dim, config.n_classes),  # (B, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) — feature maps from backbone
        Returns: (B, n_classes) — class logits
        """
        x = self.pool(x)                               # (B, C, 1, 1)
        return self.head(x)                            # (B, n_classes)


# ═══════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════

class LassoConvNet(nn.Module):
    """
    LassoConvNet — CNN with integrated Lasso regression.

    A complete image classifier where the standard ReLU activation is
    replaced by soft-thresholding (the Lasso proximal operator), making
    activations sparse by design, not just by regularization.

    Three operating modes:
    1. loss_only:     Standard CNN + L1 loss penalty (baseline)
    2. proximal:      Soft-thresholding in every conv block (architectural)
    3. lista_unrolled: Unrolled ISTA iterations for sparse coding

    Key architectural features:
    - 4-stage ResNet-style pyramid with progressive downsampling
    - Per-channel learnable thresholds (shape (1, C, 1, 1))
    - Group-lasso structured sparsity on filters (via loss)
    - bf16/fp16 safe forward pass
    - Gradient checkpointing support
    - Sparsity ratio tracking for evaluation

    Usage:
        >>> cfg = LassoConvConfig()
        >>> model = LassoConvNet(cfg)
        >>> logits = model(x)           # (B, n_classes)
        >>> sparsity = model.get_sparsity_ratio()  # scalar
    """

    def __init__(self, config: LassoConvConfig):
        super().__init__()
        self.config = config

        # ── Backbone (feature extractor) ──
        self.backbone = LassoConvNetBackbone(config)

        # ── Classifier head ──
        self.head = ClassifierHead(config)

        # ── Sparsity tracking (hooks) ──
        self._activation_store: Optional[torch.Tensor] = None
        self._register_sparsity_hooks()

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, in_channels, H, W) — input image
        Returns: (B, n_classes) — class logits

        Shape guarantee:
            Input:  (B, 3, H, W)     — for CIFAR: H=W=32
            Output: (B, n_classes)   — for CIFAR-10: 10
        """
        B, C, H, W = x.shape                            # batch, channels, height, width

        features = self.backbone(x, use_checkpoint=use_checkpoint)
        # Proximal mode:  (B, C4, H/8, W/8)
        # LISTA mode:     (B, dict_size, H, W)

        logits = self.head(features)                     # (B, n_classes)

        # NaN guard (training only)
        if self.training:
            assert not torch.isnan(logits).any(), \
                "NaN detected in LassoConvNet output logits"

        return logits

    # ── Sparsity measurement ──

    def _register_sparsity_hooks(self):
        """Register forward hooks to capture post-threshold activations."""

        def _capture_hook(module, input, output):
            if isinstance(module, nn.ReLU):
                self._activation_store = output.detach()

        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(_capture_hook)

    def get_sparsity_ratio(self, eps: float = 1e-6) -> Optional[torch.Tensor]:
        """
        Compute the fraction of near-zero activations in the model.

        Returns the fraction of ReLU activations with |x| < eps.
        Returns None if no forward pass has been run yet.
        """
        if self._activation_store is None:
            return None
        return compute_sparsity_ratio(self._activation_store, eps=eps)

    def get_channel_sparsity(
        self,
        eps: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        """
        Compute per-channel sparsity from the last forward pass.

        Returns (C,) tensor of sparsity per channel, or None.
        """
        if self._activation_store is None:
            return None
        return compute_channel_sparsity(self._activation_store, eps=eps)

    # ── Weight initialization ──

    def _init_weights(self):
        """Initialize conv weights with Kaiming normal and zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Threshold management ──

    def clamp_thresholds(self, min_val: float = 1e-6):
        """
        Clamp all learnable thresholds to [min_val, ∞).

        Prevents thresholds from becoming negative or vanishing to zero,
        which would eliminate the sparsity-inducing effect.
        """
        for name, param in self.named_parameters():
            if "theta" in name:
                param.data.clamp_(min=min_val)

    def get_thresholds(self) -> dict:
        """Return a dict mapping parameter names to threshold values."""
        thresholds = {}
        for name, param in self.named_parameters():
            if "theta" in name:
                thresholds[name] = param.detach().cpu().squeeze()
        return thresholds

    # ── Utility ──

    def set_check_nan(self, enabled: bool = True):
        """Enable or disable NaN checking in all blocks."""
        for module in self.modules():
            if hasattr(module, "_check_nan"):
                module._check_nan = enabled

    def count_zero_filters(self, eps: float = 1e-6) -> int:
        """
        Count the number of convolution filters whose weights
        are all near-zero (can be pruned).
        """
        total_zero = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.dim() == 4:
                # weight shape: (out_channels, in_channels, kH, kW)
                w = m.weight.detach()
                # Filter is "zero" if all its elements are < eps
                filter_max = w.view(w.shape[0], -1).abs().max(dim=1).values  # (out_ch,)
                total_zero += (filter_max < eps).sum().item()
        return total_zero


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Count Helper
# ═══════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module, verbose: bool = True) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model
        verbose: if True, prints the count
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    if verbose:
        print(f"Total params:     {total:>12,}")
        print(f"Trainable params: {trainable:>12,}")
        print(f"Non-trainable:    {total - trainable:>12,}")

    return total, trainable
