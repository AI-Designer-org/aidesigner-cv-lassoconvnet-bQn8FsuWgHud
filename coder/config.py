"""
LassoConvNet — Configuration dataclass.

Defines all hyperparameters for a CNN architecture that integrates Lasso (L1)
proximal operators into the forward pass for sparse feature learning.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LassoConvConfig:
    # ── Input / Data ──
    in_channels: int = 3
    img_size: int = 32           # CIFAR default; 224 for ImageNet
    n_classes: int = 10          # CIFAR-10

    # ── Backbone structure ──
    base_channels: int = 64
    stage_depths: Tuple[int, ...] = (2, 2, 2, 2)   # layers per spatial stage
    stage_channels: Tuple[int, ...] = (64, 128, 256, 512)
    kernel_size: int = 3
    downsample: str = "stride"   # "stride" | "pool" — how to reduce spatial dim

    # ── Lasso integration ──
    lasso_mode: str = "proximal" # "loss_only" | "proximal" | "lista_unrolled"
    threshold_init: float = 0.01 # initial soft-threshold value (theta)
    threshold_learnable: bool = True   # learn per-channel thresholds via backprop
    use_group_lasso: bool = True       # apply group-lasso on output filters
    group_lasso_strength: float = 1e-4
    group_size: int = 8               # filter group size for structured sparsity

    # ── Proximal operator ──
    proximal_type: str = "soft"  # "soft" (L1) | "adaptive" (sigmoid-gated)
    norm_before_prox: bool = True
    norm_type: str = "batch_norm"  # "batch_norm" | "layer_norm" | "none"

    # ── Loss terms ──
    l1_weight_decay: float = 1e-5     # element-wise L1 on conv kernel weights
    l2_weight_decay: float = 0.0      # standard L2 (set to 0 when using L1)
    activation_sparsity_weight: float = 1e-4  # auxiliary L1 on activations (loss_only mode)
    l1_decay_schedule: str = "constant"  # "constant" | "warmup" | "cosine_decay"

    # ── Regularization (standard) ──
    dropout: float = 0.0
    use_bias: bool = False
    dtype: str = "float32"

    # ── LISTA unrolled options (lista_unrolled mode) ──
    lista_iters: int = 6          # number of unrolled ISTA iterations
    lista_dictionary_size: int = 256  # overcomplete dictionary atoms
    lista_tie_weights: bool = True    # tied vs untied weights across iterations

    # ── Classification head ──
    head_hidden_dim: int = 256
    global_pool: str = "avg"       # "avg" | "max" | "none"

    # ── Training ──
    lr: float = 1e-3
    lr_threshold: float = 1e-4    # slower LR for threshold params (avoid oscillation)
    l1_warmup_epochs: int = 5
    theta_min: float = 1e-6       # lower bound clamp on theta
