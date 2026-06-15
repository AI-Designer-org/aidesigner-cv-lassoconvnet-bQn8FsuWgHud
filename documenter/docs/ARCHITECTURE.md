# Architecture

## 1. Motivation

Standard CNNs for image classification use ReLU activations and rely on external
L1 regularization in the loss function to encourage sparsity. This separation
between architecture (ReLU, which does not induce sparsity) and regularization
(L1 loss, which must compete with the cross-entropy gradient) is indirect and
often produces limited sparsity in practice. An alternative approach —
post-hoc pruning — requires a separate fine-tuning step and has no influence
on the learned representations during training.

LassoConvNet addresses this gap by integrating the Lasso (L1) proximal operator
directly into the CNN forward pass as a differentiable layer. Soft-thresholding

\[
\text{prox}_{\theta \|\cdot\|_1}(x) = \operatorname{sign}(x) \cdot \max(|x| - \theta, 0)
\]

replaces ReLU as the activation function, making activation sparsity an
**architectural property** rather than just a regularization target. Per-channel
learnable thresholds let each feature map discover its own sparsity level.

The approach builds on the LISTA framework (Gregor & LeCun, 2010), which showed
that iterative soft-thresholding algorithms can be unrolled into feedforward
networks. LassoConvNet extends this to a modern ResNet-style pyramid with three
levels of lasso integration:

1. **Loss-level**: standard L1 weight decay + group lasso on filters
2. **Architectural (proximal)**: soft-thresholding in every conv block's forward pass
3. **Optimization-level (LISTA unrolled)**: full unrolled ISTA iterations

**Hypothesis.** Replacing ReLU with soft-thresholding in a CNN's forward pass
produces feature maps that are measurably sparser (>20% near-zero activations at
convergence) while maintaining classification accuracy within 3% of an
equivalent-width baseline CNN.

## 2. At a glance

```
Input Image (B, 3, H, W)
         │
         ▼
  ┌─────────────────────┐
  │  Stem Conv 3×3, s2  │  → (B, C1, H/2, W/2)
  └─────────┬───────────┘
            │
     ╔══════╧═══════════════════════════════════╗
     ║  Stage 1: C1 channels,  H/2 × W/2      ║
     ║  ┌─────────────────────────────────────┐ ║
     ║  │  LassoProxConvBlock × depth[0]      │ ║
     ║  │  ┌──────────────────────────────┐   │ ║
     ║  │  │ Conv2d 3×3                   │   │ ║
     ║  │  │   → BatchNorm                │   │ ║
     ║  │  │   → SoftThreshold(θ_ch) ←────┼───╫─── Lasso integrated here
     ║  │  │   → + residual shortcut      │   │ ║
     ║  │  └──────────────────────────────┘   │ ║
     ║  └─────────────────────────────────────┘ ║
     ╚══════╤═══════════════════════════════════╝
            │  Downsample (stride 2 conv)
            ▼
     ╔══════╧═══════════════════════════════════╗
     ║  Stage 2: C2 channels,  H/4 × W/4      ║
     ║  ┌─────────────────────────────────────┐ ║
     ║  │  LassoProxConvBlock × depth[1]      │ ║
     ║  └─────────────────────────────────────┘ ║
     ╚══════╤═══════════════════════════════════╝
            │
            ▼
     ╔══════╧═══════════════════════════════════╗
     ║  Stage 3: C3 channels,  H/8 × W/8      ║
     ║  ┌─────────────────────────────────────┐ ║
     ║  │  LassoProxConvBlock × depth[2]      │ ║
     ║  └─────────────────────────────────────┘ ║
     ╚══════╤═══════════════════════════════════╝
            │
            ▼
     ╔══════╧═══════════════════════════════════╗
     ║  Stage 4: C4 channels,  H/16 × W/16    ║
     ║  ┌─────────────────────────────────────┐ ║
     ║  │  LassoProxConvBlock × depth[3]      │ ║
     ║  └─────────────────────────────────────┘ ║
     ╚══════╤═══════════════════════════════════╝
            │
            ▼
  ┌─────────────────────┐
  │  Global Avg Pool    │  → (B, C4)
  └─────────┬───────────┘
            │
  ┌─────────────────────┐
  │  FC → ReLU → FC     │  Classification head
  └─────────┬───────────┘
            │
            ▼
      (B, n_classes)      ← Logits
```

| Property | Value |
|---|---|
| Parameter count (default CIFAR-10 config) | 5,032,138 |
| Time complexity | O(B · C · H · W · K²) per layer — standard conv scaling |
| Space complexity | O(B · C · H · W) activations + O(K² · C_in · C_out) weights |
| Hardware requirements | Single GPU (tested on CPU; A100/RTX 3090 for training) |
| Custom kernels | None — pure PyTorch, no custom CUDA |

### LISTA-Unrolled Mode

```
Input X (B, C, H, W)
         │
         ▼
  ┌─────────────────────┐
  │  W_encode (conv)    │  → initial sparse code Z_0
  └─────────┬───────────┘
            │
   ╔════════╧══════════════════════════════════════╗
   ║  LISTA Iterations × K                        ║
   ║                                               ║
   ║  Z_1 = soft_th( Z_0 - η·Dᵀ(D·Z_0 - X), θ₁ )  ║
   ║  Z_2 = soft_th( Z_1 - η·Dᵀ(D·Z_1 - X), θ₂ )  ║
   ║  ...                                          ║
   ║  Z_K = soft_th(...)                           ║
   ║                                               ║
   ║  Each iteration = one conv layer               ║
   ║  D = conv dictionary (shared or per-iteration) ║
   ╚════════════════════════════════════════════════╝
            │
            ▼
  ┌─────────────────────┐
  │  Classifier Head    │
  └─────────┬───────────┘
            │
            ▼
      (B, n_classes)
```

## 3. The core component

### 3.1 Intuition

**Why soft-thresholding instead of ReLU?** ReLU sets negative values to zero but
passes all positive values unchanged. Soft-thresholding creates a "dead zone"
around zero: any activation with magnitude below the threshold `θ` becomes exactly
zero. This is the proximal operator for L1 regularization — the exact operation
that produces sparse solutions in Lasso regression. By placing this in the forward
pass, the network architecturally *must* produce sparse feature maps; it cannot
"cheat" by using a ReLU and hoping the L1 loss will do the work.

**How thresholds work.** Each output channel has its own threshold `θ_c` (shape
`(1, C, 1, 1)`). When `θ_c` is learnable, the network can decide which feature
channels should be highly sparse (large `θ_c`, many activations zeroed) and which
should be denser (small `θ_c`, near-identity behavior). The residual shortcut
bypasses the threshold, ensuring gradient flow even when most activations fall
into the dead zone.

### 3.2 Equations

**Soft-thresholding (L1 proximal operator):**

\[
z_{c} = \operatorname{prox}_{\theta_c \|\cdot\|_1}(h_c)
      = \operatorname{sign}(h_c) \cdot \max(|h_c| - \theta_c, 0)
\]

where `h_c` is the pre-activation for channel `c` (after conv + norm) and `θ_c`
is the per-channel threshold. When `θ_c = 0`, this reduces to identity. When
`θ_c → ∞`, all activations are zeroed.

**Adaptive soft-thresholding (sigmoid-gated):**

\[
g_c(h) = \sigma(\alpha \cdot (|h_c| - \theta_c)), \quad
z_c = g_c(h) \cdot h_c
\]

where `σ` is the sigmoid function. For `α → ∞`, this converges to standard
soft-thresholding. For finite `α`, the gradient is non-zero for `|h_c| < θ_c`,
preventing gradient starvation.

**Composite loss:**

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(y_{\text{pred}}, y_{\text{true}})
    + \lambda_1 \|W\|_1
    + \lambda_g \sum_g \|W_g\|_F
    + \lambda_{\text{act}} \|Z\|_1 \quad (\text{loss\_only mode only})
\]

**LISTA iteration (one step):**

\[
Z_{k+1} = \operatorname{soft\_th}\bigl( Z_k - \eta \cdot D^\top (D \cdot Z_k - X),\; \theta_k \bigr)
\]

### 3.3 Reference implementation walk-through

The core block (`LassoProxConvBlock` in `coder/blocks.py`) implements the
convolution-normalization-threshold sequence:

```python
def _forward(self, x):
    # ── Identity (shortcut) path ──
    identity = self.shortcut(x)          # (B, out_ch, H_out, W_out)

    # ── Main path ──
    h = self.conv(x)                     # (B, out_ch, H_out, W_out)

    if self.config.norm_before_prox:
        h = self.norm(h)                 # (B, out_ch, H_out, W_out)

    # ── Lasso proximal operator (core novelty) ──
    h = self.prox_op(h, self.theta)      # (B, out_ch, H_out, W_out)

    # ── Residual connection ──
    out = identity + h                   # (B, out_ch, H_out, W_out)
    return out
```

The `SoftThreshold.forward()` method (`coder/layers.py`):

```python
def forward(self, x, theta):
    # bf16 safety: cast to float32 for the abs and subtraction
    dtype = x.dtype
    if dtype in (torch.float16, torch.bfloat16):
        x = x.float()
        theta = theta.float()

    out = torch.sign(x) * F.relu(torch.abs(x) - theta)  # same shape as x
    return out.to(dtype)
```

The `AdaptiveSoftThreshold` variant:

```python
def forward(self, x, theta):
    magnitude = torch.abs(x)
    gate = torch.sigmoid(self.alpha * (magnitude - theta))
    out = gate * x
    return out
```

## 4. Tensor shape evolution

Default config (CIFAR-10): `in_channels=3, img_size=32, n_classes=10, stage_channels=(64, 128, 256, 512), stage_depths=(2, 2, 2, 2), dtype=float32`.

| Stage | Shape | Notes |
|---|---|---|
| Input | `(B, 3, 32, 32)` | CIFAR-10 image, uint8 → float32 |
| Stem | `(B, 64, 32, 32)` | Conv 3×3 stride 1, BN, ReLU |
| Stage 1 block 1 | `(B, 64, 32, 32)` | LassoProxConvBlock, stride 1 |
| Stage 1 block 2 | `(B, 64, 32, 32)` | LassoProxConvBlock, stride 1 |
| Stage 2 block 1 | `(B, 128, 16, 16)` | Stride 2, channel doubling |
| Stage 2 block 2 | `(B, 128, 16, 16)` | LassoProxConvBlock, stride 1 |
| Stage 3 block 1 | `(B, 256, 8, 8)` | Stride 2, channel doubling |
| Stage 3 block 2 | `(B, 256, 8, 8)` | LassoProxConvBlock, stride 1 |
| Stage 4 block 1 | `(B, 512, 4, 4)` | Stride 2, channel doubling |
| Stage 4 block 2 | `(B, 512, 4, 4)` | LassoProxConvBlock, stride 1 |
| Global avg pool | `(B, 512, 1, 1)` | AdaptiveAvgPool2d(1) |
| Flatten | `(B, 512)` | — |
| Hidden FC | `(B, 256)` | Linear → ReLU |
| Logits | `(B, 10)` | Linear, no activation |

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| Soft-thresholding as architectural layer (proximal mode) | Only L1 loss penalty (loss_only mode) | Makes sparsity a forward-pass property, not dependent on loss competition | Gradient starvation in dead zone; adaptive variant mitigates |
| Per-channel learnable thresholds `(1,C,1,1)` | Global fixed scalar threshold | Each channel can discover its own sparsity level; more expressive | Threshold oscillation risk in late training; cosine decay mitigates |
| Pre-conv normalization before soft-threshold | No normalization; post-threshold norm | Decouples threshold scale from batch statistics, making θ meaningful across layers | Extra compute; small overhead (1 BN per block) |
| Residual shortcut bypassing threshold | No shortcut (plain conv-threshold) | Ensures gradient flow through the block even when activations are fully zeroed | Slightly more params (1×1 conv for channel dim changes) |
| Group-lasso on filter Frobenius norms | Only element-wise L1 on weights | Structured sparsity lets entire filters be pruned without sparse-matrix hardware | Additional loss hyperparameter to tune |
| LISTA unrolled mode | Standard feedforward only | Connects the network to optimization theory; provides convergence interpretation | Sequential scan prevents GPU parallelization |
| AdamW optimizer with separate param groups | Single-group SGD/Adam | Slower LR for thresholds prevents oscillation; no decay for biases/norm | More complex optimizer setup |
| ReLU in stem, soft-threshold in stages | Soft-threshold everywhere | Stem features should be dense (low-level edges) before sparsification | Inconsistency in activation choice |

## 6. Domain-specific considerations (CV)

### Spatial handling

| Concern | Design decision | Justification |
|---|---|---|
| Input resolution | Flexible stem (no stride for CIFAR-32, stride 2 for ImageNet-224) | Works for both small and large inputs |
| Translation equivariance | All conv ops preserve spatial structure; soft-threshold is element-wise | No global ops until final pooling — standard conv property |
| Multi-scale features | 4-stage pyramid: 1×, 1/2, 1/4, 1/8 resolution | Captures fine + coarse features |
| Scale invariance | Implicit via pyramid (no hard-coded multi-scale) | Sufficient for classification; detection would need FPN |

### Dense vs. global operations

All mixing is **dense** (local 3×3 convolutions) throughout. The Lasso proximal
operator is element-wise and does not benefit from global context. Global pooling
appears only at the final stage before the classifier head. This is intentional:
the lasso integration belongs in the local feature extraction path.

### Soft-thresholding vs. ReLU

| Property | ReLU | Soft-threshold |
|---|---|---|
| Negative values | Clipped to 0 | Preserved (passed with offset if < -θ) |
| Sparsity guarantee | None — any positive activation passes | Activations with `|x| < θ` become exactly zero |
| Gradient for small activations | 1 (for x > 0) | 0 (for `|x| < θ`) — starvation risk |
| Learnable parameter | No | Yes (θ per channel) |

## 7. Known limitations

- **No training experiments exist.** All six performance claims (P1–P8 in
  [BENCHMARKS.md](BENCHMARKS.md#research-quality-evaluation)) are marked
  `TODO: unverified` — the core hypothesis (proximal mode beats loss-only on
  the accuracy-sparsity Pareto frontier) has not been tested. See the
  [research evaluation](BENCHMARKS.md#research-quality-evaluation) section
  for the full gap analysis.
- **Gradient starvation risk.** Standard soft-thresholding has zero gradient
  in the dead zone (`|x| < θ`). The adaptive variant mitigates this but is
  not the proximal operator of any convex regularizer — the theoretical
  connection to Lasso is weakened.
- **LISTA sequential bottleneck.** The LISTA-unrolled mode requires sequential
  computation across iterations, preventing GPU parallelization across layers.
  Default mode is `proximal` which has standard parallel CNN computation.
- **No external baseline comparison.** A standard ResNet-18 with L2-only decay
  has not been implemented or tested. The `loss_only` mode provides the internal
  baseline.
- **No post-training pruning pipeline.** While `count_zero_filters()` can
  identify removable filters, there is no export/retrain step to verify
  compressed-model accuracy.
- **Small-scale only.** The architecture has been validated at CIFAR-10 scale
  (32×32 inputs, ~5M params). ImageNet-scale validation (224×224, deeper stages)
  has not been attempted.
