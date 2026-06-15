# LassoConvNet: A CNN Architecture with Integrated Lasso Regression

## Domain Identification

| Field | Value |
|---|---|
| **Domain** | Computer Vision (CV) |
| **Sub-domain** | Interpretable / Sparse Feature Learning, Image Classification |
| **Core problem** | Learning sparse feature hierarchies by integrating L1 (Lasso) proximal operators directly into the CNN forward pass, not merely as a training-loss penalty term |

---

## Step 1 — Design Constraints and Assumptions

### Upstream Research Contract Status

**Status**: `TODO: upstream research contract missing` — the research stage was initiated but produced only clarification questions. The following design is based on the clarified intent captured in `.clarification.json`.

### Confirmed Design Constraints

| Constraint | Value |
|---|---|
| **Target capability** | Image classification with built-in sparsity (weight + activation) for compression and interpretability |
| **Scale** | CIFAR-10/100 (toy-scale); design generalizes to ImageNet-scale |
| **Hardware target** | Single GPU (A100, RTX 3090) |
| **Starting point** | Pure novel design: a CNN augmented with unrolled lasso proximal operators |
| **Required baseline** | Standard CNN of equivalent depth/width (e.g., ResNet-18, VGG-11) with only loss-based L2 decay |
| **Falsification target** | If the sparsity rate (fraction of near-zero activations) is < 20% at convergence OR accuracy drops > 3% vs. the matched-width baseline, the architectural integration provides no benefit over vanilla L1-on-loss |

### Assumptions Carried Forward

1. L1 penalty is applied during training (not post-hoc pruning).
2. Target task is image classification (CIFAR-10/ImageNet scale).
3. Input is 2D image data; design generalizes to 1D/3D.
4. Both element-wise L1 on kernel weights and group-lasso on filters/channels are supported.

---

## Step 2 — ModelConfig Dataclass

```python
from dataclasses import dataclass, field
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
    threshold_learnable: bool = True    # learn thresholds per channel via backprop
    use_group_lasso: bool = True        # apply group-lasso on output filters
    group_lasso_strength: float = 1e-4
    group_size: int = 8                 # filter group size for structured sparsity

    # ── Proximal operator ──
    proximal_type: str = "soft"         # "soft" (L1) | "hard" (L0 approx) | "adaptive"
    norm_before_prox: bool = True       # BN/LN before soft-thresholding
    norm_type: str = "batch_norm"       # "batch_norm" | "layer_norm" | "none"

    # ── Loss terms ──
    l1_weight_decay: float = 1e-5      # element-wise L1 on conv kernel weights
    l2_weight_decay: float = 0.0       # standard L2 (set to 0 when using L1)
    activation_sparsity_weight: float = 1e-4  # auxiliary L1 on activations (if loss_only mode)
    l1_decay_schedule: str = "constant" # "constant" | "warmup" | "cosine_decay"

    # ── Regularization (standard) ──
    dropout: float = 0.0
    use_bias: bool = False
    dtype: str = "float32"

    # ── LISTA unrolled options (lista_unrolled mode) ──
    lista_iters: int = 6          # number of unrolled ISTA iterations (= num layers)
    lista_dictionary_size: int = 256  # overcomplete dictionary atoms
    lista_tie_weights: bool = True    # tied vs untied weights across iterations

    # ── Classification head ──
    head_hidden_dim: int = 256
    global_pool: str = "avg"       # "avg" | "max" | "none"
```

---

## Step 3 — Core Novel Block: LassoProxConvBlock

### 3.1 Pseudocode

```python
def lasso_prox_conv_block(x, config, theta=None):
    """
    Lasso-integrated convolutional block.

    Forward pass:
        y = Conv2D(x) + bias
        a = Norm(y)
        z = soft_threshold(a, theta)     ← Lasso proximal operator integrated in forward
        return z

    The soft-thresholding operator is the proximal map for L1:
        prox_{lambda * L1}(x) = sign(x) * max(|x| - lambda, 0)

    When theta=0, this block reduces to Conv+BN+ReLU.
    """
    B, C, H, W = x.shape

    # ── 1. Convolution ──
    h = F.conv2d(x, weight, bias, padding=config.kernel_size // 2)

    # ── 2. Normalization (before prox — stabilizes thresholding scale) ──
    if config.norm_before_prox:
        if config.norm_type == "batch_norm":
            h = F.batch_norm(h, running_mean, running_var, weight, bias)
        elif config.norm_type == "layer_norm":
            h = layer_norm(h)  # channel-wise LN over (C, H, W)

    # ── 3. Lasso Proximal Operator (soft-thresholding) ──
    # theta shape: (1, C, 1, 1) — one threshold per output channel
    # This IS the architectural lasso integration.
    z = torch.sign(h) * F.relu(torch.abs(h) - theta)

    # ── 4. Optional: group-lasso structured sparsity ──
    if config.use_group_lasso and config.lasso_mode != "loss_only":
        # Group-wise soft-thresholding on filter groups
        # Residual connection handled outside this block
        pass  # penalty computed in loss, not in forward

    return z


def adaptive_soft_threshold(x, theta, alpha=0.1):
    """
    Adaptive soft-thresholding with a smooth transition.

    Standard soft-threshold produces hard kinks at |x|=theta.
    Adaptive version uses a smooth gating:
        g(x) = sigmoid(alpha * (|x| - theta))
        z = g(x) * x + (1 - g(x)) * 0
    which is differentiable everywhere and avoids gradient starvation.
    """
    magnitude = torch.abs(x)
    gate = torch.sigmoid(alpha * (magnitude - theta))
    return gate * x   # equivalent: gate * x + (1-gate) * 0
```

### 3.2 LISTA-Unrolled Variant (Full Integration)

```python
def lista_conv_encoder(x, config):
    """
    Unroll ISTA iterations into a feedforward network.

    Solves:  min_Z  0.5 * ||X - D * Z||^2  + lambda * ||Z||_1

    Each layer computes one ISTA step:
        Z_{k+1} = soft_threshold(Z_k - eta * D^T * (D * Z_k - X),  theta)

    When tied_weights=True, the same D (conv weight) is reused.
    When tied_weights=False, each layer has independent D_k.

    This turns the iterative optimization algorithm into a
    feedforward network with L layers = L ISTA iterations.
    """
    # Initial estimate: encoding via transposed convolution
    Z = F.conv2d(x, config.W_encode, padding=...)   # D^T * X

    for k in range(config.lista_iters):
        # Reconstruction residual: D * Z - X
        residual = F.conv_transpose2d(Z, config.W_dict[k]) - x  # tied? use shared W

        # Gradient step: Z - eta * D^T * residual
        grad = F.conv2d(residual, config.W_dict[k].T)  # D^T * residual
        Z = Z - config.eta * grad

        # Proximal operator (lasso!)
        Z = soft_threshold(Z, config.theta[k])

    return Z
```

### 3.3 Inductive Bias Statements

| Design choice | Inductive bias statement |
|---|---|
| **Soft-thresholding after Conv+BN** | Sparse features are more robust and interpretable; thresholding the activations is the proximal operator for an L1-regularized convolutional optimization problem |
| **Learnable per-channel thresholds** | Each feature channel has a different sparsity level — some channels encode sparse edge detectors, others encode denser texture patterns |
| **Pre-conv normalization before soft-thresholding** | Normalizing the pre-activation distribution prevents the threshold from being scale-dependent, decoupling sparsity from batch statistics |
| **Group-lasso on filters** | Structured sparsity (dropping entire filters) is preferable for model compression over unstructured element-wise sparsity, which requires specialized hardware |
| **LISTA unrolling** | Deep feedforward computation can be interpreted as optimizing a sparse coding objective; unrolled ISTA provides theoretical convergence guarantees for the network's forward pass |
| **L1 weight decay on kernels (loss term)** | L1 on weights complements activation sparsity by also encouraging the dictionary atoms (conv filters) to be sparse, reducing model size |

---

## Step 4 — Architecture Diagram (ASCII)

### Main Backbone (Proximal Mode)

```
Input Image (B, 3, H, W)
         │
         ▼
  ┌─────────────────────┐
  │  Stem Conv 3×3, s2   │  → (B, C1, H/2, W/2)
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
     ║  │  │   → (optional 1×1 proj)      │   │ ║
     ║  │  └──────────────────────────────┘   │ ║
     ║  │          ↺ + residual shortcut      │ ║
     ║  └─────────────────────────────────────┘ ║
     ╚══════╤═══════════════════════════════════╝
            │  Downsample (stride 2 conv or pool)
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

### LISTA Unrolled Mode (Full Lasso Integration)

```
Input Image X (B, 3, H, W)
         │
         ▼
  ┌─────────────────────┐
  │  Patch Embed / Conv │  → initial encoding Z_0
  └─────────┬───────────┘
            │
   ╔════════╧══════════════════════════════════════╗
   ║         LISTA Unrolled Iterations × K         ║
   ║                                               ║
   ║  Z_1 = soft_th( Z_0 - η·D^T(D·Z_0 - X), θ_1 ) ║
   ║  Z_2 = soft_th( Z_1 - η·D^T(D·Z_1 - X), θ_2 ) ║
   ║  ...                                          ║
   ║  Z_K = soft_th(...)                           ║
   ║                                               ║
   ║  Each iteration = one conv layer               ║
   ║  D = conv kernel (shared or per-iteration)     ║
   ║  soft_th = lasso proximal operator             ║
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

---

## Step 5 — Research-to-Architecture Traceability

| Research contract item | Architecture decision | Evidence status | Validation hook |
|---|---|---|---|
| "Lasso regression integrated into CNN" | `lasso_mode="proximal"` — soft-thresholding applied as architectural layer, not just loss penalty | `hypothesis` | Ablation A1: swap `proximal` → `loss_only`, measure sparsity rate and accuracy delta |
| "Apply penalty" (on weights for sparsity/compression) | `l1_weight_decay` on kernel weights + `group_lasso_strength` on filter groups; both in loss | `grounded` (L1 weight decay is standard) | Compare weight sparsity histogram with & without L1 decay |
| "Group lasso on filters" | `use_group_lasso=True` with group size hyperparameter; structured sparsity in loss | `hypothesis` | Ablation A3: `use_group_lasso` True → False; count zero-filters at convergence |
| Sparse feature hierarchy | Per-channel learnable thresholds (`threshold_learnable=True`, shape (1,C,1,1)) in every LassoProxConvBlock | `hypothesis` | Ablation A2: `threshold_learnable` True → False (fixed threshold); compare per-channel sparsity variance |
| Interpretability via sparse features | Soft-thresholding produces exactly-zero activations, making feature maps sparser and more interpretable | `hypothesis` | Visualization: max-activation per channel, compare standard ReLU vs. soft-threshold |
| Model compression | Group-lasso on filters + L1 on weights → entire filters become zero → removable at export | `grounded` (well-known in pruning literature) | Count removable filters (% of total); compare inference speed after pruning |
| Unrolled optimization interpretation | `lasso_mode="lista_unrolled"` — each block = one ISTA iteration with tied/untied weights | `hypothesis` | Ablation A5: `lista_tie_weights` True → False; measure accuracy improvement vs. parameter count |

---

## Step 6 — Domain-Specific Considerations (CV)

### 6.1 Spatial Handling

| Concern | Design decision | Justification |
|---|---|---|
| Input resolution | Flexible: stem conv with stride 2, then 4 stages with progressive downsampling | Works for CIFAR-32×32 (shallow) to ImageNet-224×224 (deeper stages) |
| Translation equivariance | All ops are convolution-based; soft-threshold is element-wise and preserves translation equivariance | Critical for convolutional models; no global ops until final pooling |
| Multi-scale features | Standard ResNet-style pyramid: 4 stages at 1/2, 1/4, 1/8, 1/16 resolution | Captures both fine and coarse features; standard practice |
| Scale invariance | Not hard-coded; relies on pyramid (implicit) | A fixed-size pyramid is sufficient for classification; segmentation/ detection would need explicit FPN |

### 6.2 Dense vs. Global Operations

All mixing is **dense** (local convolutions, 3×3 kernels) throughout. No global attention. This is intentional:
- Lasso proximal operator is element-wise and does not benefit from global context
- Global pooling only at the very end
- The LISTA variant's dictionary is convolutional (local), not global

### 6.3 Why Soft-Thresholding Instead of ReLU?

ReLU sets negative values to 0. Soft-thresholding sets values in `[-θ, θ]` to 0 — this is a **continuous sparsification** with a dead zone. The key difference:

| Property | ReLU | Soft-threshold |
|---|---|---|
| Negative values | Clipped to 0 | Preserved (less than -θ passed through with offset) |
| Sparsity guarantee | None — any positive activation passes | Activations with magnitude < θ become exactly zero |
| Gradient for small activations | 1 (for x > 0) | 0 (for |x| < θ) — gradient starvation risk |
| Learnable parameter | No | Yes (θ per channel) |

The gradient starvation in the dead zone (`|x| < θ`) is a real concern — see Implementation Risks.

---

## Step 7 — Implementation Risks

### Risk 1: Gradient Starvation in Soft-Thresholding Dead Zone

When `|x| < θ`, the gradient of soft-thresholding is 0. During training, if activations fall into the dead zone, they stop receiving gradient signal entirely. This can cause:

- **Neuron death**: once an entire channel's activations fall below θ, it never recovers.
- **Threshold collapse**: if θ becomes too large, all activations are zeroed.

**Mitigations proposed:**
1. **Adaptive soft-threshold** (Section 3.1): use a sigmoid-gated version that has non-zero gradient everywhere.
2. **Warmup schedule**: start θ near 0 and gradually increase during training (like L1 warmup).
3. **Residual connections**: bypassing the threshold with a residual path ensures gradient flow around the dead zone.

### Risk 2: Numerical Instability with Combined L1/L2

Using both `l1_weight_decay > 0` and `l2_weight_decay > 0` creates a non-smooth objective (L1 is non-differentiable at 0). With bfloat16 training, the proximal operator's kink at |x|=θ can cause gradient underflow.

**Mitigation:** Set `l2_weight_decay=0` when using L1 (recommended in config). Use float32 for the soft-threshold operation even when training in bfloat16.

### Risk 3: Threshold Oscillation in Late Training

With learnable thresholds and a constant L1 penalty, the threshold θ and weight magnitudes can oscillate in a feedback loop: high θ → sparser activations → weaker gradients → thresholds drift.

**Mitigation:** Cosine decay schedule for L1 penalty (`l1_decay_schedule="cosine_decay"`) and lower-bound clamp on θ (e.g., `theta >= 1e-6`).

### Risk 4: LISTA Unrolling → Sequential Scan Bottleneck

The LISTA mode requires sequential computation: iteration k+1 depends on iteration k. This prevents parallelization across layers on GPU, unlike a standard feedforward CNN where all layers in a stage can be fused.

**Mitigation:** Reserve LISTA mode for small-scale experiments. Default mode is `proximal` which has standard parallel CNN computation.

---

## Step 8 — Loss Function Formulation

The total loss integrates lasso at three levels:

```
L_total = L_ce(y_true, y_pred)                     # Cross-entropy (task)
        + λ_weight · ||W||_1                       # Element-wise L1 on weights
        + λ_group · Σ_g ||W_g||_F                  # Group lasso on filters (Frobenius norm per group)
        + λ_act · ||Z||_1                          # Auxiliary L1 on activations (if loss_only mode)
```

Where:
- `||W||_1` = sum of absolute values of all convolution + FC weights
- `||W_g||_F` = Frobenius norm of filter group g (group = `group_size` consecutive filters)
- `||Z||_1` = sum of absolute values of activations (feature maps)

In `proximal` mode, `λ_act·||Z||_1` is **not** needed because the soft-thresholding operation **in the forward pass** already induces activation sparsity. The L1 on weights and group lasso remain as loss terms.

In `loss_only` mode, all three penalties are in the loss only — no architectural change to the forward pass.

### Gradient Flow for Proximal Mode

| Operation | Forward | Backward |
|---|---|---|
| `z = sign(h) · max(|h| - θ, 0)` | `∂L/∂θ = -Σ sign(h) · 1_{|h|>θ}` |
| | `∂L/∂h = ∂L/∂z · 1_{|h|>θ}` |

The indicator `1_{|h|>θ}` means gradients flow **only** through active (non-thresholded) units. Dead zones get zero gradient — which is why the adaptive variant or residual paths are important.

---

## Step 9 — Suggested Ablations

| # | Ablation | Config field | Baseline value | Ablated value | Hypothesis tested | Expected metric movement | Failure interpretation | Owning stage |
|---|---|---|---|---|---|---|---|---|
| A1 | Lasso mode: proximal → loss_only | `lasso_mode` | `"proximal"` | `"loss_only"` | Architectural soft-thresholding produces sparser feature maps than loss-only L1 at equal accuracy | Accuracy: ↓0–1%; Activation sparsity: ↓20–40% | If accuracy drops >3%, architectural prox hurts; if sparsity doesn't drop, loss alone is sufficient | `ml-architect` |
| A2 | Learnable → fixed thresholds | `threshold_learnable` | `True` | `False` | Per-channel adaptive thresholds improve accuracy–sparsity tradeoff vs. a global fixed threshold | Accuracy: ↓0.5–1.5%; Sparsity: ↓5–10% | If accuracy unchanged, per-channel thresholds are unnecessary complexity | `ml-architect` |
| A3 | Group lasso on/off | `use_group_lasso` | `True` | `False` | Group-lasso induces structured (filter-level) sparsity without harming accuracy | Filter sparsity: ↓0→>30%; Accuracy: ↓0–1% | If accuracy drops >2%, group lasso penalty is too strong; reduce `group_lasso_strength` | `ml-research` |
| A4 | Adaptive vs. vanilla soft-threshold | `proximal_type` | `"soft"` | `"adaptive"` | Adaptive sigmoid-gated threshold prevents gradient starvation and improves accuracy | Accuracy: ↑0.5–1%; Dead channels: ↓ | If adaptive variant underperforms, gradient starvation is not a problem in practice | `ml-architect` |
| A5 | LISTA tied vs. untied weights | `lista_tie_weights` | `True` | `False` | Untied weights increase capacity per ISTA iteration, improving reconstruction quality | Accuracy: ↑1–2%; Params: ↑2× | If accuracy does not improve, tied weights are sufficient (supports the LISTA theory) | `ml-architect` |
| A6 | Norm before prox on/off | `norm_before_prox` | `True` | `False` | Normalization before thresholding stabilizes the scale so θ is meaningful across layers | Accuracy: ↓1–2% without norm; Training loss variance: ↑ | If accuracy unchanged, normalization is unnecessary overhead | `ml-validator` |

### Ablation Ordering

Run in this order if the model does not converge or sparsity is below target:

1. **A4** (adaptive threshold) — fixes gradient starvation, the most likely training failure
2. **A3** (remove group lasso) — removes the most aggressive structured penalty
3. **A1** (fall back to loss-only) — removes the architectural integration entirely
4. **A6** (remove pre-prox norm) — reduces compute
5. **A2** (fixed thresholds) — reduces complexity
6. **A5** (untie LISTA weights) — only relevant for LISTA mode

---

## Step 10 — Implementation Blueprint

### File Structure

```
lasso_convnet/
├── config.py              ← LassoConvConfig dataclass
├── blocks/
│   ├── __init__.py
│   ├── lasso_prox.py      ← LassoProxConvBlock (core novel block)
│   ├── adaptive_thresh.py ← adaptive soft-threshold (sigmoid-gated)
│   ├── lista_encoder.py   ← LISTA-unrolled encoder
│   └── group_lasso.py     ← group-lasso penalty helper
├── backbone.py            ← LassoConvNet backbone (4-stage pyramid)
├── classifier.py          ← Classification head
├── loss.py                ← Composite loss (CE + L1_weight + group_lasso + L1_act)
└── train.py               ← Training loop with lasso schedule
```

### Key Training Considerations

1. **Threshold initialization**: Start θ at 0.01 (very small) so early training is close to standard CNN. Ramp up via cosine schedule over first 50% of epochs.
2. **L1 warmup**: Apply a linear warmup for L1 weight decay over first 5 epochs to avoid overwhelming the CE loss.
3. **Optimizer**: AdamW with separate parameter groups:
   - Group 1: conv weights — `l1_weight_decay` + `l2_weight_decay`
   - Group 2: thresholds — learning rate × 0.1 (slower to avoid oscillation)
   - Group 3: biases and norm params — no weight decay
4. **Export/pruning**: After training, filters where all group members are zero can be removed; the model can be exported without the thresholding ops (set θ→0).

---

## Output Checklist

- [x] Domain identified (CV — sparse feature learning)
- [x] Upstream research lifecycle contract read or marked missing (marked `TODO: upstream research contract missing`)
- [x] ModelConfig dataclass with all hyperparameters (`LassoConvConfig`)
- [x] Pseudocode for the novel block (LassoProxConvBlock, LISTA encoder)
- [x] ASCII architecture diagram (Proximal mode + LISTA mode)
- [x] Inductive bias justification (one sentence per decision)
- [x] Research-to-architecture traceability table included
- [x] Claims labeled as `grounded`, `hypothesis`, or `TODO: unverified`
- [x] Domain-specific considerations addressed (CV: spatial handling, equivariance, pyramid)
- [x] Implementation risk flags (4 risks with mitigations)
- [x] Baseline and evaluation requirements carried forward
- [x] Suggested ablations (6 ablations, each = single ModelConfig field change tied to a hypothesis)
