# API Reference

## `config.py` — LassoConvConfig

### `class LassoConvConfig`
Configuration dataclass for all LassoConvNet hyperparameters.

**Fields:**

| Field | Type | Default | Rationale |
|---|---|---|---|
| `in_channels` | `int` | `3` | RGB input images |
| `img_size` | `int` | `32` | CIFAR default; use 224 for ImageNet |
| `n_classes` | `int` | `10` | CIFAR-10 number of classes |
| `base_channels` | `int` | `64` | Initial feature map width |
| `stage_depths` | `Tuple[int, ...]` | `(2, 2, 2, 2)` | Number of blocks per stage |
| `stage_channels` | `Tuple[int, ...]` | `(64, 128, 256, 512)` | Output channels per stage |
| `kernel_size` | `int` | `3` | Conv kernel size throughout |
| `downsample` | `str` | `"stride"` | Spatial reduction: `"stride"` or `"pool"` |
| `lasso_mode` | `str` | `"proximal"` | `"loss_only"` / `"proximal"` / `"lista_unrolled"` |
| `threshold_init` | `float` | `0.01` | Initial soft-threshold value |
| `threshold_learnable` | `bool` | `True` | Whether theta is learnable per channel |
| `use_group_lasso` | `bool` | `True` | Apply group-lasso structured sparsity |
| `group_lasso_strength` | `float` | `1e-4` | Group-lasso regularization coefficient |
| `group_size` | `int` | `8` | Filters per group for structured sparsity |
| `proximal_type` | `str` | `"soft"` | `"soft"` / `"adaptive"` / `"hard"` |
| `norm_before_prox` | `bool` | `True` | Normalize before thresholding |
| `norm_type` | `str` | `"batch_norm"` | `"batch_norm"` / `"layer_norm"` / `"none"` |
| `l1_weight_decay` | `float` | `1e-5` | Element-wise L1 on conv kernel weights |
| `l2_weight_decay` | `float` | `0.0` | Standard L2 (set to 0 when using L1) |
| `activation_sparsity_weight` | `float` | `1e-4` | Auxiliary L1 on activations (loss_only mode) |
| `l1_decay_schedule` | `str` | `"constant"` | `"constant"` / `"warmup"` / `"cosine_decay"` |
| `dropout` | `float` | `0.0` | Dropout rate in classifier head |
| `use_bias` | `bool` | `False` | Use bias in conv layers |
| `dtype` | `str` | `"float32"` | Model dtype |
| `lista_iters` | `int` | `6` | Number of unrolled ISTA iterations |
| `lista_dictionary_size` | `int` | `256` | Overcomplete dictionary atoms |
| `lista_tie_weights` | `bool` | `True` | Tied vs untied weights across iterations |
| `head_hidden_dim` | `int` | `256` | MLP classifier hidden dimension |
| `global_pool` | `str` | `"avg"` | `"avg"` / `"max"` |
| `lr` | `float` | `1e-3` | Peak learning rate |
| `lr_threshold` | `float` | `1e-4` | Slower LR for threshold params |
| `l1_warmup_epochs` | `int` | `5` | Linear warmup epochs for L1 penalty |
| `theta_min` | `float` | `1e-6` | Lower bound clamp on theta |

---

## `layers.py` — Proximal Operators, Normalization, Sparsity Utils

### `class BaseProxOperator(ABC, nn.Module)`
Abstract base class for Lasso proximal operators.

**Methods:**
- `forward(x: Tensor, theta: Tensor) -> Tensor` — Abstract. Apply proximal operator.

### `class SoftThreshold(BaseProxOperator)`
Standard L1 proximal operator (soft-thresholding).

**Forward:** `out = sign(x) * relu(|x| - theta)`

When `theta=0`, this reduces to identity. When `theta >> |x|`, output is all zeros.

**bf16/fp16 safety:** casts inputs to float32 internally, casts output back to original dtype.

**Args:**
- `x: (B, C, H, W)` or any shape — input tensor
- `theta: (1, C, 1, 1)` — threshold, broadcastable to x shape

**Returns:** same shape as x

### `class AdaptiveSoftThreshold(BaseProxOperator)`
Smooth approximation of soft-thresholding using sigmoid gating.

**Forward:** `gate = sigmoid(alpha * (|x| - theta)); out = gate * x`

- Non-zero gradient for `|x| < theta` (unlike standard soft-thresholding)
- C^inf smooth everywhere
- When `alpha → inf`, converges to standard soft-thresholding

**Constructor:** `AdaptiveSoftThreshold(alpha: float = 0.1)`

### `class HardThreshold(BaseProxOperator)`
Hard-thresholding (L0 approximation).

**Forward:** `out = x if |x| > theta else 0`

**Warning:** Gradient is zero almost everywhere. Not recommended for training without gradient surrogates.

### `def build_prox_operator(proximal_type: str, **kwargs) -> BaseProxOperator`
Factory function for proximal operators.

**Args:**
- `proximal_type: str` — `"soft"`, `"adaptive"`, or `"hard"`
- `**kwargs` — passed to operator constructor (e.g., `alpha` for adaptive)

**Returns:** `SoftThreshold`, `AdaptiveSoftThreshold`, or `HardThreshold`

**Raises:** `ValueError` for unknown type

### `def get_normalization_layer(norm_type: str, num_features: int) -> nn.Module`
Get a 2D normalization layer.

**Args:**
- `norm_type: str` — `"batch_norm"`, `"layer_norm"`, or `"none"`
- `num_features: int` — number of channels C

**Returns:** `nn.BatchNorm2d`, `nn.GroupNorm(1, C)` (layer norm), or `nn.Identity`

### `def compute_sparsity_ratio(x: Tensor, eps: float = 1e-6) -> Tensor`
Compute the fraction of near-zero elements.

**Args:**
- `x: Tensor` — any shape
- `eps: float` — threshold below which an element is "zero"

**Returns:** scalar tensor — `mean(|x| < eps)`

### `def compute_channel_sparsity(x: Tensor, eps: float = 1e-6) -> Tensor`
Compute per-channel sparsity ratio.

**Args:**
- `x: Tensor (B, C, H, W)` — activation tensor
- `eps: float` — threshold for "zero"

**Returns:** `(C,)` tensor — sparsity ratio per channel

---

## `blocks.py` — LassoProxConvBlock, ListaConvLayer, ListaEncoder

### `class LassoProxConvBlock(nn.Module)`
Convolutional block with integrated Lasso proximal operator.

**Forward:** Conv2d → BatchNorm → SoftThreshold(theta) → + residual

**Inductive bias:** "Sparse features are more robust and interpretable; thresholding activations is the proximal operator for an L1-regularized convolutional optimization problem."

**Constructor:** `LassoProxConvBlock(in_channels: int, out_channels: int, stride: int = 1, config: Optional[LassoConvConfig] = None)`

**Fields:**
- `self.conv: nn.Conv2d` — 3x3 convolution
- `self.norm: nn.Module` — normalization (BN/LN/identity)
- `self.theta: nn.Parameter` — shape `(1, out_channels, 1, 1)`, per-channel threshold
- `self.prox_op: BaseProxOperator` — soft/adaptive/hard threshold
- `self.shortcut: nn.Module` — 1x1 conv or identity for residual connection

**Methods:**
- `forward(x: Tensor, use_checkpoint: bool = False) -> Tensor`
  - `x: (B, in_channels, H, W)`
  - Returns: `(B, out_channels, H/stride, W/stride)`
  - If `use_checkpoint=True` and in training mode, uses gradient checkpointing

### `class ListaConvLayer(nn.Module)`
One ISTA iteration unrolled as a feedforward layer.

**Forward:** `Z_{k+1} = soft_th(Z_k - eta * D^T * (D * Z_k - X), theta)`

**Constructor:** `ListaConvLayer(in_channels: int, dict_channels: int, kernel_size: int = 3, tied: bool = False, shared_dict: Optional[nn.Conv2d] = None)`

**Fields:**
- `self.D: nn.Conv2d` — dictionary (decoding), shape `(in_ch, dict_ch, K, K)`
- `self.Dt: nn.Conv2d` — transpose (encoding), shape `(dict_ch, in_ch, K, K)`
- `self.eta: nn.Parameter` — learnable step size
- `self.theta: nn.Parameter` — learnable threshold

**Methods:**
- `forward(Z: Tensor, X: Tensor) -> Tensor`
  - `Z: (B, dict_channels, H, W)` — current sparse code
  - `X: (B, in_channels, H, W)` — original input to reconstruct
  - Returns: `(B, dict_channels, H, W)` — updated sparse code

### `class ListaEncoder(nn.Module)`
Full LISTA network: unrolls K ISTA iterations for convolutional sparse coding.

**Inductive bias:** "Deep feedforward computation can be interpreted as optimizing a sparse coding objective; unrolled ISTA provides theoretical convergence guarantees."

**Constructor:** `ListaEncoder(config: LassoConvConfig)`

**Methods:**
- `forward(x: Tensor) -> Tensor`
  - `x: (B, C, H, W)` — stem output features
  - Returns: `(B, dict_channels, H, W)` — sparse code after K iterations

---

## `backbone.py` — LassoConvStage, LassoConvNetBackbone

### `class LassoConvStage(nn.Module)`
One spatial stage of the LassoConvNet pyramid.

Contains N `LassoProxConvBlock`s. The first block downsamples via stride-2 if `downsample_first=True`.

**Constructor:** `LassoConvStage(in_channels: int, out_channels: int, depth: int, downsample_first: bool, config: LassoConvConfig)`

**Methods:**
- `forward(x: Tensor, use_checkpoint: bool = False) -> Tensor`
  - `x: (B, in_channels, H, W)`
  - Returns: `(B, out_channels, H_out, W_out)` — H_out = H/2 if downsampled

### `class LassoConvNetBackbone(nn.Module)`
Full 4-stage pyramid backbone with lasso-integrated convolutions.

**Architecture:**
- Stem: 3x3 Conv → BN → ReLU
- Stage 1: depth[0] blocks, C1 channels
- Stage 2: depth[1] blocks, C2 channels (H/2)
- Stage 3: depth[2] blocks, C3 channels (H/4)
- Stage 4: depth[3] blocks, C4 channels (H/8)
- LISTA encoder (if `lasso_mode="lista_unrolled"`)

**Constructor:** `LassoConvNetBackbone(config: LassoConvConfig)`

**Methods:**
- `forward(x: Tensor, use_checkpoint: bool = False) -> Tensor`
  - `x: (B, in_channels, H, W)`
  - Returns (proximal mode): `(B, C4, H/8, W/8)`
  - Returns (LISTA mode): `(B, dict_size, H, W)`

### `def make_downsample_layer(mode: str, in_channels: int, out_channels: int, stride: int) -> nn.Module`
Create a downsampling layer for "stride" or "pool" mode.

---

## `model.py` — LassoConvNet, ClassifierHead

### `class ClassifierHead(nn.Module)`
Classification head with optional global pooling.

**Architecture:** GlobalPool → Flatten → Linear → ReLU → Dropout → Linear → Logits

**Constructor:** `ClassifierHead(config: LassoConvConfig)`

**Methods:**
- `forward(x: Tensor) -> Tensor`
  - `x: (B, C, H, W)` — feature maps
  - Returns: `(B, n_classes)` — logits

### `class LassoConvNet(nn.Module)`
Complete LassoConvNet model: backbone + classifier head.

**Three operating modes:**
1. `loss_only` — Standard CNN + L1 loss penalty (baseline)
2. `proximal` — Soft-thresholding in every conv block (architectural)
3. `lista_unrolled` — Unrolled ISTA iterations for sparse coding

**Constructor:** `LassoConvNet(config: LassoConvConfig)`

**Methods:**

- `forward(x: Tensor, use_checkpoint: bool = False) -> Tensor`
  - `x: (B, in_channels, H, W)` — input image
  - Returns: `(B, n_classes)` — class logits
  - Shape guarantee: Input `(B, 3, 32, 32)`, Output `(B, 10)` for CIFAR-10
  - NaN guard: `assert not torch.isnan(logits).any()` when in training mode

- `get_sparsity_ratio(eps: float = 1e-6) -> Optional[Tensor]`
  - Fraction of near-zero activations from the last forward pass
  - Returns scalar tensor, or `None` if no forward pass has been run

- `get_channel_sparsity(eps: float = 1e-6) -> Optional[Tensor]`
  - Per-channel sparsity from the last forward pass
  - Returns `(C,)` tensor, or `None`

- `clamp_thresholds(min_val: float = 1e-6)`
  - Clamp all learnable thresholds to `[min_val, inf)`

- `get_thresholds() -> dict`
  - Returns `{param_name: threshold_tensor}` for all theta parameters

- `set_check_nan(enabled: bool)`
  - Enable/disable NaN checking in all blocks

- `count_zero_filters(eps: float = 1e-6) -> int`
  - Count conv filters whose weights are all near-zero (prunable)

### `def count_params(model: nn.Module, verbose: bool = True) -> Tuple[int, int]`
Count total and trainable parameters.

**Args:**
- `model: nn.Module` — any PyTorch model
- `verbose: bool` — if True, prints counts

**Returns:** `(total_params, trainable_params)`

---

## `loss.py` — Lasso Loss Functions

### `def l1_weight_penalty(model: nn.Module, strength: float = 1e-5, exclude_types: tuple = (...)) -> Tensor`
Element-wise L1 penalty on all weight parameters (dim >= 2).

Excludes biases and normalization parameters. bf16 safe (casts to float32).

**Returns:** scalar tensor — `strength * sum(|w_i|)`

### `def group_lasso_penalty(model: nn.Module, group_size: int = 8, strength: float = 1e-4) -> Tensor`
Group lasso penalty on conv filter groups.

Groups consecutive `group_size` output filters and penalizes each group's Frobenius norm: `strength * sum_g ||W_g||_F`.

**Returns:** scalar tensor

### `def activation_l1_penalty(model: nn.Module, strength: float = 1e-4, capture_hook: Optional[str] = None) -> Tensor`
Auxiliary L1 penalty on feature map activations (loss_only mode only).

In proximal mode, activation sparsity is induced architecturally. This function returns 0 unless the caller provides hooks.

**Returns:** scalar tensor (0.0 if not captured, otherwise the L1 penalty)

### `def lasso_total_loss(ce_loss: Tensor, model: nn.Module, config: LassoConvConfig) -> Tensor`
Composite loss with all Lasso penalties.

**Formula:** `L = CE + l1 * ||W||_1 + gl * sum_g ||W_g||_F [+ act * ||Z||_1]`

Mode-specific behavior:
- `proximal`: L1 on weights + group lasso (activation sparsity from architecture)
- `loss_only`: L1 on weights + group lasso + activation L1
- `lista_unrolled`: L1 on weights + group lasso

**Raises:** `ValueError` if NaN detected in total loss.
