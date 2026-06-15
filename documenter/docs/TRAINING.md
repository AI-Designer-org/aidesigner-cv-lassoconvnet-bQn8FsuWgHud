# Training & Reproduction

> **IMPORTANT**: No training loop has been implemented yet. The following is the
> **recommended recipe** derived from the architect's design document
> (`architect/architecture_design.md`). It has not been validated experimentally.
> All claims about expected convergence behavior are `TODO: unverified`.

## Environment

- Python: 3.10+
- PyTorch: 2.0+
- CUDA: 11.7+, tested on A100 or RTX 3090
- Other: None (pure PyTorch, no custom CUDA kernels)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Default hyperparameters

All hyperparameters are defined in the `LassoConvConfig` dataclass
(`coder/config.py`).

| Field | Default | Rationale |
|---|---|---|
| `in_channels` | 3 | RGB images |
| `img_size` | 32 | CIFAR-10 default; set to 224 for ImageNet |
| `n_classes` | 10 | CIFAR-10 |
| `base_channels` | 64 | Initial feature width |
| `stage_channels` | (64, 128, 256, 512) | ResNet-style channel progression |
| `stage_depths` | (2, 2, 2, 2) | 2 blocks per stage; total 8 conv blocks |
| `kernel_size` | 3 | Standard 3x3 convolution |
| `downsample` | "stride" | Strided conv for spatial reduction (vs. pooling) |
| `lasso_mode` | "proximal" | Architectural soft-thresholding (core mode) |
| `threshold_init` | 0.01 | Small initial theta so early training ≈ standard CNN |
| `threshold_learnable` | True | Per-channel thresholds adapt during training |
| `proximal_type` | "soft" | Standard L1 proximal operator |
| `norm_before_prox` | True | BN before thresholding stabilizes scale |
| `norm_type` | "batch_norm" | Standard BatchNorm2d |
| `l1_weight_decay` | 1e-5 | Unstructured sparsity on kernel weights |
| `l2_weight_decay` | 0.0 | Set to 0 when using L1 (non-smooth objective) |
| `use_group_lasso` | True | Structured filter-group sparsity |
| `group_lasso_strength` | 1e-4 | Frobenius norm penalty per filter group |
| `group_size` | 8 | Consecutive filters per group |
| `head_hidden_dim` | 256 | MLP classifier hidden dimension |
| `lr` | 1e-3 | Peak learning rate |
| `lr_threshold` | 1e-4 | Slower LR for threshold params (avoids oscillation) |
| `l1_warmup_epochs` | 5 | Linear warmup for L1 weight decay |
| `theta_min` | 1e-6 | Lower bound clamp on thresholds |

## Recommended training recipe (CIFAR-10)

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Three parameter groups (see below) |
| Peak LR | 1e-3 | Linear warmup over first 5 epochs, cosine decay to 0 |
| Threshold LR | 1e-4 | 10x slower than backbone to prevent oscillation |
| Batch size | 128 | Gradient accumulation if GPU memory limited |
| Weight decay (conv weights) | L1: 1e-5 | Applied via `l1_weight_penalty`, not optimizer weight_decay |
| Weight decay (thresholds) | 0.0 | No decay on theta parameters |
| Weight decay (bias/norm) | 0.0 | Standard practice: no decay on bias/norm |
| Grad clip | 1.0 | Global norm clipping |
| Precision | float32 | bf16 supported for inference; float32 recommended for training due to soft-threshold precision sensitivity |
| Epochs | 200 | Full CIFAR-10 training budget |
| Threshold schedule | Cosine warmup | theta starts at threshold_init, warms up over first 50% of epochs |
| L1 schedule | Linear warmup over 5 epochs | Prevents L1 penalty from overwhelming CE loss initially |

### Optimizer parameter groups

```python
optim_groups = [
    {"params": conv_weights, "weight_decay": 0.0},       # L1 applied via l1_weight_penalty()
    {"params": threshold_params, "lr": cfg.lr_threshold, "weight_decay": 0.0},
    {"params": bias_and_norm_params, "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr)
```

The L1 penalty is applied as an **explicit loss term** (`l1_weight_penalty()`),
not through the optimizer's `weight_decay` parameter (which implements L2).
This avoids the non-smooth L1/L2 combined objective.

### Training loop structure

```python
model = LassoConvNet(cfg)
optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr)

for epoch in range(num_epochs):
    for x, y in train_loader:
        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)
        total_loss = lasso_total_loss(ce_loss, model, cfg)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.clamp_thresholds(min_val=cfg.theta_min)
        optimizer.zero_grad()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        sparsity = model.get_sparsity_ratio()
        zero_filters = model.count_zero_filters()
```

### CV-specific training considerations

| Concern | Recommendation |
|---|---|
| Augmentation | Standard CIFAR-10: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize |
| Data splits | 45k train / 5k validation (standard CIFAR-10 split) |
| No class imbalance | CIFAR-10 is balanced; no special sampling needed |
| Learning rate schedule | Cosine decay from peak LR to 0 over full training budget |
| Threshold warmup | Start theta at threshold_init (close to 0), ramp via cosine over first 50% of epochs |
| EMA teacher weights | Optional but recommended for stable threshold evolution |

## Expected behavior

> `TODO: unverified` — No reference training run exists. The following is the
> expected behavior based on the design document.

- **Loss curve**: Cross-entropy should decrease monotonically from ~2.3 (random
  init for 10 classes) toward ~0.3-0.5 (depending on architecture size). The
  L1 and group-lasso penalties should add ~0.01-0.1 to the total loss.
- **Activation sparsity**: Should increase from near 0 (init, theta very small)
  toward >20% as thresholds adapt during training.
- **Threshold evolution**: Per-channel theta values should converge to distinct
  values, with higher thresholds on less informative channels.
- **Gradient norms**: Should not vanish; if most channels show zero-gradient,
  switch to adaptive threshold mode (`proximal_type="adaptive"`).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN in first steps | bf16 precision in soft-threshold | Use float32 for training (`dtype="float32"`) |
| Loss stops decreasing early | Thresholds too high, killing gradient flow | Switch to `proximal_type="adaptive"` (ablation A4) |
| All thresholds converge to ~0 | L1 penalty too weak; CE dominates | Increase `l1_weight_decay` to 1e-4 |
| Thresholds oscillate in late training | LR too high for theta params | Reduce `lr_threshold` to 1e-5 or use cosine decay |
| Activation sparsity < 5% at convergence | Thresholds too small; thresholds need warmup | Increase `threshold_init` to 0.05 or use cosine warmup schedule |
| Accuracy drop > 3% vs loss_only baseline | Architectural prox too aggressive | Set `lasso_mode="loss_only"` (ablation A1) to diagnose |
| Group lasso kills too many filters | `group_lasso_strength` too high | Reduce from 1e-4 to 1e-5 or disable (ablation A3) |
| Feature collapse (all channels same) | Over-smoothing from many stages | Reduce depth or add skip connections (not yet implemented) |
| Per-channel sparsity all identical | `threshold_learnable=False` (fixed global theta) | Set `threshold_learnable=True` |
| Training very slow | LISTA mode sequential bottleneck | Use `lasso_mode="proximal"` instead of `"lista_unrolled"` |
