# Changelog

## [0.1.0] — 2026-06-15

### Added
- Initial implementation of LassoConvNet: a CNN architecture with integrated Lasso regression via soft-thresholding proximal operators.
- Three operating modes: `proximal` (architectural soft-thresholding), `loss_only` (standard CNN + L1 loss), and `lista_unrolled` (unrolled ISTA iterations).
- Core components:
  - `LassoProxConvBlock` — conv-bn block with per-channel learnable soft-thresholding and residual shortcut.
  - `SoftThreshold`, `AdaptiveSoftThreshold`, `HardThreshold` — proximal operator implementations with bf16/fp16 safety.
  - `ListaEncoder` / `ListaConvLayer` — unrolled ISTA iterations for convolutional sparse coding.
  - `LassoConvNetBackbone` — 4-stage ResNet-style pyramid with lasso-integrated convolutions.
  - `ClassifierHead` — global pooling + MLP classifier.
- Composite loss function: cross-entropy + L1 weight penalty + group-lasso filter penalty (+ optional activation L1 in loss_only mode).
- `LassoConvConfig` — single dataclass for all hyperparameters with sensible defaults.
- Unit test suite (35+ tests) covering shapes, gradient flow, numerical stability, sparsity correctness, CV domain properties, loss functions, and gradient checkpointing.
- Domain-specific CV benchmarks: translation robustness, noise input entropy, sparsity profile, linear probe proxy, spatial structure preservation.
- Ablation runner (`ablate.py`) with all 6 architect-proposed ablations (A1–A6) as single-field config changes, including dry-run mode.
- Profiling script (`model_profile.py`) with forward, train, memory, and cross-mode comparison modes using torch.profiler.
- Research evaluation artifacts: scorecard, claim grounding, experiment coverage, and rubric.
- Documentation: README, ARCHITECTURE, TRAINING, BENCHMARKS, API.
