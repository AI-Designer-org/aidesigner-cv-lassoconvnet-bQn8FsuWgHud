# Benchmarks

All numbers are reproducible with the commands shown.
Numbers marked `TODO` have not been measured — do not cite them.

> **Status: Pre-experimental.** No training experiments have been run. All
> accuracy and sparsity measurements are `TODO: unverified` until a training
> harness and CIFAR-10 pipeline are implemented. The architecture's correctness
> is validated by 35+ unit tests (shapes, gradients, numerics, sparsity, CV
> properties).

## Parameter count

Default CIFAR-10 config (`stage_channels=(64, 128, 256, 512)`, `stage_depths=(2, 2, 2, 2)`):

| Mode | Params | Command |
|---|---|---|
| Proximal | 5,032,138 | `python coder/smoke_test.py` (printed at end) |
| Loss-only | ~5,032,120 | Same config, theta not counted |
| LISTA-unrolled (tied, 6 iters, dict=256) | ~1,290,000 | `python validator/ablate.py --dry-run --ablation A5_lista_tied` |
| LISTA-unrolled (untied, 6 iters, dict=256) | ~2,340,000 | `python validator/ablate.py --dry-run --ablation A5_lista_untied` |

## Unit tests

```bash
cd validator
pytest test_model.py -v                          # all 35+ tests
pytest test_model.py -v -k "shape"               # shape tests
pytest test_model.py -v -k "gradient"            # gradient flow tests
pytest test_model.py -v -k "numeric"             # numerical stability tests
pytest test_model.py -v -k "CV"                  # CV domain benchmarks
```

### Correctness tests — all pass

| Test class | Tests | What it verifies |
|---|---|---|
| `TestShapes` | 6 | Output shapes for all 3 modes, variable batch/spatial sizes, operator shapes, backbone stage shapes |
| `TestGradients` | 6 | All params receive gradients, theta gradients, no NaN gradients, gradient flow through soft/adaptive threshold, loss_only mode |
| `TestSparsity` | 4 | Soft-threshold creates exact zeros in dead zone, identity at theta=0, sparsity from forward pass, channel sparsity distribution |
| `TestProximalOperators` | 5 | Identity at theta=0, all-zeros at large theta, adaptive smoothness (C^inf), hard threshold correctness, factory dispatch |
| `TestSparsityMetrics` | 3 | Sparsity ratio correctness, channel sparsity shape, all-zero channel detection |
| `TestNumerics` | 8 | bf16 forward stability, bf16 operator stability, extreme/constant/zero inputs, threshold clamping, NaN guard, composite loss |
| `TestCVProperties` | 4 | Translation robustness, noise input entropy, spatial invariance, zero-filter counting |
| `TestCVBenchmarks` | 4 | Feature variance (linear probe proxy), spatial structure preservation, depth-wise sparsity profile, cross-mode consistency |
| `TestLossFunctions` | 5 | L1 penalty non-negative/scaling, group lasso non-negative/zero-for-large-groups, composite loss = CE + penalties |
| `TestUtilities` | 4 | Parameter count, threshold dict, NaN guard toggle, theta param group |
| `TestCheckpointing` | 2 | Forward consistency, backward pass with checkpointing |

### All tests pass on CPU and CUDA (when available)

```
Results: 14 passed, 0 failed / 14 total  (smoke_test.py)
Results: 35+ passed, 0 failed            (pytest test_model.py -v)
```

## CV benchmark results

> These are **proxy benchmarks** that run in under 30 seconds at initialization.
> They do NOT replace full CIFAR-10/100 training.

| Task | Metric | Value | Command | Notes |
|---|---|---|---|---|
| Translation robustness | Agreement | > 10% (random baseline) | `pytest -k "translation"` | At init; full metric requires trained model |
| Noise input entropy | Relative entropy | > 50% of max | `pytest -k "entropy"` | Proxies for no spatial shortcut |
| Feature dead channels | % near-zero std | < 50% | `pytest -k "linear_probe_proxy"` | At init; most channels have non-zero variance |
| Spatial feature variance | Mean std | > 1e-4 | `pytest -k "masked_reconstruction"` | Spatial structure is preserved |
| Per-stage sparsity | Ratio per stage | Varies | `pytest -k "sparsity_profile"` | Deeper stages may differ; requires training |
| Cross-mode consistency | All finite | All 3 modes valid | `pytest -k "forward_consistent"` | Proximal, loss_only, lista all produce valid output |

## Ablation study

> All ablations are defined as single-field config changes in
> `validator/ablate.py`. Dry-run results (parameter count + sparsity at
> initialization) are available. **Training results are `TODO: unverified`.**

Reproduce dry-run: `python validator/ablate.py --dry-run --output results.json`

| Ablation | Config field | Baseline | Ablated | Params | Init sparsity | Notes |
|---|---|---|---|---|---|---|
| A0 (baseline) | — | — | — | 5,032,138 | ~0.00 (theta=0.01) | Proximal mode, all features enabled |
| A1 | `lasso_mode` | proximal | loss_only | ~5,032,120 | ~0.00 | ReLU instead of soft-threshold; sparsity only from L1 loss |
| A2 | `threshold_learnable` | True | False | ~5,032,010 | ~0.00 | Global fixed theta=0.01; fewer params |
| A3 | `use_group_lasso` | True | False | 5,032,138 | ~0.00 | No group-lasso penalty; same architecture |
| A4 | `proximal_type` | soft | adaptive | 5,032,138 | ~0.00 | Sigmoid-gated threshold (smooth gradients) |
| A5 (tied) | `lista_tie_weights` | — | True | ~1,290,000 | ~0.00 | LISTA with shared dictionary; fewer params |
| A5 (untied) | `lista_tie_weights` | — | False | ~2,340,000 | ~0.00 | LISTA with per-iteration dictionaries |
| A6 | `norm_before_prox` | True | False | 5,032,138 | ~0.00 | No BN before soft-threshold |

> **Note:** Sparsity at initialization is ~0 because `threshold_init=0.01` is
> small relative to initial random activations (`std ~1.0`). Sparsity becomes
> meaningful only after training when thresholds adapt.

### Hypothesis table (all `TODO: unverified`)

| Ablation | Hypothesis tested | Expected Δ accuracy | Expected Δ sparsity |
|---|---|---|---|
| A1 (proximal → loss_only) | Architectural prox produces sparser feature maps | ↓0–1% | ↓20–40% |
| A2 (learnable → fixed theta) | Per-channel thresholds improve tradeoff | ↓0.5–1.5% | ↓5–10% |
| A3 (group lasso on/off) | Group-lasso induces filter sparsity without hurting accuracy | ↓0–1% | Filter sparsity: 0 → >30% |
| A4 (soft → adaptive) | Adaptive threshold prevents gradient starvation | ↑0.5–1% | ↓ dead channels |
| A5 (tied → untied LISTA) | Untied weights increase capacity | ↑1–2% | Params: ↑2x |
| A6 (norm before prox off) | Pre-prox norm stabilizes theta | ↓1–2% | Loss variance: ↑ |

## Profiling

> `TODO: unverified` — The profiling script is implemented but has not been run
> on a GPU. The following are theoretical estimates using the standard 2x/6x
> FLOPs convention (Kaplan et al.).

Estimated for default CIFAR-10 config (~5M params):

| Phase | Est. FLOPs | Notes |
|---|---|---|
| Forward (inference) | ~10 GFLOPS | 2 × params |
| Forward + Backward (training) | ~30 GFLOPS | 6 × params |

Reproduce: `python validator/model_profile.py --mode forward --steps 10`

Profiling modes available:

```bash
python validator/model_profile.py --mode forward     # inference profiling
python validator/model_profile.py --mode train       # forward + backward
python validator/model_profile.py --mode compare     # compare all three lasso modes
python validator/model_profile.py --mode memory      # per-operator memory breakdown
```

## Research-quality evaluation

| Dimension | Score | Evidence | Gaps |
|---|---|---|---|
| Novelty | 2/5 | Soft-thresholding as architectural layer clearly articulated vs. loss-only L1 | No literature survey; no experimental evidence of benefit over vanilla L1 regularization |
| Experimental comprehensiveness | 2/5 | 35+ unit tests; all 6 ablations defined; profiling scripts | **BLOCKING**: No training loop; no baseline comparison; no experiment tracking |
| Theoretical foundation | 3/5 | Loss function, gradient flow, LISTA-ISTA connection documented | No stationary-point analysis; adaptive variant not a convex proximal operator |
| Result analysis | 1/5 | Smoke test + init measurements only | **CRITICAL**: No training results exist |
| Implementation reproducibility | 4/5 | Complete runnable code; clear config; bf16/checkpoint safety | No setup.py/requirements.txt; no training script |
| Writing readiness | 1/5 | Excellent design doc and code docstrings | No experimental results; no related-work section |
| **Overall** | **2.2/5** | | Pre-experimental: architecture validated, experimental validation needed |

### Required next experiments

1. **Train proximal vs. loss_only mode on CIFAR-10** (priority: critical)
   - Architecture: `stage_channels=(64,128,256,512)`, `stage_depths=(2,2,2,2)`
   - Budget: 200 epochs
   - Report: test accuracy + activation sparsity at convergence
   - Falsification: if accuracy drops >3% or sparsity <5%, architectural integration provides no benefit over loss-only L1

2. **Run all 6 ablations (A1–A6) on CIFAR-10** (priority: high)
   - Budget: 100 epochs per ablation
   - Rank by accuracy-sparsity Pareto efficiency

3. **Compare against standard ResNet-18** (priority: high)
   - Standard L2 decay baseline
   - Compare accuracy, parameter count, inference speed

4. **Threshold evolution analysis** (priority: medium)
   - Track per-channel theta during training
   - Measure convergence, channel specialization, oscillation

5. **Gradient starvation test** (priority: medium)
   - Compare soft vs. adaptive threshold during training
   - Measure dead neuron fraction, gradient norm, final accuracy

### Claims status summary

| Claim | Status | Evidence location |
|---|---|---|
| "Soft-thresholding integrated as architectural layer" | ✅ Grounded | `coder/blocks.py LassoProxConvBlock.forward()` |
| "Three operating modes (loss_only, proximal, lista)" | ✅ Grounded | `coder/config.py lasso_mode`, `coder/model.py` |
| "Per-channel learnable thresholds" | ✅ Grounded | `coder/blocks.py` line 93-95 |
| "Pre-norm stabilizes threshold scale" | ✅ Grounded | `coder/blocks.py` line 142 |
| "bf16 safe forward pass" | ✅ Grounded | `coder/layers.py` lines 79-85 |
| "Gradient checkpointing support" | ✅ Grounded | `coder/blocks.py` line 128 |
| "Proximal mode sparser than loss-only" | 🔴 TODO: unverified | `ablate.py` A1 defined; needs training |
| "Learnable thresholds improve tradeoff" | 🔴 TODO: unverified | `ablate.py` A2 defined; needs training |
| "Group-lasso achieves >30% filter sparsity" | 🔴 TODO: unverified | `ablate.py` A3 defined; needs training |
| "Adaptive threshold prevents gradient starvation" | 🔴 TODO: unverified | `ablate.py` A4 defined; needs training |
| "LISTA tied weights sufficient" | 🔴 TODO: unverified | `ablate.py` A5 defined; needs training |
| "Accuracy drop <3% vs matched-width baseline" | 🔴 TODO: unverified | No baseline comparison implemented |
