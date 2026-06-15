# LassoConvNet — Experiment Coverage

## Required vs. Implemented Experiments

This document maps the experiments required by the upstream constraints (from `ml-research` and `ml-architect`) to implemented artifacts. Where an experiment is missing, we identify the gap and recommend next steps.

---

## 1. Baselines

| Baseline | Required by | Status | Location | Notes |
|---|---|---|---|---|
| Standard CNN with L2-only decay (e.g., ResNet-18) | `ml-research` (falsification target) | **MISSING** | — | No external baseline implemented |
| Loss-only L1 (same architecture, L1 in loss) | `ml-architect` (A1 ablation baseline) | **IMPLEMENTED** | `config.py`: `lasso_mode="loss_only"` | Config exists, but no training comparison |
| Equivalent-depth vanilla CNN (ReLU, no threshold) | Implicit | **MISSING** | — | The loss_only mode with l1_weight_decay=0 approximates this |

## 2. Evaluation Requirements

| Requirement | Required by | Status | Location | Notes |
|---|---|---|---|---|
| Test accuracy on CIFAR-10/100 | `ml-research` | **MISSING** | — | No training loop exists |
| Activation sparsity ratio (% of near-zero activations) | `ml-research` (falsification target: >20%) | **IMPLEMENTED (partial)** | `model.py` `get_sparsity_ratio()`, `layers.py` `compute_sparsity_ratio()` | Can measure at init, needs training to be meaningful |
| Filter sparsity (% of zero-filters) | `ml-architect` (group-lasso claim) | **IMPLEMENTED** | `model.py` `count_zero_filters()` | Works at init and after training |
| Parameter count and inference throughput | General | **IMPLEMENTED** | `profile.py`, `ablate.py` | Dry-run mode works; throughput requires GPU benchmark |

## 3. Single-Field Ablations (from ml-architect)

| # | Ablation | Config Field | Baseline | Ablated | Status | Validated? |
|---|---|---|---|---|---|---|
| A1 | Lasso mode: proximal → loss_only | `lasso_mode` | `"proximal"` | `"loss_only"` | **IMPLEMENTED** in `ablate.py` | Dry-run only (sparsity at init). Needs training. |
| A2 | Learnable → fixed thresholds | `threshold_learnable` | `True` | `False` | **IMPLEMENTED** in `ablate.py` | Dry-run only. Needs training. |
| A3 | Group lasso on/off | `use_group_lasso` | `True` | `False` | **IMPLEMENTED** in `ablate.py` | Dry-run only. Needs training. |
| A4 | Adaptive vs. vanilla soft-threshold | `proximal_type` | `"soft"` | `"adaptive"` | **IMPLEMENTED** in `ablate.py` | Dry-run only. Needs training. |
| A5 | LISTA tied vs. untied weights | `lista_tie_weights` | `True` | `False` | **IMPLEMENTED** in `ablate.py` | Dry-run only. LISTA-specific. |
| A6 | Norm before prox on/off | `norm_before_prox` | `True` | `False` | **IMPLEMENTED** in `ablate.py` | Dry-run only. Needs training. |

**Status:** All 6 ablations are defined and executable in dry-run mode. None can produce accuracy comparisons without a training harness.

## 4. Synthetic Benchmarks (Implemented)

| Benchmark | File | What it measures |
|---|---|---|
| Shape correctness (all 3 modes) | `test_model.py TestShapes` | Output shapes match expectations |
| Gradient flow (all params, theta, adaptive) | `test_model.py TestGradients` | No dead parameters, finite gradients |
| Numerical stability (bf16, extreme inputs, zero inputs) | `test_model.py TestNumerics` | No NaN/Inf under various input conditions |
| Soft-threshold correctness | `test_model.py TestSparsity` | Exact zeros in dead zone, identity at θ=0 |
| Proximal operator correctness | `test_model.py TestProximalOperators` | Soft/adaptive/hard variants match mathematical definitions |
| Sparsity metrics | `test_model.py TestSparsityMetrics` | compute_sparsity_ratio and compute_channel_sparsity return correct values |
| Loss function correctness | `test_model.py TestLossFunctions` | Penalties non-negative, composite loss = CE + penalties |
| Translation robustness | `test_model.py TestCVProperties` | Output agreement under small spatial shifts |
| Noise input entropy | `test_model.py TestCVProperties` | High entropy on random noise |
| Feature quality (linear probe proxy) | `test_model.py TestCVBenchmarks` | Feature variance and covariance condition number |
| Spatial structure preservation | `test_model.py TestCVBenchmarks` | Spatial variance across feature map positions |
| Gradient checkpointing consistency | `test_model.py TestCheckpointing` | Same output with/without checkpointing |
| Zero-filter counting | `test_model.py TestCVProperties` | Counts zero-filters correctly |

## 5. Missing Experiments

| Experiment | Gap | Impact | Effort to Fix |
|---|---|---|---|
| **Training loop** on CIFAR-10 | No training code exists | **CRITICAL**: cannot evaluate core accuracy-sparsity claim | ~200 lines (data loading + train loop + eval) |
| **Baseline comparison** (loss_only, ResNet-18) | No trained baseline measurements | **CRITICAL**: cannot distinguish from trivial baseline | ~100 lines (wrap model in train harness) |
| **Hyperparameter tuning** | No sweep over l1_weight_decay, threshold_init, group_lasso_strength | High: optimal settings unknown | ~100 lines (grid search script) |
| **Threshold evolution** over training | No tracking of theta during training | Medium: would validate learnable thresholds hypothesis | ~50 lines (hook + logging) |
| **Gradient starvation measurement** | No comparison of dead neuron fraction for soft vs adaptive | Medium: key claim about adaptive variant | ~50 lines (count dead units during training) |
| **Post-training pruning** | No export/retrain after zero-filter removal | Low: known technique, well-studied | ~100 lines (mask + fine-tune) |
| **LISTA reconstruction quality** | No measurement of D*Z vs X reconstruction MSE | Low: would validate LISTA convergence | ~30 lines (MSE computation) |

## 6. Metrics Reported

| Metric | Implemented? | At Init? | After Training? |
|---|---|---|---|
| Top-1 accuracy | Yes (`ablate.py` `measure_accuracy`) | N/A | No training implemented |
| Activation sparsity ratio | Yes (`model.py` `get_sparsity_ratio`) | Yes | No training implemented |
| Per-channel sparsity | Yes (`model.py` `get_channel_sparsity`) | Yes | No training implemented |
| Zero-filter count | Yes (`model.py` `count_zero_filters`) | Yes | No training implemented |
| Parameter count | Yes (`model.py` `count_params`) | Yes | N/A |
| Inference throughput | Yes (`profile.py`) | Yes (dry run) | No training implemented |
| Training loss | Yes (`loss.py` `lasso_total_loss`) | Yes | No training implemented |
| Threshold values | Yes (`model.py` `get_thresholds`) | Yes | No training implemented |

## 7. Can the benchmark suite distinguish from trivial baseline?

**Partially.** The test suite can verify that:
- The model produces valid output shapes and finite values
- Soft-thresholding creates exact zeros in the forward pass
- The loss function correctly aggregates CE + L1 + group lasso penalties
- Gradients flow through all parameters

The test suite **cannot** yet distinguish whether the architecture provides any benefit over a standard CNN with L1-on-loss regularization, because:
- No training accuracy comparison exists
- Sparsity at initialization is not a reliable predictor of sparsity at convergence
- The A1 ablation (proximal → loss_only) is defined but produces no accuracy comparison

## 8. Results Still `TODO: unverified`

All items from the architect's traceability table marked as `hypothesis`:

1. "Proximal mode produces sparser feature maps than loss-only at equal accuracy" — ✅ Ablation defined, ❌ no training results
2. "Per-channel learnable thresholds improve accuracy-sparsity tradeoff" — ✅ Ablation defined (A2), ❌ no training results
3. "Adaptive threshold prevents gradient starvation" — ✅ Ablation defined (A4), ❌ no training results
4. "Group-lasso induces structured filter sparsity without harming accuracy" — ✅ Ablation defined (A3), ❌ no training results
5. "LISTA unrolling provides theoretical convergence guarantees" — ✅ Implemented, ❌ no convergence measurement
6. "Untied weights improve accuracy over tied weights in LISTA" — ✅ Ablation defined (A5), ❌ no training results
