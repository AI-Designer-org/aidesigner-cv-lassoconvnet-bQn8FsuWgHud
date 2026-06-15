# LassoConvNet — Claim Grounding

Every architectural or performance claim should point to specific source files, test cases, benchmark commands, ablation results, or profiler output. Claims without grounding are marked `TODO: unverified`.

---

## Architectural Claims

| # | Claim | Source / Evidence | Status |
|---|---|---|---|
| C1 | "Lasso regression is integrated into the CNN forward pass via soft-thresholding" | `coder/blocks.py LassoProxConvBlock.forward()` line 147: `h = self.prox_op(h, self.theta)` | ✅ Grounded |
| C2 | "Soft-thresholding is the proximal operator for the L1 norm" | `coder/layers.py SoftThreshold.forward()` line 71: documented docstring with mathematical formula | ✅ Grounded |
| C3 | "Three operating modes: loss_only, proximal, lista_unrolled" | `coder/config.py LassoConvConfig.lasso_mode` line 27; `coder/model.py LassoConvNet` docstring lines 87-90 | ✅ Grounded |
| C4 | "Per-channel learnable thresholds of shape (1, C, 1, 1)" | `coder/blocks.py` lines 93-95: `self.theta = nn.Parameter(torch.full((1, out_channels, 1, 1), init_val))` | ✅ Grounded |
| C5 | "Pre-norm normalization before thresholding stabilizes the scale" | `coder/blocks.py` line 142: `if self.config.norm_before_prox: h = self.norm(h)` | ✅ Grounded |
| C6 | "Residual shortcut bypasses the threshold for gradient flow" | `coder/blocks.py` line 156: `return identity + h` | ✅ Grounded |
| C7 | "LISTA unrolling connects feedforward computation to ISTA optimization" | `coder/blocks.py ListaEncoder` docstring lines 254-265 | ✅ Grounded |
| C8 | "Group-lasso on filters induces structured sparsity" | `coder/loss.py group_lasso_penalty()` lines 73-130, groups filters into `group_size` chunks | ✅ Grounded |
| C9 | "bf16/fp16 safe forward pass" | `coder/layers.py SoftThreshold.forward()` lines 79-85: casts to float32 for thresholding | ✅ Grounded |
| C10 | "Gradient checkpointing support" | `coder/blocks.py` line 128: `checkpoint(self._forward, x, use_reentrant=False)` | ✅ Grounded |
| C11 | "Adaptive soft-threshold has non-zero gradient in the dead zone" | `coder/layers.py AdaptiveSoftThreshold.forward()` uses sigmoid gate; `test_model.py TestProximalOperators::test_adaptive_threshold_smoothness` | ✅ Grounded (tested) |

---

## Performance / Sparsity Claims

| # | Claim | Expected Outcome | Evidence | Status |
|---|---|---|---|---|
| P1 | "Proximal mode produces sparser feature maps than loss-only at equal accuracy" | Accuracy: ↓0–1%; Sparsity: ↑20–40% | Ablation A1 in `ablate.py`; no training results | 🔴 TODO: unverified |
| P2 | "Per-channel adaptive thresholds improve accuracy-sparsity tradeoff" | Accuracy: ↑0.5–1%; Dead channels: ↓ | Ablation A4 in `ablate.py`; no training results | 🔴 TODO: unverified |
| P3 | "Group-lasso achieves >30% filter sparsity without >1% accuracy drop" | Filter sparsity: >30%; Accuracy: ↓0–1% | Ablation A3 in `ablate.py`; `model.py count_zero_filters()`; no training results | 🔴 TODO: unverified |
| P4 | "Fixed thresholds (not learnable) barely hurt accuracy" | Accuracy: ↓0.5–1.5%; Sparsity: ↓5–10% | Ablation A2 in `ablate.py`; no training results | 🔴 TODO: unverified |
| P5 | "LISTA tied weights are sufficient (match untied at fewer params)" | Accuracy: ↑1–2% with untied; Params: ↑2× | Ablation A5 in `ablate.py`; no training results | 🔴 TODO: unverified |
| P6 | "Norm before prox can be removed without accuracy loss" | Accuracy: ↓1–2% without norm; Loss variance: ↑ | Ablation A6 in `ablate.py`; no training results | 🔴 TODO: unverified |
| P7 | "Sparsity rate >20% at convergence" | Activation sparsity: >20% | `model.py get_sparsity_ratio()`; no training results | 🔴 TODO: unverified |
| P8 | "Accuracy drop <3% vs matched-width baseline" | Accuracy: within 3% of loss_only baseline | No baseline comparison implemented | 🔴 TODO: unverified |

---

## Correctness Claims

| # | Claim | Test / Evidence | Status |
|---|---|---|---|
| T1 | "Model produces correct output shapes for all three modes" | `test_model.py TestShapes::test_output_shape_proximal`, `test_output_shape_loss_only`, `test_output_shape_lista` | ✅ Verified |
| T2 | "All parameters receive gradients" | `test_model.py TestGradients::test_all_params_receive_gradients` | ✅ Verified |
| T3 | "Thresholds (theta) are learnable and receive gradients" | `test_model.py TestGradients::test_theta_params_receive_gradients` | ✅ Verified |
| T4 | "Soft-threshold with θ=0 reduces to identity" | `test_model.py TestProximalOperators::test_soft_threshold_identity_at_zero` | ✅ Verified |
| T5 | "Soft-threshold with θ >> |x| produces all zeros" | `test_model.py TestProximalOperators::test_soft_threshold_vanishes_at_large_theta` | ✅ Verified |
| T6 | "bf16 forward pass produces no NaN/Inf" | `test_model.py TestNumerics::test_bf16_forward`, `test_bf16_proximal_operator` | ✅ Verified |
| T7 | "Total loss = CE + L1 + group lasso penalties" | `test_model.py TestLossFunctions::test_composite_loss_adds_penalties` | ✅ Verified |
| T8 | "Threshold clamping works correctly" | `test_model.py TestNumerics::test_threshold_clamping` | ✅ Verified |
| T9 | "Gradient checkpointing matches uncheckpointed forward" | `test_model.py TestCheckpointing::test_checkpoint_consistency` | ✅ Verified |
| T10 | "Feature maps have non-zero spatial variance" | `test_model.py TestCVBenchmarks::test_masked_reconstruction_proxy` | ✅ Verified |
| T11 | "Adaptive threshold has finite second derivative" | `test_model.py TestProximalOperators::test_adaptive_threshold_smoothness` | ✅ Verified |

---

## Profiling / Resource Claims

| # | Claim | Command / Evidence | Status |
|---|---|---|---|
| R1 | "Parameter count for CIFAR-scale model" | `python profile.py --mode forward` or `python ablate.py --dry-run`; see `ablate.py` dry-run output | ✅ Runnable |
| R2 | "Inference FLOPs estimate" | `python profile.py --mode forward` reports ~2× params FLOPs | ✅ Runnable |
| R3 | "Training FLOPs estimate" | `python profile.py --mode train` reports ~6× params FLOPs | ✅ Runnable |
| R4 | "Per-operator memory usage" | `python profile.py --mode memory` | ✅ Runnable |
| R5 | "All ablations produce valid models" | `python ablate.py --dry-run` instantiates and measures all configs | ✅ Runnable |

---

## How to Run Validation Commands

```bash
# Layer 1: Unit tests
cd /path/to/validator
pytest test_model.py -v                    # all tests
pytest test_model.py -v -k "shape"         # shape tests only
pytest test_model.py -v -k "gradient"      # gradient tests
pytest test_model.py -v -k "numeric"       # numerical stability

# Layer 2: Domain benchmarks (CV-specific)
pytest test_model.py -v -k "CV"            # CV correctness + benchmarks

# Layer 3: Ablations (dry-run)
python ablate.py --dry-run --output results.json

# Layer 4: Profiling
python profile.py --mode forward --steps 10
python profile.py --mode train --steps 5
python profile.py --mode memory
python profile.py --mode compare           # all three lasso modes

# Layer 5: Research quality evaluation
# (This file and rubric.md / scorecard.json are the evaluation)
```

---

## Claims Referencing External Sources

| Claim | Reference | Status |
|---|---|---|
| "Adaptive soft-threshold inspired by Gregor & LeCun 2010" | Mentioned in `coder/layers.py` docstring | ✅ Cited |
| "Standard CNN baseline should be ResNet-18 or VGG-11" | Mentioned in `architect/architecture_design.md` | 🔴 Not implemented |
| "FLOPs multiplier: 2× for inference, 6× for training" | Mentioned in `validator/profile.py` docstring (Kaplan et al.) | ✅ Cited |

---

## Unverifiable Claims

The following claims cannot be verified with the current artifacts and require experimental results:

1. "LassoConvNet matches ResNet-18 accuracy with fewer parameters" — No external baseline
2. "Group-lasso achieves structured sparsity for model compression" — Post-training pruning not implemented
3. "Proximal mode converges faster than loss-only" — No training curves
4. "Thresholds converge to distinct per-channel values during training" — No threshold tracking during training
5. "LISTA mode has sequential scan bottleneck" — No throughput comparison between proximal and LISTA

**Recorded as**: `TODO: unverified` in `scorecard.json` `claim_status_summary`
