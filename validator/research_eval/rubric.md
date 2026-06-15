# LassoConvNet — Research Quality Rubric

## Scoring Scale

| Score | Meaning |
|---|---|
| 0 | Not addressed or no artifact exists |
| 1 | Mentioned but unsupported |
| 2 | Partially supported with major gaps |
| 3 | Plausible and minimally supported |
| 4 | Strong, with clear evidence and reproducible checks |
| 5 | Publication-ready for this scaffold's scope |

---

## Dimension 1: Novelty (score: 2/5)

**What is being evaluated.** Whether the method is genuinely new relative to existing work, and whether the validation demonstrates that novelty is meaningful (not just a trivial re-labeling of an existing technique).

**Evidence we look for:**
- Clear statement of what is novel: "We replace ReLU with soft-thresholding, the proximal operator for L1, making activations architecturally sparse."
- Distinction from related approaches (e.g., L1-on-loss regularization, post-hoc pruning, LISTA)
- Baseline comparison that would falsify the novelty claim

**Current state:**
- The architect's design doc clearly articulates the novelty: integrating the Lasso proximal operator as a architectural layer (soft-thresholding) rather than only a loss penalty.
- Three operating modes (loss_only, proximal, lista_unrolled) enable direct ablation of the novelty claim.
- However, the upstream research stage (`research/`) produced only clarification questions — no literature survey, no related-work comparison.
- No evidence that the approach outperforms or differs meaningfully from vanilla L1 regularization in practice.
- The "adaptive" variant (sigmoid-gated soft-threshold) is well-known in the sparse coding literature (Gregor & LeCun 2010).

**TODO before publication:**
1. Literature survey comparing against: L1-regularized CNNs, LISTA variants, proximal operators in deep learning, Sparse CNN pruning methods
2. Training experiments showing the accuracy–sparsity Pareto frontier is strictly better than loss-only L1

---

## Dimension 2: Experimental Comprehensiveness (score: 2/5)

**What is being evaluated.** Whether the experiments would, if run, distinguish the proposed architecture from baselines and identify failure modes.

**Evidence we look for:**
- Unit tests covering all components
- Domain-specific correctness tests
- Baseline comparison (loss_only mode as trivial baseline)
- Ablation experiments isolating each design decision
- Profiling (memory, throughput, FLOPs)

**Current state:**
- Comprehensive unit tests exist (shapes, gradients, numerics, sparsity, loss functions)
- Domain-specific CV benchmarks are implemented (translation robustness, noise entropy, sparsity profile, linear probe proxy)
- All 6 proposed ablations (A1–A6) are defined as single-field config changes in the ablation runner
- Profiling script supports forward/train/memory modes
- **BLOCKING GAP: No training loop or training experiment exists.** The ablations can only be run in dry-run mode (parameter count + initial sparsity). The core claim ("proximal mode produces sparser activations at equal accuracy") cannot be tested without training.
- **BLOCKING GAP: No baseline model comparison.** The "loss_only" mode is the proposed minimal baseline, but there is no training harness to compare accuracy.
- No data-loading, no CIFAR-10/100 pipeline, no experiment tracking.

**Required experiments:**
1. Train proximal vs. loss_only on CIFAR-10 — report accuracy and sparsity at convergence
2. Train with all 6 ablations — rank by accuracy–sparsity Pareto efficiency
3. Compare against a standard ResNet-18 with L2-only decay (external baseline)
4. Measure inference throughput (images/sec) for proximal vs. loss_only
5. Zero-filter removal: measure compressed model accuracy after pruning

---

## Dimension 3: Theoretical Foundation (score: 3/5)

**What is being evaluated.** Whether claims about the method's behavior are grounded in theory, optimization, or known convergence properties.

**Evidence we look for:**
- Optimization objective stated clearly
- Gradient flow analysis
- Convergence arguments for unrolled iterations
- Inductive bias statements

**Current state:**
- The total loss function is clearly stated: CE + L1_weight + group_lasso (+ activation L1 in loss_only mode)
- Gradient flow analysis for soft-thresholding is documented: `∂L/∂h = ∂L/∂z · 1_{|h|>θ}`
- The LISTA encoder's connection to ISTA optimization is rigorously documented
- Four implementation risks are identified with mitigations
- Inductive bias statements are provided for each design choice
- **Gap: No proof that the forward pass solves a well-posed optimization problem** (except LISTA mode)
- **Gap: No analysis of the stationary points of the proximal-mode training objective** (the soft-threshold creates a non-smoothness not present in standard CNNs)
- **Gap: The adaptive soft-threshold's connection to L1 regularization is heuristic** — it is not the proximal operator of any convex regularizer

---

## Dimension 4: Result Analysis (score: 1/5)

**What is being evaluated.** Whether the results (if any exist) are correctly interpreted and whether failure modes are identified.

**Evidence we look for:**
- Numerical results (accuracy, sparsity, throughput)
- Error analysis (where does the model fail?)
- Ablation results with interpretation
- Comparison to expected behavior

**Current state:**
- **No training results exist.** All validation is at initialization.
- The smoke test verifies shapes, gradients, and basic numerics but does not measure task performance.
- Sparsity measurements at init are reported but not meaningful without training.
- No validation loss curve, no accuracy metric, no convergence analysis.

**Critical gap:** The entire "result analysis" dimension cannot be scored until training experiments are performed.

---

## Dimension 5: Implementation Reproducibility (score: 4/5)

**What is being evaluated.** Whether someone else could re-implement the method, run the experiments, and verify the claims using the provided artifacts.

**Evidence we look for:**
- Complete, runnable code
- Clear configuration interface
- Tests that validate correctness
- Documentation of hyperparameters

**Current state:**
- Complete implementation with clear separation of concerns (config, layers, blocks, backbone, model, loss)
- Well-documented docstrings for every class and method
- 30+ pytest tests covering shapes, gradients, numerics, sparsity, loss functions, checkpointing
- All three operating modes are implemented and tested
- Configuration through a single dataclass (LassoConvConfig) with sensible defaults
- bf16/fp16 safe with automatic dtype casting in proximal operators
- Gradient checkpointing support with consistency verification
- **Minor gap: No `setup.py` / `requirements.txt`** (only `import` based on `sys.path.insert`)
- **Minor gap: No training script** (the model can be instantiated but not trained)

---

## Dimension 6: Writing Readiness (score: 1/5)

**What is being evaluated.** Whether the research artifact is ready to be described in a paper, technical report, or blog post.

**Evidence we look for:**
- Clear description of the method
- Motivation and inductive bias
- Diagrams or architecture descriptions
- Experimental results or a plan for them

**Current state:**
- The architect's `architecture_design.md` is an excellent research write-up with:
  - ASCII architecture diagrams for both modes
  - Inductive bias table
  - Traceability table linking claims to ablations
  - Implementation risk analysis
- The code itself is well-documented and ready for reading
- **Critical gap: No experimental results to report** — a paper without numbers is not credible
- **Gap: No related-work section** (upstream research stage was incomplete)
- **Gap: No ablation results table** (no training experiments run yet)

---

## CV-Specific Research Questions

| Question | Assessment |
|---|---|
| Does the benchmark suite test invariance/equivariance claims? | Partially — translation robustness test exists but is weak (at init). No rotation/scale equivariance test. |
| Does it test resolution behavior? | Yes — variable spatial size test exists. |
| Does it test feature quality? | Partially — linear probe proxy checks variance, not actual classification accuracy on frozen features. |
| Does it compare against a simple CNN/ViT baseline? | No external baseline implemented. The "loss_only" mode is the internal baseline only. |
| Does it test compressed model accuracy after pruning? | No — zero-filter counting exists but no export/pruning/retraining pipeline. |
| Does it test training convergence with all three lasso modes? | No — no training loop implemented. |

---

## Summary

| Dimension | Score | Key Gap |
|---|---|---|
| Novelty | 2/5 | No literature survey; novelty claim unverified |
| Experimental Comprehensiveness | 2/5 | No training experiments; no baseline comparison |
| Theoretical Foundation | 3/5 | Well-documented loss and gradient flow; no stationary-point analysis |
| Result Analysis | 1/5 | No results at all — only initialization measurements |
| Implementation Reproducibility | 4/5 | Complete runnable code; minor packaging gaps |
| Writing Readiness | 1/5 | Excellent design doc but no experimental results |
| **Overall** | **2.2/5** | **Pre-experimental: architecture validated, experimental validation still needed** |
