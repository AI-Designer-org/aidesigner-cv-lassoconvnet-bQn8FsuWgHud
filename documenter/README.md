# LassoConvNet

A CNN architecture for image classification where Lasso (L1) regression is integrated directly into the forward pass via soft-thresholding proximal operators, producing architecturally sparse feature maps.

Standard CNNs rely on ReLU activations and external L1 loss penalties to encourage sparsity. LassoConvNet replaces ReLU with the L1 proximal operator (soft-thresholding), making activation sparsity an architectural property rather than merely a regularization target. Per-channel learnable thresholds allow each feature map to discover its own sparsity level during training. An optional LISTA-unrolled mode interprets the full feedforward computation as solving an iterative sparse coding objective.

**Status:** Pre-experimental. The architecture is implemented and validated for shapes, gradients, and numerical stability (35+ unit tests pass). No training experiments have been run yet — see [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the current gap status.

## Highlights

- **Architectural Lasso integration** — Soft-thresholding (the L1 proximal operator) replaces ReLU in every convolutional block, making sparsity a forward-pass property rather than just a loss penalty. See [ARCHITECTURE.md#3-the-core-component](docs/ARCHITECTURE.md#3-the-core-component).
- **Three operating modes** — `proximal` (soft-thresholding in every block), `loss_only` (standard CNN + L1 loss, the baseline), and `lista_unrolled` (unrolled ISTA iterations for sparse coding). See [ARCHITECTURE.md#4-tensor-shape-evolution](docs/ARCHITECTURE.md#4-tensor-shape-evolution).
- **Per-channel learnable thresholds** — Each feature map learns its own sparsity threshold (shape `(1, C, 1, 1)`), letting some channels become highly sparse while others remain dense. See [ARCHITECTURE.md#5-design-decisions](docs/ARCHITECTURE.md#5-design-decisions).
- **Group-lasso structured sparsity** — A penalty on filter-group Frobenius norms encourages entire filters to be pruned, enabling model compression without specialized hardware. See [ARCHITECTURE.md#6-domain-specific-considerations](docs/ARCHITECTURE.md#6-domain-specific-considerations).

## Quick start

```bash
pip install -r requirements.txt
python coder/smoke_test.py        # smoke test, prints param count + output shape
pytest validator/test_model.py -v  # full unit-test suite (35+ tests)
```

## Repository layout

```
coder/                       # Implementation
├── config.py                # LassoConvConfig dataclass (all hyperparameters)
├── layers.py                # Proximal operators (Soft/Adaptive/HardThreshold), sparsity utils
├── blocks.py                # LassoProxConvBlock, ListaConvLayer, ListaEncoder
├── backbone.py              # 4-stage pyramid backbone
├── model.py                 # LassoConvNet full model + classifier head
├── loss.py                  # Composite loss (CE + L1 weights + group lasso)
└── smoke_test.py            # 14-test smoke suite (shapes, gradients, bf16, thresholds)

validator/                   # Validation, benchmarks, profiling
├── test_model.py            # 35+ pytest tests (shapes, gradients, numerics, sparsity, CV)
├── ablate.py                # Ablation runner (6 config ablations A1–A6)
├── model_profile.py         # torch.profiler (forward/train/memory modes)
├── research_eval/
│   ├── scorecard.json       # 6-dimension research quality scores
│   ├── claim_grounding.md   # Traceability of all claims to source/test/artifact
│   ├── experiment_coverage.md # Required vs implemented experiments
│   └── rubric.md            # Scoring rubric and gap analysis

docs/                        # Documentation
├── ARCHITECTURE.md          # Design, inductive biases, equations, decisions
├── TRAINING.md              # Training recipe, environment, troubleshooting
├── BENCHMARKS.md            # Results, ablations, profiling, research evaluation
└── API.md                   # Full API reference
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design and inductive biases
- [docs/TRAINING.md](docs/TRAINING.md) — how to train, recipe, troubleshooting
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — results, ablations, profiling
- [docs/API.md](docs/API.md) — module-level API reference

## Citation

```bibtex
@misc{lassoconvnet,
  title  = {LassoConvNet: A CNN Architecture with Integrated Lasso Regression},
  author = {<TODO>},
  year   = {2026},
  note   = {Generated via ml-designer pipeline; pre-experimental}
}
```
