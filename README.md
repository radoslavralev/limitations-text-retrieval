# Semantic Cache Limitations: Identity Trade-off Experiments

This repository contains experiments evaluating the identity trade-off in dual encoder models for semantic caching.

## Overview

We investigate two key questions:
1. **Identity Trade-off**: Does training with structural negatives (Model B) improve robustness to structural perturbations at the cost of retrieval performance compared to a baseline (Model A)?
2. **Verifier Architectures**: Can lightweight verifiers (F0-F4) recover structural awareness without sacrificing retrieval quality?

## Setup

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Or with pip
pip install -e .
```

## Reproducing Results

### Experiment 1: Model A vs Model B (Identity Trade-off)

Trains two model variants across 4 encoder backbones with 3 random seeds:
- **Model A (Baseline)**: Trained on NQ triplets only
- **Model B (Structured)**: Trained on NQ + structural negatives

```bash
bash run_experiments.sh
```

Results are saved to `./results/`. View aggregated results:

```bash
uv run python show_results.py --results-dir ./results
```

### Experiment 2: Frozen vs End-to-End Verifiers

Trains verifiers (F0-F4) in frozen and end-to-end modes:

```bash
bash run_frozen_vs_e2e_experiment.sh
```

Results are saved to `./verifier_experiment_results/`. View results with plots:

```bash
uv run python show_verifier_results.py -r ./verifier_experiment_results/<experiment_dir> --plot
```

## Verifier Types

| Verifier | Description | Learnable Params |
|----------|-------------|------------------|
| None | Embedding cosine similarity (baseline) | 0 |
| F0 | Cosine similarity on token embeddings | 0 |
| F1 | Absolute difference aggregation | 0 |
| F2 | Concatenation-based | 0 |
| F3 | CNN over token embeddings | ~100K-500K |
| F4 | Transformer over token embeddings | ~1M+ |

## Data

The `data/` directory contains:
- `nq_triplets_100k.csv`: Natural Questions triplets for baseline training
- `unified_negatives_with_nq_train.csv`: NQ + structural negatives for structured training
- `unified_negatives_test.csv`: Test set with structural negative pairs

## Hardware Requirements

- GPU with 24GB+ VRAM recommended (tested on NVIDIA L4 and A10)
- Training time: ~4 minutes per model/verifier configuration
- Full experiment suite: ~2-3 hours

## Key Files

| File | Purpose |
|------|---------|
| `run_experiments.sh` | Model A/B training across backbones |
| `run_frozen_vs_e2e_experiment.sh` | Verifier training (frozen/E2E) |
| `train.py` | Core encoder training script |
| `train_and_eval_verifiers_experiment.py` | Verifier training + evaluation |
| `show_results.py` | Generate LaTeX tables for encoder results |
| `show_verifier_results.py` | Generate LaTeX tables + plots for verifiers |
| `evaluate_on_mrpc_negatives.py` | Evaluate on structural negative pairs |
| `verifiers.py` | Verifier architecture implementations |
