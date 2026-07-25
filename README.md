# Uncertainty-Aware Skin Lesion Classification with Human Referral

This repository contains a reproducible research pipeline scaffold for
seven-class HAM10000 dermoscopic skin lesion classification with calibrated
confidence, uncertainty estimation, and selective human referral.

**Research question:** How can calibrated confidence and uncertainty estimation
be used to support human referral in skin lesion classification?

## Motivation

Deep learning classifiers can produce confident predictions even when they are
wrong or presented with difficult cases. This project studies whether
post-hoc calibration and uncertainty estimates can identify predictions that
should be referred to a human reviewer instead of being automatically accepted.

This is a research prototype only, not a medical diagnostic tool. It makes no
medical deployment claim.

## Method Overview

- Train EfficientNet-B0 for seven HAM10000 classes.
- Calibrate validation logits with Temperature Scaling.
- Compare uncertainty estimates from a three-member EfficientNet-B0 Deep
  Ensemble and MC Dropout.
- Select referral thresholds on the validation split and freeze them before
  test evaluation.
- Report classification, calibration, uncertainty, and selective-referral
  metrics.

## Repository Structure

```text
configs/                 Experiment configuration
notebooks/               Kaggle experiment notebook scaffold
outputs/                 Local artifact folders tracked with .gitkeep files
src/                     Research pipeline source modules
tests/                   Lightweight local tests
requirements-local.txt   Lightweight local development dependencies
requirements.txt         Kaggle/full experiment dependencies
```

## Local Setup

Use Python 3.10+ and install only the lightweight local dependencies when
developing source code or running tests locally:

```bash
python -m pip install -r requirements-local.txt
python -m compileall src tests
pytest -q
```

Local development is intended for code quality, configuration validation, and
small tests. Do not train models locally.

## Kaggle Training Note

The full experiments are intended to run on Kaggle GPU notebooks. The full
requirements file intentionally excludes `torch` and `torchvision` because
Kaggle usually provides GPU-compatible builds.

## Dataset Note

HAM10000 images and metadata are not included in GitHub. Dataset locations must
be supplied externally in future phases; source code should not hardcode local
or Kaggle dataset paths.

Phase 2 includes reusable helpers for HAM10000 path discovery, metadata loading,
image-path attachment, grouped stratified splitting, split summaries, and class
distribution plots. These helpers accept caller-supplied paths and are suitable
for Kaggle notebooks.

Phase 3 adds the Kaggle-facing PyTorch image dataset, torchvision transforms,
DataLoader creation, EfficientNet-B0 model factory, class-weight summaries, and
batch/model preview helpers while keeping torch, torchvision, and timm imports
lazy for lightweight local testing.

Phase 4 adds reusable single-model baseline training, validation evaluation,
checkpoint/history export hooks, logits and prediction table export, core
classification metrics, training curves, and confusion matrix plotting.

Phase 5 adds Temperature Scaling utilities, calibration metrics, calibrated
probability prediction tables, and reliability diagram plotting for comparing
uncalibrated and calibrated confidence.

Phase 6 adds calibrated-baseline uncertainty scores, error-detection AUROC,
validation-selected referral thresholds, selective classification metrics, and
risk-coverage plotting.

Phase 8 adds Deep Ensemble probability aggregation, ensemble calibration from
mean probabilities, ensemble uncertainty scores, and baseline-versus-ensemble
comparison plotting.

Phase 9 adds MC Dropout stochastic inference, MC Dropout uncertainty scoring,
and uncertainty-method comparison plotting against the calibrated baseline and
Deep Ensemble.

### MC Dropout Correction

The initial Phase 9 MC Dropout run was invalid: timm's configured `drop_rate`
used functional dropout during training but created no `torch.nn.Dropout`
module, so enabling dropout modules at inference left all 20 passes identical.
The corrected EfficientNet-B0 classifier now applies an explicit Dropout module
before its final Linear operation.

The Linear parameters retain the state-dict keys `classifier.weight` and
`classifier.bias`, and Dropout has no parameters. The existing seed-42
checkpoint therefore remains strictly loadable and does not require
retraining. MC Dropout inference, calibration, uncertainty, and referral
results must be rerun because the previous stochastic passes were invalid.

## Phase Roadmap

- Phase 0: repository scaffold, configuration, path utilities,
  reproducibility helper, placeholder modules, README, and lightweight tests.
- Phase 1: Kaggle setup and HAM10000 dataset discovery.
- Phase 2: metadata loading, image-path attachment, leakage-safe grouped
  stratified splitting, split summaries, and class-distribution plots.
- Phase 3: PyTorch image dataset, transforms, DataLoaders, EfficientNet-B0
  classifier creation, class weights, and lightweight summaries.
- Phase 4: single EfficientNet-B0 baseline training, evaluation, inference
  export, classification metrics, training curves, and confusion matrices.
- Phase 5: Temperature Scaling, calibration metrics, calibrated probabilities,
  prediction tables, and reliability diagrams.
- Phase 6: calibrated-baseline uncertainty scoring, referral threshold
  selection, selective classification, and risk-coverage evaluation.
- Phase 7: Kaggle training and export of three EfficientNet-B0 ensemble members.
- Phase 8: Deep Ensemble aggregation, ensemble calibration, ensemble
  uncertainty, referral comparison, and model comparison plots.
- Phase 9: MC Dropout stochastic inference, uncertainty scoring, calibration
  reuse, referral evaluation, and method comparison plots.
- Phase 10: final tables,
  figures, and reproducibility cleanup.
