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
- Phase 5: Temperature Scaling, deep ensemble and MC Dropout uncertainty
  scoring, referral threshold selection, and test-set evaluation.
- Phase 6: reliability diagrams, risk-coverage curves, tables, final report
  figures, and reproducibility cleanup.
