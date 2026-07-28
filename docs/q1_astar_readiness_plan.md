# Q1 / Rank-A Readiness Plan

This document lists the next scientific steps after the locked final rerun and
repository cleanup.

## Current Strengths

- Locked 624-image sealed-test protocol with deterministic split and audit files.
- Paired C0/P comparisons for DenseNet121 and ResNet50 across three seeds.
- Full sealed-test prediction CSVs retained for statistical validation.
- Bootstrap confidence intervals and McNemar tests generated from locked outputs.
- XAI evaluation uses a deterministic balanced 128-case sample with lung-mask
  localization metrics.
- Manuscript now states required caveats: no external cohort, no expert lesion
  annotations, no reader study, and no isolated CBAM-only/mask-loss-only ablation.

## Highest-Priority Additions

1. External validation cohort

   Add at least one independent public pediatric/adult CXR cohort with a locked
   preprocessing and evaluation protocol. Report calibration, AUROC/AUPRC,
   sensitivity, specificity, balanced accuracy, F1, Brier, and ECE.

2. Component ablation

   Separate the proposed method into:

   - control baseline
   - CBAM only
   - mask loss only
   - CBAM plus mask loss

   Run matched seeds and report paired deltas. This is the cleanest way to support
   causal claims about the proposed method components.

3. Clinical/XAI validity

   Add expert-reviewed localization or lesion-level annotation if available. If
   expert labels are unavailable, keep the current claim as lung-region attribution
   alignment, not lesion localization.

4. Robustness and sensitivity

   Report sensitivity to:

   - threshold choice
   - mask quality or mask perturbation
   - image resolution/preprocessing
   - class imbalance
   - seed variance

5. Reproducibility package

   Provide exact environment versions, dataset slugs, SHA-256 manifests, notebook
   execution order, and a minimal script that regenerates paper tables and figures
   from locked CSV/JSON outputs.

## Reviewer-Risk Items To Keep Explicit

- The original reference paper did not fully disclose seeds and split construction,
  so exact reproduction cannot be guaranteed.
- The proposed P model combines CBAM and mask loss; without component ablation,
  the paper should avoid claiming which component independently caused the gains.
- Lung masks support attribution localization inside lung fields, not direct
  clinical lesion localization.
- Internal sealed-test performance is not deployment evidence without external
  validation.

## Recommended Next Execution Order

1. Finish GitHub cleanup and initialize/repair Git repository state.
2. Add a script/notebook to regenerate all paper tables and figures from
   `results/locked_final/`.
3. Run component ablation if compute time is available.
4. Run external validation.
5. Update manuscript and final audit after each new evidence batch.
