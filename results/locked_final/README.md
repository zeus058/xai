# Locked Final Rerun Artifacts

This directory is the canonical evidence bundle for the manuscript.

The retained artifact checksum manifest is:

- `retained_artifact_manifest.csv`

## Reproduction

Source kernel:

- `hintrngia/kagglea01`

Purpose:

- Reproduce the reference-paper-style DenseNet121/ResNet50 baseline under the
  available public data constraints.
- Document that the original paper did not fully disclose random seeds and split
  construction, so exact reproduction is not guaranteed.

## Classification Battery

Source kernels:

- `hintrngia/kaggleb01` through `hintrngia/kaggleb12`

Conditions:

- `D-C0`: DenseNet121 control
- `D-P`: DenseNet121 proposed
- `R-C0`: ResNet50 control
- `R-P`: ResNet50 proposed

Seeds:

- `3407`
- `42`
- `2024`

Each run retains configs, environment metadata, dataset audits, training history,
thresholds, calibration outputs, sealed-test predictions, confusion matrix, and
test metrics. Checkpoint `.pt` files are excluded from the cleaned GitHub package.

## Statistical Evidence

Source kernel:

- `KAGGLE_D01_statistical_validation.ipynb`

Canonical paper tables:

- `classification_statistics/classification_summary_table.csv`
- `classification_statistics/paired_delta_bootstrap_ci.csv`
- `classification_statistics/classification_delta_table.csv`
- `classification_statistics/classification_statistics_summary.md`

## XAI Manifest

Source kernel:

- `hintrngia/kagglec00`

Canonical manifest:

- `xai_manifest/xai_sample_manifest_128.csv`

Validation:

- 128 rows
- 64 normal and 64 pneumonia cases
- SHA-256:
  `ab1f02d26d2cc7f0db86d25f8f05bf87e9069be48c4a33ea04e7a580911ebbef`
- 624/624 sealed-test cases had both image and mask available

## XAI Battery

Source kernels:

- `hintrngia/xaie01`
- `hintrngia/xaie02`
- `hintrngia/xaie03`
- `hintrngia/xaie04`

Aggregate evidence:

- `xai_runs/aggregate/xai_summary_table.csv`
- `xai_runs/aggregate/xai_p_minus_c0_paired_delta_ci.csv`

Per-case rendered PNG panels are excluded from the cleaned GitHub package. The
manuscript figure is kept under
`paper/figures/xai_qualitative_densenet_guided_cases.png`.
