# B01-B12 Classification Validation Analysis

## Scope

This note validates the locked classification battery for the four final settings:

- D-C0: DenseNet121 baseline
- D-P: DenseNet121 proposed CBAM + mask-loss model
- R-C0: ResNet50 baseline
- R-P: ResNet50 proposed CBAM + mask-loss model

Each setting was run with three seeds: 3407, 42, and 2024. All runs use the locked 624-case test set.

## Artifact Integrity

All 12 runs completed successfully. Each run contains the expected paper-facing artifacts:

- `run_config.json`
- `dataset_audit.json`
- `test_predictions.csv`
- `test_metrics.json`
- `confusion_matrix.json`
- `threshold.json`
- `calibration.json`
- `train_history.csv`
- `best_checkpoint.pt`
- `environment.json`
- `run_status.json`

For every run, `test_predictions.csv` contains exactly 624 rows and 624 unique image IDs. The test label distribution is stable across all runs:

- Normal: 234
- Pneumonia: 390

The C0/P comparisons are image-aligned for every architecture and seed, meaning that each paired comparison is performed on the same 624 test images.

## Metric Consistency

The metrics recomputed from `test_predictions.csv` match the stored `test_metrics.json` values. No evidence was found that the reported classification metrics were copied from mismatched runs or computed on a different test set.

## Main Results

### DenseNet121

The proposed D-P model improves the three-seed mean over D-C0:

- AUROC: +0.006898
- AUPRC: +0.005724
- Accuracy: +0.010150
- Balanced accuracy: +0.012963
- Specificity: +0.024216
- F1: +0.006638
- Brier score: -0.012786
- ECE: -0.015253

Accuracy, balanced accuracy, NPV, F1, Brier score, and ECE improve in all three seeds. AUROC and AUPRC improve in two of three seeds.

### ResNet50

The proposed R-P model improves the three-seed mean over R-C0:

- AUROC: +0.003168
- AUPRC: +0.001386
- Accuracy: +0.012821
- Balanced accuracy: +0.016524
- Specificity: +0.031339
- F1: +0.008224
- Brier score: -0.012669
- ECE: -0.014655

AUROC, NPV, Brier score, and ECE improve in all three seeds. Accuracy, balanced accuracy, specificity, precision, F1, and AUPRC improve in two of three seeds.

## Paired Correctness Tests

Exact McNemar tests were computed per architecture and seed using paired image-level correctness:

| Architecture | Seed | C0-only correct | P-only correct | McNemar p-value |
|---|---:|---:|---:|---:|
| DenseNet121 | 3407 | 10 | 11 | 1.0000 |
| DenseNet121 | 42 | 6 | 21 | 0.0059 |
| DenseNet121 | 2024 | 8 | 11 | 0.6476 |
| ResNet50 | 3407 | 15 | 14 | 1.0000 |
| ResNet50 | 42 | 13 | 21 | 0.2295 |
| ResNet50 | 2024 | 4 | 21 | 0.0009 |

These results show that the proposed model is not uniformly significant at the single-seed level. The improvements are directionally positive, but the paper should avoid claiming large or universal significance unless supported by aggregated bootstrap confidence intervals and paired statistical tests.

## Interpretation

The classification results are plausible and internally consistent. The proposed method produces modest but stable gains, especially in specificity, balanced accuracy, Brier score, and ECE. This aligns with the intended research direction: lung-region guidance and attention regularization appear to reduce false positives and improve calibration more clearly than they improve already-saturated sensitivity.

The sensitivity values are very high for all models, approximately 0.994 to 0.997. Because sensitivity is already saturated, the meaningful improvements are mostly in specificity, calibration, and balanced accuracy. The paper should frame the contribution accordingly.

The calibration gains are useful but should be described carefully. ECE improves, but the absolute ECE remains around 0.16 to 0.18, so the correct claim is improved calibration, not well-calibrated prediction.

## Warnings for Paper Use

1. The B01-B12 classification battery is suitable as the main locked classification evidence after adding confidence intervals and aggregate paired tests.
2. The effect size is modest. The paper should not claim a dramatic performance jump.
3. Single-seed significance is mixed. Use three-seed mean plus paired bootstrap confidence intervals.
4. Reproduction of the original paper did not satisfy the pre-defined +/-2% criterion, so comparison to the original paper must be framed as best-effort reproduction under a cleaned protocol.
5. Claims about reduced shortcut learning or better localization are not fully supported by B01-B12 alone. They require the locked Guided Grad-CAM/XAI validity battery.
6. Because the original paper did not disclose full seeds and split assignments, exact reproduction is not possible. This must be stated explicitly in the manuscript.

## Recommendation

The B01-B12 results are reasonable, technically valid, and aligned with the research direction. They are not yet sufficient as final Q1/A-level evidence by themselves. Before writing the final results section, complete:

- bootstrap confidence intervals for all main metrics,
- aggregate paired tests over the locked 624-image test set,
- final Guided Grad-CAM validity analysis on the locked 128-case XAI sample,
- paper tables that report mean +/- SD and confidence intervals rather than single-run numbers.

