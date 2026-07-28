# B07/B10 Locked Classification Audit

Source kernels:

- `hintrngia/kaggleb07`
- `hintrngia/kaggleb10`

Runs:

- B07: `R-C0 / ResNet50 / seed 3407`
- B10: `R-P / ResNet50 / seed 3407`

## Technical Validation

Both runs completed successfully.

| Run | Status | Train | Val | Test | Test prediction rows | Unique images | Mask missing | Config check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B07 R-C0 | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=False`, `use_mask_loss=False` |
| B10 R-P | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=True`, `use_mask_loss=True` |

## Seed-3407 Result

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R-C0 | 0.971915 | 0.981173 | 0.871795 | 0.830769 | 0.994872 | 0.666667 | 0.134765 | 0.179367 | 156/78/2/388 |
| R-P | 0.976063 | 0.982116 | 0.870192 | 0.827778 | 0.997436 | 0.658120 | 0.124107 | 0.168741 | 154/80/1/389 |

## Delta: R-P minus R-C0

| Metric | Delta |
|---|---:|
| AUROC | +0.004147 |
| AUPRC | +0.000943 |
| Accuracy | -0.001603 |
| Balanced accuracy | -0.002991 |
| Sensitivity | +0.002564 |
| Specificity | -0.008547 |
| Brier | -0.010658 |
| ECE | -0.010627 |

## Interpretation

For ResNet50 seed 3407, the proposed method improves ranking metrics and calibration metrics: AUROC, AUPRC, Brier, and ECE. Thresholded operating metrics are mixed: sensitivity improves slightly, but accuracy, balanced accuracy, specificity, precision, and F1 are slightly lower.

This is not sufficient for a ResNet50 paper-level claim. Seeds 42 and 2024 are required.
