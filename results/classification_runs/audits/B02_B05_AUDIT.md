# B02/B05 Locked Classification Audit

Source kernels:

- `hintrngia/kaggleb02`
- `hintrngia/kaggleb05`

Runs:

- B02: `D-C0 / DenseNet121 / seed 42`
- B05: `D-P / DenseNet121 / seed 42`

## Technical Validation

Both runs completed successfully.

| Run | Status | Train | Val | Test | Test prediction rows | Unique images | Mask missing | Config check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B02 D-C0 | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=False`, `use_mask_loss=False` |
| B05 D-P | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=True`, `use_mask_loss=True` |

## Seed-42 Result

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D-C0 | 0.971039 | 0.979733 | 0.838141 | 0.785897 | 0.994872 | 0.576923 | 0.125014 | 0.162900 | 135/99/2/388 |
| D-P | 0.970321 | 0.979018 | 0.862179 | 0.817949 | 0.994872 | 0.641026 | 0.112609 | 0.142219 | 150/84/2/388 |

## Delta: D-P minus D-C0

| Metric | Delta |
|---|---:|
| AUROC | -0.000718 |
| AUPRC | -0.000715 |
| Accuracy | +0.024038 |
| Balanced accuracy | +0.032051 |
| Sensitivity | +0.000000 |
| Specificity | +0.064103 |
| Brier | -0.012406 |
| ECE | -0.020680 |

## Current DenseNet121 Two-Seed Mean

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-C0 | 0.968538 | 0.977786 | 0.840545 | 0.789957 | 0.992308 | 0.587607 | 0.133565 | 0.172128 |
| D-P | 0.971562 | 0.979570 | 0.853365 | 0.806197 | 0.994872 | 0.617521 | 0.122205 | 0.157778 |

## Interpretation

For DenseNet121 seed 42, the proposed method improves thresholded operating metrics and calibration metrics, especially specificity, accuracy, balanced accuracy, Brier, and ECE. AUROC and AUPRC are slightly lower than C0 for this seed.

Across the first two DenseNet121 seeds, D-P currently has better mean AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, Brier, and ECE. The third DenseNet121 seed 2024 is still required before making a stable paper-level claim.
