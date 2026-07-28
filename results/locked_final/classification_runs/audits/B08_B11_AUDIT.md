# B08/B11 Locked Classification Audit

Source kernels:

- `hintrngia/kaggleb8`
- `hintrngia/kaggleb11`

Runs:

- B08: `R-C0 / ResNet50 / seed 42`
- B11: `R-P / ResNet50 / seed 42`

## Technical Validation

Both runs completed successfully.

| Run | Status | Train | Val | Test | Test prediction rows | Unique images | Mask missing | Config check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B08 R-C0 | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=False`, `use_mask_loss=False` |
| B11 R-P | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=True`, `use_mask_loss=True` |

## Seed-42 Result

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R-C0 | 0.968305 | 0.978867 | 0.790064 | 0.720940 | 0.997436 | 0.444444 | 0.140538 | 0.184839 | 104/130/1/389 |
| R-P | 0.970080 | 0.978305 | 0.802885 | 0.738034 | 0.997436 | 0.478632 | 0.136449 | 0.177315 | 112/122/1/389 |

## Delta: R-P minus R-C0

| Metric | Delta |
|---|---:|
| AUROC | +0.001775 |
| AUPRC | -0.000562 |
| Accuracy | +0.012821 |
| Balanced accuracy | +0.017094 |
| Sensitivity | +0.000000 |
| Specificity | +0.034188 |
| Brier | -0.004090 |
| ECE | -0.007523 |

## Current ResNet50 Two-Seed Mean

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R-C0 | 0.970110 | 0.980020 | 0.830929 | 0.775855 | 0.996154 | 0.555556 | 0.137651 | 0.182103 |
| R-P | 0.973071 | 0.980210 | 0.836538 | 0.782906 | 0.997436 | 0.568376 | 0.130278 | 0.173028 |

## Interpretation

For ResNet50 seed 42, the proposed method improves AUROC, accuracy, balanced accuracy, specificity, Brier, and ECE. AUPRC is slightly lower. Sensitivity is unchanged.

Across the first two ResNet50 seeds, R-P currently has better mean AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, Brier, and ECE. Seed 2024 remains required before a paper-level ResNet50 claim.
