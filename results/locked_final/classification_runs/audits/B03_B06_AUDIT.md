# B03/B06 Locked Classification Audit

Source kernels:

- `hintrngia/kaggleb03`
- `hintrngia/kaggleb06`

Runs:

- B03: `D-C0 / DenseNet121 / seed 2024`
- B06: `D-P / DenseNet121 / seed 2024`

## Technical Validation

Both runs completed successfully.

| Run | Status | Train | Val | Test | Test prediction rows | Unique images | Mask missing | Config check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B03 D-C0 | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=False`, `use_mask_loss=False` |
| B06 D-P | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=True`, `use_mask_loss=True` |

## Seed-2024 Result

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D-C0 | 0.966157 | 0.974641 | 0.826923 | 0.770085 | 0.997436 | 0.542735 | 0.148725 | 0.189456 | 127/107/1/389 |
| D-P | 0.980802 | 0.988247 | 0.831731 | 0.776496 | 0.997436 | 0.555556 | 0.133086 | 0.172397 | 130/104/1/389 |

## Delta: D-P minus D-C0

| Metric | Delta |
|---|---:|
| AUROC | +0.014645 |
| AUPRC | +0.013607 |
| Accuracy | +0.004808 |
| Balanced accuracy | +0.006410 |
| Sensitivity | +0.000000 |
| Specificity | +0.012821 |
| Brier | -0.015639 |
| ECE | -0.017059 |

## Interpretation

For DenseNet121 seed 2024, the proposed method improves AUROC, AUPRC, accuracy, balanced accuracy, specificity, Brier, and ECE. Sensitivity is unchanged.

Together with B01/B04 and B02/B05, the DenseNet121 arm now has a complete 3-seed locked classification battery.
