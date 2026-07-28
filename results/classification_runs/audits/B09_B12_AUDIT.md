# B09/B12 Locked Classification Audit

Source kernels:

- `hintrngia/kaggleb9`
- `hintrngia/kaggleb12`

Runs:

- B09: `R-C0 / ResNet50 / seed 2024`
- B12: `R-P / ResNet50 / seed 2024`

## Technical Validation

Both runs completed successfully.

| Run | Status | Train | Val | Test | Test prediction rows | Unique images | Mask missing | Config check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B09 R-C0 | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=False`, `use_mask_loss=False` |
| B12 R-P | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=True`, `use_mask_loss=True` |

## Seed-2024 Result

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R-C0 | 0.969494 | 0.976562 | 0.846154 | 0.796581 | 0.994872 | 0.598291 | 0.144525 | 0.189809 | 140/94/2/388 |
| R-P | 0.973077 | 0.980335 | 0.873397 | 0.832051 | 0.997436 | 0.666667 | 0.121265 | 0.163995 | 156/78/1/389 |

## Delta: R-P minus R-C0

| Metric | Delta |
|---|---:|
| AUROC | +0.003583 |
| AUPRC | +0.003773 |
| Accuracy | +0.027244 |
| Balanced accuracy | +0.035470 |
| Sensitivity | +0.002564 |
| Specificity | +0.068376 |
| Brier | -0.023261 |
| ECE | -0.025815 |

## Interpretation

For ResNet50 seed 2024, the proposed method improves every reported primary classification and calibration metric.

Together with B07/B10 and B08/B11, the ResNet50 arm now has a complete 3-seed locked classification battery.
