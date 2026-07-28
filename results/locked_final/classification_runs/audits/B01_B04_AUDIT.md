# B01/B04 Locked Classification Audit

Source kernels:

- `hintrngia/kaggleb01`
- `hintrngia/kaggleb04`

Runs:

- B01: `D-C0 / DenseNet121 / seed 3407`
- B04: `D-P / DenseNet121 / seed 3407`

## Technical Validation

Both runs completed successfully.

| Run | Status | Train | Val | Test | Test prediction rows | Unique images | Mask missing | Config check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B01 D-C0 | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=False`, `use_mask_loss=False` |
| B04 D-P | SUCCESS | 4411 | 779 | 624 | 624 | 624 | 0 | `use_cbam=True`, `use_mask_loss=True` |

## Seed-3407 Result

| Condition | AUROC | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | Brier | ECE | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D-C0 | 0.966037 | 0.975839 | 0.842949 | 0.794017 | 0.989744 | 0.598291 | 0.142116 | 0.181357 | 140/94/4/386 |
| D-P | 0.972803 | 0.980121 | 0.844551 | 0.794444 | 0.994872 | 0.594017 | 0.131802 | 0.173336 | 139/95/2/388 |

## Delta: D-P minus D-C0

| Metric | Delta |
|---|---:|
| AUROC | +0.006766 |
| AUPRC | +0.004282 |
| Accuracy | +0.001603 |
| Balanced accuracy | +0.000427 |
| Sensitivity | +0.005128 |
| Specificity | -0.004274 |
| Brier | -0.010314 |
| ECE | -0.008021 |

## Interpretation

For DenseNet121 seed 3407, the proposed method improves AUROC, AUPRC, Brier, ECE, and sensitivity, while accuracy and balanced accuracy are nearly unchanged and specificity is slightly lower.

This is a positive first locked result, but it is not sufficient for a paper-level claim. DenseNet121 seeds 42 and 2024 must be run before stating a stable D-P vs D-C0 effect.
