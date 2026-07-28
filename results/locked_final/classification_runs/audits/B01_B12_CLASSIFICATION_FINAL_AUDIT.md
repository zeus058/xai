# B01-B12 Locked Classification Final Audit

## Technical Status

All 12 locked classification runs completed successfully.

- Conditions: `D-C0`, `D-P`, `R-C0`, `R-P`
- Architectures: DenseNet121, ResNet50
- Seeds: 3407, 42, 2024
- Test predictions per run: 624
- Unique test images per run: 624
- Test class distribution per run: 234 normal, 390 pneumonia
- Dataset coverage issues: none detected
- Config mismatches: none detected

## Three-Seed Summary

| Condition | Architecture | AUROC mean ± SD | AUPRC mean ± SD | Accuracy mean ± SD | Balanced Acc. mean ± SD | Sensitivity mean ± SD | Specificity mean ± SD | Brier mean ± SD | ECE mean ± SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-C0 | DenseNet121 | 0.967744 ± 0.002854 | 0.976738 ± 0.002662 | 0.836004 ± 0.008224 | 0.783333 ± 0.012170 | 0.994017 ± 0.003917 | 0.572650 ± 0.028023 | 0.138618 ± 0.012236 | 0.177904 ± 0.013611 |
| D-P | DenseNet121 | 0.974642 ± 0.005477 | 0.982462 ± 0.005040 | 0.846154 ± 0.015287 | 0.796296 ± 0.020788 | 0.995726 ± 0.001480 | 0.596866 ± 0.042806 | 0.125832 ± 0.011470 | 0.162651 ± 0.017700 |
| R-C0 | ResNet50 | 0.969905 ± 0.001840 | 0.978867 ± 0.002306 | 0.836004 ± 0.041800 | 0.782764 ± 0.056203 | 0.995726 ± 0.001480 | 0.569801 ± 0.113818 | 0.139943 ± 0.004907 | 0.184672 ± 0.005223 |
| R-P | ResNet50 | 0.973073 ± 0.002991 | 0.980252 ± 0.001907 | 0.848825 ± 0.039818 | 0.799288 ± 0.053090 | 0.997436 ± 0.000000 | 0.601140 ± 0.106180 | 0.127273 ± 0.008072 | 0.170017 ± 0.006752 |

## Mean Delta: Proposed minus Control

DenseNet121, `D-P minus D-C0`:

- AUROC: +0.006898
- AUPRC: +0.005724
- Accuracy: +0.010150
- Balanced accuracy: +0.012963
- Sensitivity: +0.001709
- Specificity: +0.024216
- Brier: -0.012786
- ECE: -0.015253

ResNet50, `R-P minus R-C0`:

- AUROC: +0.003168
- AUPRC: +0.001386
- Accuracy: +0.012821
- Balanced accuracy: +0.016524
- Sensitivity: +0.001709
- Specificity: +0.031339
- Brier: -0.012669
- ECE: -0.014655

## Interpretation

The locked classification battery supports a consistent improvement of the proposed method over the matched C0 control in mean AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, Brier score, and ECE for both DenseNet121 and ResNet50.

The strongest and most stable improvements are in calibration/error metrics (`Brier`, `ECE`) and specificity/balanced accuracy. AUROC/AUPRC improvements are positive but modest, so final paper claims should report effect sizes with paired confidence intervals rather than overstate the magnitude.

## Remaining Work Before Paper Claim

This classification battery is complete, but the final paper still needs:

- Bootstrap confidence intervals.
- Paired C0-vs-P statistical comparisons.
- McNemar tests for thresholded predictions.
- XAI battery on the locked 128-case manifest.
- Final aggregation notebook that maps every paper number to a file path.
