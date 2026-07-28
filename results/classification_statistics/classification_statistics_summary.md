# D01 Statistical Validation Summary

## Execution

The local D01 statistical validation completed successfully.

- Input: 12 locked B01-B12 `test_predictions.csv` files
- Test set per run: 624 images
- Bootstrap iterations: 5000
- Bootstrap seed: 3407
- CI level: 95%
- Input validation: passed

## Paper-Ready Classification Summary

| Condition | Architecture | AUROC | AUPRC | Accuracy | Balanced Accuracy |
|---|---|---:|---:|---:|---:|
| D-C0 | DenseNet121 | 0.9677 +/- 0.0029 | 0.9767 +/- 0.0027 | 0.8360 +/- 0.0082 | 0.7833 +/- 0.0122 |
| D-P | DenseNet121 | 0.9746 +/- 0.0055 | 0.9825 +/- 0.0050 | 0.8462 +/- 0.0153 | 0.7963 +/- 0.0208 |
| R-C0 | ResNet50 | 0.9699 +/- 0.0018 | 0.9789 +/- 0.0023 | 0.8360 +/- 0.0418 | 0.7828 +/- 0.0562 |
| R-P | ResNet50 | 0.9731 +/- 0.0030 | 0.9803 +/- 0.0019 | 0.8488 +/- 0.0398 | 0.7993 +/- 0.0531 |

## Paired P-minus-C0 Bootstrap Results

### DenseNet121

| Metric | Mean Delta | 95% CI | Supports Improvement |
|---|---:|---:|---|
| AUROC | +0.0069 | [0.0034, 0.0108] | Yes |
| AUPRC | +0.0057 | [0.0026, 0.0092] | Yes |
| Accuracy | +0.0101 | [0.0016, 0.0192] | Yes |
| Balanced accuracy | +0.0130 | [0.0014, 0.0246] | Yes |
| Sensitivity | +0.0017 | [0.0000, 0.0043] | No |
| Specificity | +0.0242 | [0.0014, 0.0470] | Yes |
| Precision | +0.0097 | [0.0009, 0.0188] | Yes |
| NPV | +0.0051 | [0.0001, 0.0121] | Yes |
| F1 | +0.0066 | [0.0011, 0.0124] | Yes |
| Brier score | -0.0128 | [-0.0166, -0.0089] | Yes |
| ECE | -0.0153 | [-0.0190, -0.0113] | Yes |

### ResNet50

| Metric | Mean Delta | 95% CI | Supports Improvement |
|---|---:|---:|---|
| AUROC | +0.0032 | [-0.0003, 0.0067] | No |
| AUPRC | +0.0014 | [-0.0026, 0.0055] | No |
| Accuracy | +0.0128 | [0.0027, 0.0230] | Yes |
| Balanced accuracy | +0.0165 | [0.0034, 0.0296] | Yes |
| Sensitivity | +0.0017 | [0.0000, 0.0043] | No |
| Specificity | +0.0313 | [0.0057, 0.0570] | Yes |
| Precision | +0.0122 | [0.0021, 0.0223] | Yes |
| NPV | +0.0049 | [0.0000, 0.0117] | Yes |
| F1 | +0.0082 | [0.0020, 0.0145] | Yes |
| Brier score | -0.0127 | [-0.0164, -0.0088] | Yes |
| ECE | -0.0147 | [-0.0184, -0.0107] | Yes |

## McNemar Tests

Single-seed McNemar evidence is mixed:

- DenseNet121 seed 42 is significant: p = 0.0059
- DenseNet121 seeds 3407 and 2024 are not significant
- ResNet50 seed 2024 is significant: p = 0.0009
- ResNet50 seeds 3407 and 42 are not significant

Pooled seed-run McNemar results favor the proposed model:

- DenseNet121 pooled: p = 0.0271
- ResNet50 pooled: p = 0.0138

These pooled rows reuse the same locked test images across seeds, so they should be reported cautiously or relegated to supplementary analysis.

## Interpretation For Manuscript

The locked classification evidence supports a modest but consistent improvement from the proposed method.

The strongest support is for:

- specificity,
- balanced accuracy,
- precision,
- F1,
- Brier score,
- ECE.

DenseNet121 also has clear support for AUROC and AUPRC gains. ResNet50 does not have 95% CI support for AUROC/AUPRC gains, although its threshold-dependent and calibration metrics improve.

The paper should avoid claiming broad sensitivity improvement because sensitivity is already saturated and the paired CI touches zero.

The correct high-level claim is:

> The proposed attention and lung-mask-guided training improves classification robustness primarily through better specificity, balanced accuracy, and calibration, with stronger ranking-metric evidence for DenseNet121 than for ResNet50.

## Required Caveats

- These results validate classification performance only.
- Shortcut-learning and localization claims still require the locked Guided Grad-CAM validity battery.
- Bootstrap CIs are conditional on the locked 624-image test set.
- Pooled McNemar tests reuse the same test images across seeds and should be interpreted cautiously.

