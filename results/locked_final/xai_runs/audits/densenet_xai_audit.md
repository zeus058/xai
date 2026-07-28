# E01-E02 DenseNet XAI Audit

## Scope

This audit covers the first two locked Guided Grad-CAM validity notebooks:

- E01: D-C0 / DenseNet121 baseline
- E02: D-P / DenseNet121 proposed model

Each notebook evaluates the locked C00 128-image XAI manifest across three seeds: 3407, 42, and 2024.

## Artifact Integrity

Both notebooks completed successfully.

| Run | Expected Rows | Output Rows | Unique Images | Seeds | Labels |
|---|---:|---:|---:|---|---|
| E01 D-C0 DenseNet121 | 384 | 384 | 128 | 42, 2024, 3407 | 192 normal, 192 pneumonia |
| E02 D-P DenseNet121 | 384 | 384 | 128 | 42, 2024, 3407 | 192 normal, 192 pneumonia |

No missing values were found in the main XAI metrics.

## Main XAI Metrics

| Metric | D-C0 Mean | D-P Mean | Mean Delta |
|---|---:|---:|---:|
| Accuracy on XAI sample | 0.8359 | 0.8438 | +0.0078 |
| Grad-CAM LRR+ | 0.3798 | 0.7752 | +0.3955 |
| Guided Grad-CAM LRR+ | 0.3616 | 0.7979 | +0.4363 |
| Grad-CAM pointing inside lung | 0.4036 | 0.9948 | +0.5911 |
| Guided Grad-CAM pointing inside lung | 0.2135 | 0.9688 | +0.7552 |
| Grad-CAM deletion AUC | 0.7235 | 0.7692 | +0.0456 |
| Guided Grad-CAM deletion AUC | 0.7221 | 0.7450 | +0.0228 |
| Grad-CAM insertion AUC | 0.8625 | 0.8362 | -0.0264 |
| Guided Grad-CAM insertion AUC | 0.8738 | 0.8638 | -0.0100 |
| Grad-CAM input-stability LRR delta | 0.0016 | 0.0007 | -0.0010 |
| Guided Grad-CAM input-stability LRR delta | 0.0043 | 0.0019 | -0.0025 |
| Grad-CAM head-randomization corr | 0.1774 | 0.4660 | +0.2886 |
| Guided Grad-CAM head-randomization corr | 0.4328 | 0.5209 | +0.0882 |

## Paired Directionality

The D-P model improves lung-localization metrics very consistently:

- Grad-CAM LRR+: D-P > D-C0 for 384/384 paired seed-image cases.
- Guided Grad-CAM LRR+: D-P > D-C0 for 384/384 paired seed-image cases.
- Grad-CAM pointing inside lung: D-P improves in 227 cases, ties in the rest, and is worse in 0 cases.
- Guided Grad-CAM pointing inside lung: D-P improves in 291 cases and is worse in 1 case.

Per-seed Guided Grad-CAM LRR+:

| Seed | D-C0 | D-P | Delta |
|---:|---:|---:|---:|
| 42 | 0.3353 | 0.7848 | +0.4495 |
| 2024 | 0.3872 | 0.8385 | +0.4513 |
| 3407 | 0.3622 | 0.7703 | +0.4081 |

## Interpretation

The DenseNet XAI results strongly support the localization component of the proposed method. The proposed D-P model places substantially more Grad-CAM and Guided Grad-CAM relevance inside the lung mask than the D-C0 baseline, and this effect is consistent across all three seeds and all 128 locked XAI images.

This is aligned with the research hypothesis that CBAM plus lung-mask-guided training shifts model attention toward clinically relevant lung regions.

## Important Caveats

The deletion/insertion metrics are mixed. D-P has higher deletion AUC and lower insertion AUC. Depending on the exact deletion/insertion convention, lower deletion AUC and higher insertion AUC are often interpreted as more faithful explanations. These metrics must therefore be reviewed carefully before making a strong faithfulness claim.

The head-randomization correlations are higher for D-P than D-C0. This is not ideal for a sanity-check interpretation because lower correlation after randomization is generally better. The paper should avoid claiming that D-P improves randomization sanity unless the aggregation notebook confirms a more nuanced interpretation.

The most defensible claim from E01/E02 is localization, not broad XAI faithfulness:

> In DenseNet121, the proposed model substantially increases the proportion of Grad-CAM and Guided Grad-CAM relevance inside the lung region on the locked 128-image XAI sample.

## Status

E01/E02 are usable as DenseNet XAI evidence after final aggregation with E03/E04 and E05. Do not finalize the XAI section until the corresponding ResNet XAI runs and aggregate paired statistics are complete.

