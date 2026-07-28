# E03-E04 ResNet XAI Audit

## Scope

This audit covers:

- E03: R-C0 / ResNet50 baseline
- E04: R-P / ResNet50 proposed model

Each notebook evaluates the locked C00 128-image XAI manifest across seeds 3407, 42, and 2024.

## Artifact Integrity

Both notebooks completed successfully.

| Run | Expected Rows | Output Rows | Unique Images | Seeds | Labels |
|---|---:|---:|---:|---|---|
| E03 R-C0 ResNet50 | 384 | 384 | 128 | 42, 2024, 3407 | 192 normal, 192 pneumonia |
| E04 R-P ResNet50 | 384 | 384 | 128 | 42, 2024, 3407 | 192 normal, 192 pneumonia |

No missing values were found in the main XAI metrics.

## Main XAI Metrics

| Metric | R-C0 Mean | R-P Mean | Mean Delta |
|---|---:|---:|---:|
| Accuracy on XAI sample | 0.8438 | 0.8464 | +0.0026 |
| Grad-CAM LRR+ | 0.3606 | 0.7616 | +0.4010 |
| Guided Grad-CAM LRR+ | 0.3827 | 0.8218 | +0.4391 |
| Grad-CAM pointing inside lung | 0.2057 | 0.9870 | +0.7812 |
| Guided Grad-CAM pointing inside lung | 0.2943 | 0.9271 | +0.6328 |
| Grad-CAM deletion AUC | 0.7070 | 0.7754 | +0.0683 |
| Guided Grad-CAM deletion AUC | 0.7153 | 0.7585 | +0.0431 |
| Grad-CAM insertion AUC | 0.8254 | 0.8423 | +0.0168 |
| Guided Grad-CAM insertion AUC | 0.8446 | 0.8604 | +0.0158 |
| Grad-CAM input-stability LRR delta | 0.0014 | 0.0004 | -0.0010 |
| Guided Grad-CAM input-stability LRR delta | 0.0041 | 0.0017 | -0.0024 |
| Grad-CAM head-randomization corr | 0.2997 | 0.4446 | +0.1449 |
| Guided Grad-CAM head-randomization corr | 0.4203 | 0.4704 | +0.0501 |

## Paired Directionality

The R-P model improves lung localization metrics very consistently:

- Grad-CAM LRR+: R-P > R-C0 for 384/384 paired seed-image cases.
- Guided Grad-CAM LRR+: R-P > R-C0 for 383/384 paired seed-image cases.
- Grad-CAM pointing inside lung: R-P improves in 300 cases, ties in the rest, and is worse in 0 cases.
- Guided Grad-CAM pointing inside lung: R-P improves in 249 cases and is worse in 6 cases.

Per-seed Guided Grad-CAM LRR+:

| Seed | R-C0 | R-P | Delta |
|---:|---:|---:|---:|
| 42 | 0.3174 | 0.8379 | +0.5205 |
| 2024 | 0.3883 | 0.8148 | +0.4265 |
| 3407 | 0.4426 | 0.8127 | +0.3701 |

## Interpretation

The ResNet50 XAI results strongly support the localization claim. The proposed R-P model shifts both Grad-CAM and Guided Grad-CAM relevance into the lung region on the locked XAI sample, consistently across seeds.

The ResNet pattern matches the DenseNet pattern, which strengthens the architecture-independent localization claim.

## Important Caveats

Head-randomization correlations are higher for R-P than R-C0. As with DenseNet, this does not support a stronger randomization-sanity claim for the proposed model.

Deletion/insertion metrics are more favorable for R-P than in DenseNet, but should still be aggregated and interpreted in E05 before being used as a major paper claim.

The most defensible claim from E03/E04 is:

> In ResNet50, the proposed model substantially increases the concentration of Grad-CAM and Guided Grad-CAM relevance inside the lung region.

## Status

E03/E04 are usable as ResNet XAI evidence. The next required step is E05 aggregation across all four XAI runs.

