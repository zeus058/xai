# E05 XAI Aggregate Summary

## Execution

E05 completed successfully using the four locked XAI runs:

- E01: D-C0 / DenseNet121
- E02: D-P / DenseNet121
- E03: R-C0 / ResNet50
- E04: R-P / ResNet50

Validation passed:

- Total rows: 1536
- Unique XAI images: 128
- Rows per setting: 384 = 128 images x 3 seeds
- Bootstrap iterations: 5000
- Bootstrap seed: 3407
- CI level: 95%

## Paper-Ready XAI Table

| Condition | Architecture | Grad-CAM LRR+ | Guided Grad-CAM LRR+ | Grad-CAM pointing | Guided Grad-CAM pointing |
|---|---|---:|---:|---:|---:|
| D-C0 | DenseNet121 | 0.3798 +/- 0.0111 | 0.3616 +/- 0.0259 | 0.4036 +/- 0.0401 | 0.2135 +/- 0.0352 |
| D-P | DenseNet121 | 0.7752 +/- 0.0215 | 0.7979 +/- 0.0359 | 0.9948 +/- 0.0045 | 0.9688 +/- 0.0271 |
| R-C0 | ResNet50 | 0.3606 +/- 0.0312 | 0.3827 +/- 0.0628 | 0.2057 +/- 0.1144 | 0.2943 +/- 0.0813 |
| R-P | ResNet50 | 0.7616 +/- 0.0008 | 0.8218 +/- 0.0140 | 0.9870 +/- 0.0045 | 0.9271 +/- 0.0352 |

## Paired P-Minus-C0 Localization Effects

| Architecture | Metric | Mean Delta | 95% CI | Supports Improvement |
|---|---|---:|---:|---|
| DenseNet121 | Grad-CAM LRR+ | +0.3955 | [0.3867, 0.4044] | Yes |
| DenseNet121 | Guided Grad-CAM LRR+ | +0.4363 | [0.4220, 0.4502] | Yes |
| DenseNet121 | Grad-CAM pointing | +0.5911 | [0.5417, 0.6406] | Yes |
| DenseNet121 | Guided Grad-CAM pointing | +0.7552 | [0.7109, 0.7995] | Yes |
| ResNet50 | Grad-CAM LRR+ | +0.4010 | [0.3924, 0.4093] | Yes |
| ResNet50 | Guided Grad-CAM LRR+ | +0.4391 | [0.4232, 0.4550] | Yes |
| ResNet50 | Grad-CAM pointing | +0.7812 | [0.7422, 0.8203] | Yes |
| ResNet50 | Guided Grad-CAM pointing | +0.6328 | [0.5807, 0.6823] | Yes |

## Stability Metrics

Input-stability LRR deltas are lower for the proposed models:

- DenseNet121 Grad-CAM stability delta: -0.0010, 95% CI [-0.0012, -0.0008]
- DenseNet121 Guided Grad-CAM stability delta: -0.0025, 95% CI [-0.0029, -0.0020]
- ResNet50 Grad-CAM stability delta: -0.0010, 95% CI [-0.0011, -0.0008]
- ResNet50 Guided Grad-CAM stability delta: -0.0024, 95% CI [-0.0029, -0.0020]

This supports a cautious claim that localization scores are more stable under small input perturbations.

## Faithfulness And Sanity Caveats

Deletion/insertion results are mixed:

- DenseNet121 insertion AUC decreases for D-P, which is not favorable under the usual insertion interpretation.
- ResNet50 insertion AUC improves for R-P.
- Deletion AUC increases for both proposed models. The interpretation depends on the convention used, so this should not be over-claimed.

Head-randomization correlations are higher for the proposed models:

- DenseNet121 Grad-CAM randomization corr delta: +0.2886
- DenseNet121 Guided Grad-CAM randomization corr delta: +0.0882
- ResNet50 Grad-CAM randomization corr delta: +0.1449
- ResNet50 Guided Grad-CAM randomization corr delta: +0.0501

Because lower post-randomization correlation is generally preferable, these results do not support a claim that the proposed models improve randomization sanity.

## Recommended Manuscript Claim

The strongest supported XAI claim is:

> The proposed CBAM and mask-guided training strategy substantially increases the concentration of Grad-CAM and Guided Grad-CAM relevance within lung regions across both DenseNet121 and ResNet50 on a locked 128-image XAI sample.

The paper may also state cautiously:

> Localization scores under small input perturbations are more stable for the proposed models.

Avoid claiming:

- that the proposed models are globally more faithful under deletion/insertion;
- that randomization sanity improves;
- that XAI proves causal reasoning.

## Generated Outputs

Main tables:

- `xai_all_by_image.csv`
- `xai_summary_by_setting.csv`
- `xai_p_minus_c0_paired_delta_ci.csv`
- `xai_summary_table.csv`
- `xai_delta_table.csv`
- `xai_pointing_sign_tests.csv`
- `xai_aggregate_audit.json`

Figures:

- `figures/xai_localization_metrics_bar.png`
- `figures/xai_p_minus_c0_localization_delta_ci.png`

