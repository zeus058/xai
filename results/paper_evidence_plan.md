# Final Experiment Evidence And Paper Insertion Plan

## Purpose

This file defines the locked, paper-ready experimental evidence that should replace the outdated values currently present in `paper/main.tex`.

All numbers below are taken from the locked final artifacts under `results/locked_final/`.

## Evidence Status

| Evidence Block | Status | Use In Paper | Source Files |
|---|---|---|---|
| A01 original-paper reproduction | Complete, did not pass +/-2% tolerance | Reproducibility limitation, not main improvement claim | `reproduction/reference_reproduction_audit.md` |
| B01-B12 classification battery | Complete and internally validated | Main classification results | `statistics/classification_summary_table.csv`, `statistics/paired_delta_bootstrap_ci.csv` |
| D01 classification statistics | Complete, 5000 bootstrap iterations | Confidence intervals and paired P-C0 effects | `statistics/` |
| C00 XAI manifest | Complete | XAI sample definition | `xai/xai_sample_manifest_128.csv` |
| E01-E04 XAI battery | Complete | Main XAI localization evidence | `xai_runs/E01*` to `E04*` |
| E05 XAI aggregate | Complete, 5000 bootstrap iterations | XAI tables, figures, paired localization CIs | `xai_runs/aggregate/` |

## Dataset And Split Values

Use these values consistently:

- Source images retained: 5840
- Development set: 5216
- Sealed test set: 624
- Training set: 4411
- Validation/tuning set: 779
- Test label counts: 234 normal, 390 pneumonia
- XAI locked sample: 128 images, balanced 64 normal and 64 pneumonia
- XAI repeated across 3 seeds per setting, giving 384 rows per setting

## Original-Paper Reproduction

The manuscript must not claim exact reproduction of the reference paper.

Paper-ready wording:

> Because the reference study did not publish full seed values, image-level split assignments, and all training-selection details, exact reproduction was not possible. We therefore performed a best-effort reproduction under a cleaned and auditable protocol.

Observed A01 reproduction:

| Architecture | Mean Accuracy | Reference Accuracy | Difference | Mean AUROC | Reference AUROC | Difference | Pass +/-2% |
|---|---:|---:|---:|---:|---:|---:|---|
| DenseNet121 | 0.812500 | 0.891 | -0.078500 | 0.954562 | 0.980 | -0.025438 | No |
| ResNet50 | 0.815171 | 0.844 | -0.028829 | 0.954854 | 0.950 | +0.004854 | No |

Correct interpretation:

- Use A01 only as a reproducibility limitation.
- Do not use A01 as the baseline for proposed-method improvement.
- Main improvement claims must compare locked P versus locked C0.

## Main Classification Results

Use `statistics/classification_summary_table.csv`.

| Condition | Architecture | AUROC | AUPRC | Accuracy | Balanced Accuracy | Sensitivity | Specificity | Precision | NPV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-C0 | DenseNet121 | 0.9677 +/- 0.0029 | 0.9767 +/- 0.0027 | 0.8360 +/- 0.0082 | 0.7833 +/- 0.0122 | 0.9940 +/- 0.0039 | 0.5726 +/- 0.0280 | 0.7951 +/- 0.0100 | 0.9833 +/- 0.0102 |
| D-P | DenseNet121 | 0.9746 +/- 0.0055 | 0.9825 +/- 0.0050 | 0.8462 +/- 0.0153 | 0.7963 +/- 0.0208 | 0.9957 +/- 0.0015 | 0.5969 +/- 0.0428 | 0.8048 +/- 0.0165 | 0.9883 +/- 0.0035 |
| R-C0 | ResNet50 | 0.9699 +/- 0.0018 | 0.9789 +/- 0.0023 | 0.8360 +/- 0.0418 | 0.7828 +/- 0.0562 | 0.9957 +/- 0.0015 | 0.5698 +/- 0.1138 | 0.7957 +/- 0.0423 | 0.9879 +/- 0.0023 |
| R-P | ResNet50 | 0.9731 +/- 0.0030 | 0.9803 +/- 0.0019 | 0.8488 +/- 0.0398 | 0.7993 +/- 0.0531 | 0.9974 +/- 0.0000 | 0.6011 +/- 0.1062 | 0.8079 +/- 0.0404 | 0.9928 +/- 0.0014 |

Additional calibration metrics:

| Condition | Architecture | Brier | ECE |
|---|---|---:|---:|
| D-C0 | DenseNet121 | 0.1386 +/- 0.0122 | 0.1779 +/- 0.0136 |
| D-P | DenseNet121 | 0.1258 +/- 0.0115 | 0.1627 +/- 0.0177 |
| R-C0 | ResNet50 | 0.1399 +/- 0.0049 | 0.1847 +/- 0.0052 |
| R-P | ResNet50 | 0.1273 +/- 0.0081 | 0.1700 +/- 0.0068 |

## Classification P-Minus-C0 Effects

Use `statistics/paired_delta_bootstrap_ci.csv`.

### DenseNet121

| Metric | Mean Delta | 95% CI | Paper Interpretation |
|---|---:|---:|---|
| AUROC | +0.0069 | [0.0034, 0.0108] | Supported |
| AUPRC | +0.0057 | [0.0026, 0.0092] | Supported |
| Accuracy | +0.0101 | [0.0016, 0.0192] | Supported |
| Balanced accuracy | +0.0130 | [0.0014, 0.0246] | Supported |
| Specificity | +0.0242 | [0.0014, 0.0470] | Supported |
| F1 | +0.0066 | [0.0011, 0.0124] | Supported |
| Brier score | -0.0128 | [-0.0166, -0.0089] | Supported, lower is better |
| ECE | -0.0153 | [-0.0190, -0.0113] | Supported, lower is better |

### ResNet50

| Metric | Mean Delta | 95% CI | Paper Interpretation |
|---|---:|---:|---|
| AUROC | +0.0032 | [-0.0003, 0.0067] | Directional only |
| AUPRC | +0.0014 | [-0.0026, 0.0055] | Directional only |
| Accuracy | +0.0128 | [0.0027, 0.0230] | Supported |
| Balanced accuracy | +0.0165 | [0.0034, 0.0296] | Supported |
| Specificity | +0.0313 | [0.0057, 0.0570] | Supported |
| F1 | +0.0082 | [0.0020, 0.0145] | Supported |
| Brier score | -0.0127 | [-0.0164, -0.0088] | Supported, lower is better |
| ECE | -0.0147 | [-0.0184, -0.0107] | Supported, lower is better |

Classification claim allowed:

> The proposed configuration produced modest but consistent gains, most clearly in specificity, balanced accuracy, F1, Brier score, and ECE. DenseNet121 additionally showed supported AUROC and AUPRC gains, while ResNet50 ranking-metric gains were directional but not supported by the paired 95% CI.

Classification claim not allowed:

- Do not claim dramatic performance gains.
- Do not claim universal AUROC/AUPRC superiority across both backbones.
- Do not claim sensitivity improvement; sensitivity is already saturated and the paired CI touches zero.
- Do not claim the models are well calibrated; say calibration improved, while absolute ECE remains non-trivial.

## Main XAI Results

Use `xai_runs/aggregate/xai_summary_table.csv`.

| Condition | Architecture | Grad-CAM LRR+ | Guided Grad-CAM LRR+ | Grad-CAM Pointing | Guided Grad-CAM Pointing |
|---|---|---:|---:|---:|---:|
| D-C0 | DenseNet121 | 0.3798 +/- 0.0111 | 0.3616 +/- 0.0259 | 0.4036 +/- 0.0401 | 0.2135 +/- 0.0352 |
| D-P | DenseNet121 | 0.7752 +/- 0.0215 | 0.7979 +/- 0.0359 | 0.9948 +/- 0.0045 | 0.9688 +/- 0.0271 |
| R-C0 | ResNet50 | 0.3606 +/- 0.0312 | 0.3827 +/- 0.0628 | 0.2057 +/- 0.1144 | 0.2943 +/- 0.0813 |
| R-P | ResNet50 | 0.7616 +/- 0.0008 | 0.8218 +/- 0.0140 | 0.9870 +/- 0.0045 | 0.9271 +/- 0.0352 |

## XAI P-Minus-C0 Effects

Use `xai_runs/aggregate/xai_p_minus_c0_paired_delta_ci.csv`.

| Architecture | Metric | Mean Delta | 95% CI | Supported |
|---|---|---:|---:|---|
| DenseNet121 | Grad-CAM LRR+ | +0.3955 | [0.3867, 0.4044] | Yes |
| DenseNet121 | Guided Grad-CAM LRR+ | +0.4363 | [0.4220, 0.4502] | Yes |
| DenseNet121 | Grad-CAM pointing | +0.5911 | [0.5417, 0.6406] | Yes |
| DenseNet121 | Guided Grad-CAM pointing | +0.7552 | [0.7109, 0.7995] | Yes |
| ResNet50 | Grad-CAM LRR+ | +0.4010 | [0.3924, 0.4093] | Yes |
| ResNet50 | Guided Grad-CAM LRR+ | +0.4391 | [0.4232, 0.4550] | Yes |
| ResNet50 | Grad-CAM pointing | +0.7812 | [0.7422, 0.8203] | Yes |
| ResNet50 | Guided Grad-CAM pointing | +0.6328 | [0.5807, 0.6823] | Yes |

XAI claim allowed:

> The proposed CBAM plus mask-guided training strategy substantially increased the concentration of Grad-CAM and Guided Grad-CAM relevance inside lung regions across both DenseNet121 and ResNet50.

XAI claim allowed with caution:

> The proposed models showed lower input-stability LRR deltas under small perturbations.

XAI claim not allowed:

- Do not claim lesion localization, because there are no lesion masks or expert lesion annotations.
- Do not claim globally superior XAI faithfulness.
- Do not claim randomization sanity improves. Head-randomization correlations are higher for proposed models.
- Do not use old P-only D05 wording; E01-E04 now compare C0 versus P for both backbones.

## Figures To Use

Primary aggregate figures:

- `xai_runs/aggregate/figures/xai_localization_metrics_bar.png`
- `xai_runs/aggregate/figures/xai_p_minus_c0_localization_delta_ci.png`

Qualitative XAI figures:

- Select representative panels from E01-E04 figure folders.
- Recommended paper layout:
  - Original X-ray
  - Lung mask
  - C0 Guided Grad-CAM
  - P Guided Grad-CAM
- Use 4 to 6 representative cases in the main paper.
- Put the larger figure set in supplementary material.

Caption requirement:

> Lung masks are used for localization quantification and for the proposed mask-guided auxiliary training loss. Classification inference uses the input radiograph only.

## Current Paper Sections That Must Be Updated

The current `paper/main.tex` contains outdated numbers and claims.

Must update:

- Abstract: replace old AUROC/XAI values and remove P-only XAI framing.
- Introduction/RQs: update XAI RQ from Guided-vs-Grad-CAM within P-only to C0-vs-P localization under both attribution methods.
- Methods/Statistical Analysis: replace P-only D05 wording with paired C0-vs-P XAI comparison over locked 128-image sample.
- Results/Classification Performance: replace entire current classification paragraph and table.
- Results/Component-Wise Ablation: remove or move old ablation table unless it is clearly labeled exploratory/development-only; it is not part of the locked final evidence.
- Results/Calibration: replace tuning-set calibration claims with sealed-test Brier/ECE from D01.
- Results/Lung-Field Masks: replace 180-sample wording with locked 128-image XAI sample and C00 coverage.
- Results/Primary DenseNet121 XAI Comparison: replace old Guided-vs-Grad-CAM P-only table with D-C0 versus D-P table.
- Results/Secondary ResNet50 XAI Replication: replace R-P-only table with R-C0 versus R-P table.
- Results/Validity and Qualitative Analysis: rewrite deletion/insertion/randomization claims conservatively.
- Discussion/Principal Findings: replace AUROC 0.958/0.971 and poor LRR statements.
- Discussion/Technical Implications: replace claims that LRR fell below 0.5; locked final P models have LRR around 0.76 to 0.82.
- Limitations: update limitations to include best-effort reproduction failure, absence of lesion annotations, and randomization caveat.
- Conclusion: replace all old numerical claims.

## Recommended Paper-Level Result Narrative

Recommended concise result narrative:

> On the locked 624-image sealed test set, the proposed configuration yielded modest but consistent classification improvements over its matched C0 baseline. DenseNet121 improved from AUROC 0.9677 +/- 0.0029 to 0.9746 +/- 0.0055, with a paired mean AUROC delta of +0.0069 (95% CI [0.0034, 0.0108]). ResNet50 improved from AUROC 0.9699 +/- 0.0018 to 0.9731 +/- 0.0030, although its paired AUROC delta was directional rather than CI-supported (+0.0032, 95% CI [-0.0003, 0.0067]). Across both backbones, the proposed configuration more clearly improved specificity, balanced accuracy, F1, Brier score, and ECE.

Recommended XAI narrative:

> On the locked 128-image XAI sample, the proposed configuration substantially increased lung-region localization. DenseNet121 Guided Grad-CAM LRR+ increased from 0.3616 +/- 0.0259 to 0.7979 +/- 0.0359, with a paired delta of +0.4363 (95% CI [0.4220, 0.4502]). ResNet50 Guided Grad-CAM LRR+ increased from 0.3827 +/- 0.0628 to 0.8218 +/- 0.0140, with a paired delta of +0.4391 (95% CI [0.4232, 0.4550]). Pointing metrics showed the same direction for both backbones.

Recommended limitation narrative:

> These XAI results quantify anatomical concentration inside automatically generated lung masks, not lesion-level correctness or clinical reasoning. Deletion/insertion and parameter-randomization results were mixed and are therefore interpreted as supplementary sanity checks rather than primary evidence of explanation faithfulness.

## Final Readiness Assessment

Ready to insert into paper:

- Classification main table
- Classification P-minus-C0 delta table
- XAI localization table
- XAI localization delta table
- Reproduction limitation paragraph
- XAI aggregate figures

Needs careful wording:

- Calibration: improved but not perfectly calibrated
- ResNet AUROC/AUPRC: directional, not 95% CI-supported
- Deletion/insertion: supplementary and mixed
- Randomization: do not claim improvement
- Ablation: old development-only ablation should not be presented as locked final evidence

