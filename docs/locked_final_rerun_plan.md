# Locked Final Rerun Plan for Q1/A/A* Paper Evidence

Date: 2026-07-27  
Target environment: Kaggle, parallel notebooks, GPU T4x2 where available  
Datasets:

- `hintrngia/gate46-development-only-cxr`
- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`

Final models:

- DenseNet121
- ResNet50

Final conditions:

- `D-C0`: DenseNet121 control, `use_cbam=False`, `use_mask_loss=False`
- `D-P`: DenseNet121 proposed, `use_cbam=True`, `use_mask_loss=True`
- `R-C0`: ResNet50 control, `use_cbam=False`, `use_mask_loss=False`
- `R-P`: ResNet50 proposed, `use_cbam=True`, `use_mask_loss=True`

Final seeds:

- `3407`
- `42`
- `2024`

## 1. Non-Negotiable Rules

1. Use the development dataset only for training, validation, calibration, threshold selection, and hyperparameter decisions.
2. Use the sealed test dataset exactly once per final run for final evaluation.
3. Do not choose thresholds, model checkpoints, XAI samples, or paper tables based on sealed test performance.
4. Use the same train/tuning split for all model-condition-seed runs.
5. Use the same 128-case locked XAI sample manifest for every XAI run.
6. Save every per-case prediction for all 624 sealed test samples.
7. Save every config, package version, dataset manifest, and random seed with each run.
8. Treat any interrupted, failed, manually edited, or partially rerun notebook as invalid unless fully restarted and the rerun is logged.
9. Do not report old deletion/insertion/randomization/stability values from pre-cleanup artifacts. These must be regenerated from the fixed active XAI code.

## 2. Required Notebook Set

Run these notebooks in parallel where possible. Use one notebook per final run to reduce failure blast radius.

### Phase A: Original Paper Reproduction

Notebook:

- `KAGGLE_A01_original_reproduction.ipynb`

Purpose:

- Reproduce the original reference-paper baseline before claiming improvement.
- Use the same train/tuning/test policy as the final pipeline where possible.
- Match the original paper's reported primary metric within approximately +/-2 percentage points.

Required outputs:

- `reproduction/original_reproduction_config.json`
- `reproduction/original_reproduction_metrics_by_seed.csv`
- `reproduction/original_reproduction_predictions_seed3407.csv`
- `reproduction/original_reproduction_predictions_seed42.csv`
- `reproduction/original_reproduction_predictions_seed2024.csv`
- `reproduction/original_reproduction_summary.json`

Pass/fail:

- Pass if the primary metric from the reference paper is reproduced within +/-2 percentage points.
- If not pass, the final paper must explicitly say the original baseline could not be reproduced and must not frame the comparison as a strict reproduced improvement.

### Phase B: Final Classification Battery

Create 12 notebooks:

- `KAGGLE_B01_D_C0_seed3407.ipynb`
- `KAGGLE_B02_D_C0_seed42.ipynb`
- `KAGGLE_B03_D_C0_seed2024.ipynb`
- `KAGGLE_B04_D_P_seed3407.ipynb`
- `KAGGLE_B05_D_P_seed42.ipynb`
- `KAGGLE_B06_D_P_seed2024.ipynb`
- `KAGGLE_B07_R_C0_seed3407.ipynb`
- `KAGGLE_B08_R_C0_seed42.ipynb`
- `KAGGLE_B09_R_C0_seed2024.ipynb`
- `KAGGLE_B10_R_P_seed3407.ipynb`
- `KAGGLE_B11_R_P_seed42.ipynb`
- `KAGGLE_B12_R_P_seed2024.ipynb`

Each notebook runs exactly one `(architecture, condition, seed)`.

Common training/evaluation settings:

- Same data augmentation for C0 and P.
- Same CLAHE setting for C0 and P.
- Same focal loss setting for C0 and P.
- Same partial freezing policy for each architecture.
- Same calibration method.
- Same Youden threshold selection on validation/tuning only.
- Same TTA setting for final test evaluation.
- C0 only disables CBAM and mask-guided attention loss.
- P enables CBAM and mask-guided attention loss.

Required output directory per run:

```text
locked_outputs/classification_runs/{condition}/{architecture}/seed_{seed}/
```

Required files per run:

- `run_config.json`
- `environment.json`
- `dataset_audit.json`
- `train_history.csv`
- `calibration.json`
- `threshold.json`
- `test_metrics.json`
- `test_predictions.csv`
- `confusion_matrix.json`
- `best_checkpoint.pt`
- `run_status.json`

Required `test_predictions.csv` columns:

- `image_id`
- `image_path_or_archive_member`
- `image_sha256`
- `mask_id`
- `mask_sha256`
- `true_label`
- `logit_raw`
- `temperature`
- `logit_calibrated`
- `prob_pneumonia`
- `threshold`
- `pred_label`
- `is_correct`
- `architecture`
- `condition`
- `seed`
- `dataset_split`
- `run_id`

Required `test_metrics.json` fields:

- `n_test`
- `n_normal`
- `n_pneumonia`
- `auroc`
- `auprc`
- `accuracy`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `precision`
- `npv`
- `f1`
- `brier`
- `ece`
- `threshold`
- `temperature`
- `tn`
- `fp`
- `fn`
- `tp`

Pass/fail:

- `n_test` must equal 624.
- Predictions must have exactly 624 rows.
- No duplicated `image_id`.
- Threshold must come from validation/tuning only.
- No C0 run may have `use_cbam=True` or `use_mask_loss=True`.
- No P run may have `use_cbam=False` or `use_mask_loss=False`.
- Any failed XAI step must not invalidate classification, but must block XAI claims.

### Phase C: Locked 128-Case XAI Battery

Create one locked XAI manifest before running model-specific XAI.

Notebook:

- `KAGGLE_C00_make_xai_sample_manifest_128.ipynb`

Output:

- `locked_outputs/xai/xai_sample_manifest_128.csv`

Required columns:

- `image_id`
- `image_path_or_archive_member`
- `image_sha256`
- `mask_id`
- `mask_sha256`
- `true_label`
- `sampling_stratum`
- `sample_index`

Sampling rule:

- Use a deterministic fixed seed, preferably `3407`.
- Use 64 normal and 64 pneumonia cases if class counts allow.
- The same 128 cases must be used for all 12 XAI runs.
- Do not sample based on model correctness or confidence.

Create 12 XAI notebooks, one per trained checkpoint:

- `KAGGLE_C01_XAI_D_C0_seed3407.ipynb`
- `KAGGLE_C02_XAI_D_C0_seed42.ipynb`
- `KAGGLE_C03_XAI_D_C0_seed2024.ipynb`
- `KAGGLE_C04_XAI_D_P_seed3407.ipynb`
- `KAGGLE_C05_XAI_D_P_seed42.ipynb`
- `KAGGLE_C06_XAI_D_P_seed2024.ipynb`
- `KAGGLE_C07_XAI_R_C0_seed3407.ipynb`
- `KAGGLE_C08_XAI_R_C0_seed42.ipynb`
- `KAGGLE_C09_XAI_R_C0_seed2024.ipynb`
- `KAGGLE_C10_XAI_R_P_seed3407.ipynb`
- `KAGGLE_C11_XAI_R_P_seed42.ipynb`
- `KAGGLE_C12_XAI_R_P_seed2024.ipynb`

Required files per XAI run:

- `xai_config.json`
- `xai_sample_manifest_used.csv`
- `xai_per_case_metrics.csv`
- `xai_summary.json`
- `xai_figures_manifest.csv`
- `xai_run_status.json`

Required `xai_per_case_metrics.csv` columns:

- `image_id`
- `true_label`
- `pred_label`
- `prob_pneumonia`
- `is_correct`
- `architecture`
- `condition`
- `seed`
- `method`
- `gcam_lrr_plus`
- `gcam_lrr_abs`
- `guided_gcam_lrr_plus`
- `guided_gcam_lrr_abs`
- `lrr_chance`
- `gcam_lrr_enrichment`
- `guided_gcam_lrr_enrichment`
- `gcam_deletion_auc`
- `gcam_insertion_auc`
- `guided_gcam_deletion_auc`
- `guided_gcam_insertion_auc`
- `gcam_input_stability`
- `guided_gcam_input_stability`
- `gcam_parameter_randomization_corr`
- `guided_gcam_parameter_randomization_corr`
- `mask_area_ratio`
- `xai_target_label`
- `xai_target_definition`

Required XAI methods:

- Grad-CAM
- Guided Grad-CAM

Required XAI validity checks:

- LRR+ and LRR_abs against lung masks.
- Enrichment over lung area chance.
- Deletion AUC using the same target class explained by the saliency method.
- Insertion AUC using the same target class explained by the saliency method.
- Input stability under fixed low-amplitude noise.
- Parameter randomization sanity check using the classifier head at `model.backbone.fc` or `model.backbone.classifier`.

Pass/fail:

- Each XAI run must have exactly 128 rows.
- All 12 XAI runs must use identical `image_id` order from `xai_sample_manifest_128.csv`.
- `xai_target_definition` must be explicit: predicted class, with negative logit for class 0 and positive logit for class 1.
- Parameter randomization correlations near 1.0 across nearly all samples should be treated as failure unless independently explained.
- Missing deletion/insertion/stability/randomization columns block all XAI validity claims.

### Phase D: Aggregation, Statistics, and Paper Tables

Notebook:

- `KAGGLE_D01_aggregate_statistics_and_tables.ipynb`

Inputs:

- All Phase A, B, and C outputs.

Required output directory:

```text
locked_outputs/final_tables/
```

Required files:

- `artifact_manifest.csv`
- `classification_metrics_by_run.csv`
- `classification_metrics_summary_by_condition.csv`
- `prediction_level_master.csv`
- `xai_metrics_by_run.csv`
- `xai_metrics_summary_by_condition.csv`
- `baseline_reproduction_summary.csv`
- `paired_comparisons_classification.csv`
- `paired_comparisons_xai.csv`
- `paper_table_main_classification.csv`
- `paper_table_ablation.csv`
- `paper_table_xai.csv`
- `paper_table_reproduction.csv`
- `paper_claims_checklist.md`

Required classification statistics:

- Mean and SD across 3 seeds for each metric.
- 95% bootstrap CI for each metric per run.
- Paired bootstrap CI for P minus C0 within the same architecture and seed.
- McNemar test for thresholded predictions where comparing C0 vs P on the same test cases.
- AUROC/AUPRC paired bootstrap test for C0 vs P.
- Report effect sizes, not only p-values.

Required XAI statistics:

- Mean, SD, median, IQR across the 128 locked cases.
- 95% bootstrap CI for LRR+, enrichment, deletion AUC, insertion AUC, stability, and randomization correlation.
- Paired comparison between C0 and P within architecture, seed, and identical XAI sample.
- Separate Grad-CAM and Guided Grad-CAM summaries.

Required paper-ready tables:

1. Original reproduction table:
   - Original paper metric
   - Reproduced mean +/- SD
   - Difference from original
   - Pass/fail against +/-2 percentage-point tolerance

2. Main classification table:
   - D-C0, D-P, R-C0, R-P
   - AUROC, AUPRC, accuracy, sensitivity, specificity, F1, Brier, ECE
   - Mean +/- SD across seeds
   - 95% CI

3. Ablation/improvement table:
   - D-P minus D-C0
   - R-P minus R-C0
   - Delta AUROC, delta AUPRC, delta balanced accuracy, delta Brier, delta ECE
   - Paired CI and p-value

4. XAI localization table:
   - Grad-CAM and Guided Grad-CAM
   - LRR+, LRR_abs, chance, enrichment
   - Mean +/- SD and median [IQR]

5. XAI validity table:
   - Deletion AUC
   - Insertion AUC
   - Input stability
   - Parameter randomization correlation
   - Clear direction of better performance for each metric

## 3. Recommended Kaggle Parallelization

Batch 1:

- Run `KAGGLE_A01_original_reproduction.ipynb`.
- Run all 12 Phase B classification notebooks in parallel if Kaggle quota allows.

Batch 2:

- After all Phase B checkpoints and predictions are complete, run `KAGGLE_C00_make_xai_sample_manifest_128.ipynb`.
- Run all 12 Phase C XAI notebooks in parallel.

Batch 3:

- Run `KAGGLE_D01_aggregate_statistics_and_tables.ipynb`.
- Verify `paper_claims_checklist.md`.

Do not run Phase D until all required Phase B and C outputs exist.

## 4. Final Artifact Tree

Expected final Kaggle/Drive export:

```text
locked_outputs/
  reproduction/
  classification_runs/
    D-C0/DenseNet121/seed_3407/
    D-C0/DenseNet121/seed_42/
    D-C0/DenseNet121/seed_2024/
    D-P/DenseNet121/seed_3407/
    D-P/DenseNet121/seed_42/
    D-P/DenseNet121/seed_2024/
    R-C0/ResNet50/seed_3407/
    R-C0/ResNet50/seed_42/
    R-C0/ResNet50/seed_2024/
    R-P/ResNet50/seed_3407/
    R-P/ResNet50/seed_42/
    R-P/ResNet50/seed_2024/
  xai/
    xai_sample_manifest_128.csv
    D-C0/DenseNet121/seed_3407/
    ...
  final_tables/
  logs/
```

## 5. Paper Claim Rules

Allowed claims only after successful rerun:

- The model was evaluated on a sealed 624-case test set.
- Threshold and calibration were selected on validation/tuning only.
- Results are averaged over 3 seeds.
- Proposed method improves over C0 only if paired CI and effect size support it.
- XAI localization is evaluated on a locked 128-case test subset.
- Guided Grad-CAM validity is supported only if deletion/insertion/stability/randomization all exist for every condition and seed.

Not allowed unless specifically supported:

- Do not claim superiority over the original paper unless reproduction passes the +/-2 percentage-point tolerance or the limitation is explicitly disclosed.
- Do not report old LRP results as Guided Grad-CAM.
- Do not use Gate-5 tuning results as sealed test results.
- Do not use incomplete seed-3407-only XAI validity metrics as pooled 3-seed evidence.
- Do not call parameter randomization passed if correlations stay near 1.0 after the fixed randomization code.

## 6. Final Go/No-Go Checklist

Before updating the LNCS paper, verify:

- [ ] Original reproduction completed and pass/fail documented.
- [ ] 12 final classification runs completed.
- [ ] Every `test_predictions.csv` has exactly 624 rows.
- [ ] Every `test_metrics.json` has all required metrics.
- [ ] All thresholds come from validation/tuning only.
- [ ] `xai_sample_manifest_128.csv` exists and is identical across all XAI runs.
- [ ] 12 XAI runs completed with exactly 128 rows each.
- [ ] Guided Grad-CAM validity metrics exist for all seeds, not only seed 3407.
- [ ] Bootstrap CIs and paired comparisons generated.
- [ ] `artifact_manifest.csv` contains SHA-256 hashes for every output.
- [ ] `paper_claims_checklist.md` maps every paper number to an output file.
- [ ] No LRP claim remains in the paper unless a real LRP rerun is added.

## 7. Minimal Paper Update Mapping

Use only these generated files for the final LNCS manuscript:

- Abstract headline metrics: `final_tables/paper_table_main_classification.csv`
- Original reproduction paragraph/table: `final_tables/paper_table_reproduction.csv`
- Method ablation claim: `final_tables/paper_table_ablation.csv`
- XAI localization claim: `final_tables/paper_table_xai.csv`
- XAI validity claim: `final_tables/xai_metrics_summary_by_condition.csv`
- Statistical confidence text: `final_tables/paired_comparisons_classification.csv` and `final_tables/paired_comparisons_xai.csv`
- Provenance statement: `final_tables/artifact_manifest.csv`

If any required file is missing, the corresponding claim must be removed or moved to limitations.
