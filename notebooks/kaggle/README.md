# Kaggle Notebooks: First Locked Rerun Batch

Run these two notebooks first:

1. `KAGGLE_A01_original_reproduction.ipynb`
2. `KAGGLE_C00_make_xai_sample_manifest_128.ipynb`

Then run the first locked classification pair:

3. `KAGGLE_B01_D_C0_seed3407.ipynb`
4. `KAGGLE_B04_D_P_seed3407.ipynb`

Then run the second DenseNet121 pair:

5. `KAGGLE_B02_D_C0_seed42.ipynb`
6. `KAGGLE_B05_D_P_seed42.ipynb`

Then run the third DenseNet121 pair:

7. `KAGGLE_B03_D_C0_seed2024.ipynb`
8. `KAGGLE_B06_D_P_seed2024.ipynb`

Then run the first ResNet50 pair:

9. `KAGGLE_B07_R_C0_seed3407.ipynb`
10. `KAGGLE_B10_R_P_seed3407.ipynb`

Then run the second ResNet50 pair:

11. `KAGGLE_B08_R_C0_seed42.ipynb`
12. `KAGGLE_B11_R_P_seed42.ipynb`

Then run the final ResNet50 pair:

13. `KAGGLE_B09_R_C0_seed2024.ipynb`
14. `KAGGLE_B12_R_P_seed2024.ipynb`

## Required Kaggle Datasets

Add all three datasets to each notebook:

- `hintrngia/gate46-development-only-cxr`
- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`

## Notebook Outputs

`KAGGLE_A01_original_reproduction.ipynb` writes:

- `/kaggle/working/locked_outputs/reproduction/original_reproduction_config.json`
- `/kaggle/working/locked_outputs/reproduction/environment.json`
- `/kaggle/working/locked_outputs/reproduction/dataset_audit.json`
- `/kaggle/working/locked_outputs/reproduction/original_reproduction_metrics_by_seed.csv`
- `/kaggle/working/locked_outputs/reproduction/original_reproduction_summary.csv`
- `/kaggle/working/locked_outputs/reproduction/original_reproduction_summary.json`
- Per-run folders with train history, test metrics, predictions, and checkpoints.

`KAGGLE_C00_make_xai_sample_manifest_128.ipynb` writes:

- `/kaggle/working/locked_outputs/xai/xai_sample_manifest_128.csv`
- `/kaggle/working/locked_outputs/xai/xai_sample_manifest_128_summary.json`
- `/kaggle/working/locked_outputs/xai/xai_manifest_dataset_presence_audit.json`
- `/kaggle/working/locked_outputs/xai/xai_manifest_eligibility_audit.json`

`KAGGLE_B01_D_C0_seed3407.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/D-C0/DenseNet121/seed_3407/`

`KAGGLE_B04_D_P_seed3407.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/D-P/DenseNet121/seed_3407/`

`KAGGLE_B02_D_C0_seed42.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/D-C0/DenseNet121/seed_42/`

`KAGGLE_B05_D_P_seed42.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/D-P/DenseNet121/seed_42/`

`KAGGLE_B03_D_C0_seed2024.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/D-C0/DenseNet121/seed_2024/`

`KAGGLE_B06_D_P_seed2024.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/D-P/DenseNet121/seed_2024/`

`KAGGLE_B07_R_C0_seed3407.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/R-C0/ResNet50/seed_3407/`

`KAGGLE_B10_R_P_seed3407.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/R-P/ResNet50/seed_3407/`

`KAGGLE_B08_R_C0_seed42.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/R-C0/ResNet50/seed_42/`

`KAGGLE_B11_R_P_seed42.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/R-P/ResNet50/seed_42/`

`KAGGLE_B09_R_C0_seed2024.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/R-C0/ResNet50/seed_2024/`

`KAGGLE_B12_R_P_seed2024.ipynb` writes:

- `/kaggle/working/locked_outputs/classification_runs/R-P/ResNet50/seed_2024/`

Each B-run directory should contain:

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

## DenseNet XAI Notebooks

Run these after C00 and all DenseNet B outputs are available:

15. `KAGGLE_E01_XAI_D_C0_DenseNet121.ipynb`
16. `KAGGLE_E02_XAI_D_P_DenseNet121.ipynb`

For E01/E02, add these Kaggle inputs:

- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`
- a dataset containing `xai_sample_manifest_128.csv` from C00
- a dataset containing B-run outputs with `best_checkpoint.pt` and `threshold.json`

The checkpoint dataset must preserve folder names that include:

- `D-C0/DenseNet121/seed_3407`
- `D-C0/DenseNet121/seed_42`
- `D-C0/DenseNet121/seed_2024`
- `D-P/DenseNet121/seed_3407`
- `D-P/DenseNet121/seed_42`
- `D-P/DenseNet121/seed_2024`

E01/E02 outputs are written to:

- `/kaggle/working/locked_outputs/xai_runs/E01_XAI_D_C0_DenseNet121/`
- `/kaggle/working/locked_outputs/xai_runs/E02_XAI_D_P_DenseNet121/`

## ResNet XAI Notebooks

Run these after C00 and all ResNet B outputs are available:

17. `KAGGLE_E03_XAI_R_C0_ResNet50.ipynb`
18. `KAGGLE_E04_XAI_R_P_ResNet50.ipynb`

For E03/E04, add these Kaggle inputs:

- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`
- `hintrngia/locked-xai-manifest-c00`
- `hintrngia/locked-resnet-xai-inputs`

The checkpoint dataset must preserve folder names that include:

- `R-C0/ResNet50/seed_3407`
- `R-C0/ResNet50/seed_42`
- `R-C0/ResNet50/seed_2024`
- `R-P/ResNet50/seed_3407`
- `R-P/ResNet50/seed_42`
- `R-P/ResNet50/seed_2024`

E03/E04 outputs are written to:

- `/kaggle/working/locked_outputs/xai_runs/E03_XAI_R_C0_ResNet50/`
- `/kaggle/working/locked_outputs/xai_runs/E04_XAI_R_P_ResNet50/`

## After Running

Download or save the full `/kaggle/working/locked_outputs/` directory. The next notebooks must reuse `xai_sample_manifest_128.csv` unchanged.
