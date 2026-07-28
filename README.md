# Enhancing Explainability in Pediatric Pneumonia Detection via Feature Gating and Auxiliary Mask Supervision

[![Paper](https://img.shields.io/badge/Paper-main.pdf-blue.svg)](./main.pdf)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Locked-success.svg)](#-reproducing-the-experiments)

This repository contains the official reproducibility package for our research on mask-guided explainability for pediatric chest X-ray pneumonia classification. It is designed to meet the highest standards of scientific reproducibility, maintaining a clean audit trail from raw execution to final reported metrics.

## 📌 Repository Structure

Following a strict philosophy of auditability and clarity, this repository contains only the finalized, peer-review-ready artifacts:

- **`main.pdf`**: The finalized manuscript detailing our methodology, full experimental setup, and quantitative findings.
- **`pipeline/`**: The modular Python source code defining the core machine learning pipeline (environment setups, data loaders, model architectures, and training/evaluation routines).
- **`notebooks/`**: A complete, sequentially numbered suite of Jupyter Notebooks containing the exact execution code for all locked experiments.
- **`results/`**: A comprehensive cryptographic audit trail of our findings, including exact output metrics, statistical bootstrap tables, and input/output validation logs.

*(Note: Intermediate drafts, LaTeX source files, and temporary visualization scripts have been deliberately pruned from this repository to provide reviewers and researchers with an uncluttered view of the core scientific pipeline).*

## 🚀 Reproducing the Experiments

All experiments were strictly locked and executed in isolated environments to guarantee reproducibility. To replicate our findings, execute the Jupyter Notebooks in the `notebooks/` directory in the following sequence:

1. **Reference Reproduction**: `KAGGLE_A01_original_reproduction.ipynb`
2. **XAI Sample Generation**: `KAGGLE_C00_make_xai_sample_manifest_128.ipynb`
3. **Main Classification Battery**: `KAGGLE_B01` through `KAGGLE_B12` (12 locked runs across DenseNet121 and ResNet50 with matched controls).
4. **Statistical Validation**: `KAGGLE_D01_statistical_validation.ipynb`
5. **Explainable AI (XAI) Battery**: `KAGGLE_E01` through `KAGGLE_E04`
6. **XAI Aggregation & Statistics**: `KAGGLE_E05_XAI_STATISTICS_AND_FIGURES.ipynb`
7. **Component Ablation Battery**: `KAGGLE_F01` through `KAGGLE_F12` (Isolating the synergistic effects of the CBAM module and auxiliary mask loss).
8. **Ablation Statistics**: `KAGGLE_G01_component_ablation_statistics.ipynb`

## 📊 Audit Trail & Evidence

The `results/` directory is explicitly structured as a formal audit trail to prevent undisclosed tuning and data leakage:
- **`classification_statistics/`**: Contains `classification_summary_table.csv` and `paired_delta_bootstrap_ci.csv` (the exact numerical backing for the paper's results).
- **`xai_runs/`**: Contains paired XAI deltas, sign tests, and localized attribution metrics.
- **Audit Logs**: Cryptographic files such as `input_audit.json` and `xai_run_audit.json` securely map exact input datasets to the generated outputs, providing a transparent chain of custody for all scientific claims.

## ⚙️ Data & Checkpoint Constraints

Raw Chest X-Ray (CXR) data, automatically generated lung-mask files, and large `.pt`/`.pth` model checkpoints are intentionally excluded from Git due to size limitations. Please refer to the public Kaggle dataset slugs provided within the respective notebooks to retrieve the exact, hashed data partitions used in this study.

---
*If you find this reproducibility package or our methodology useful for your research, please consider citing our work.*
