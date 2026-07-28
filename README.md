# Pneumonia XAI Research Repository

This repository contains the reproducibility package for an LNCS-style research
study on mask-guided explainability for pediatric chest X-ray pneumonia
classification.

The current manuscript is:

- `paper/main.tex`
- `paper/main.pdf`

The original reference paper used for motivation/reproduction checking is:

- `docs/reference_paper.pdf`

## What Is In This Repository

- `paper/`: LNCS manuscript source, references, paper figures, and compiled PDF.
- `notebooks/kaggle/`: locked Kaggle notebooks used for reproduction, final
  classification batteries, XAI batteries, and statistical aggregation.
- `tools/`: notebook and figure generation utilities used to build the locked
  experiments and paper-ready visualizations.
- `pipeline/`: reusable pipeline source modules. Raw input data are not kept in
  Git.
- `results/locked_final/`: paper-ready locked evidence tables, logs,
  audits, prediction CSVs, XAI metrics, and final evidence audit.

Large local-only files were removed from the repository workspace after the
final evidence audit. This includes raw datasets, Kaggle upload staging folders,
full archives, model checkpoints, development-only pulls, and temporary visual
QA renders.

## Locked Evidence Status

The paper should cite only the locked final rerun evidence.

Main evidence files:

- `results/locked_final/classification_statistics/classification_summary_table.csv`
- `results/locked_final/classification_statistics/paired_delta_bootstrap_ci.csv`
- `results/locked_final/xai_runs/aggregate/xai_summary_table.csv`
- `results/locked_final/xai_runs/aggregate/xai_p_minus_c0_paired_delta_ci.csv`
- `results/locked_final/paper_evidence_audit.md`

The locked protocol uses a 624-image sealed test set, three seeds per condition,
paired C0/P comparisons for DenseNet121 and ResNet50, bootstrap confidence
intervals, and a 128-case balanced XAI sample reused across XAI notebooks.

## Reproducibility Order

Run the Kaggle notebooks in this order:

1. `KAGGLE_A01_original_reproduction.ipynb`
2. `KAGGLE_C00_make_xai_sample_manifest_128.ipynb`
3. `KAGGLE_B01` through `KAGGLE_B12` for the 12 locked classification runs
4. `KAGGLE_D01_statistical_validation.ipynb`
5. `KAGGLE_E01` through `KAGGLE_E04` for XAI batteries
6. `KAGGLE_E05_XAI_STATISTICS_AND_FIGURES.ipynb`

Dataset requirements and exact notebook outputs are documented in
`notebooks/kaggle/README.md`.

## GitHub Hygiene

Do not commit raw CXR data, lung-mask files, checkpoint weights, Kaggle staging
datasets, or temporary QA render images. They are intentionally ignored in
`.gitignore`.

For a public repository, cite Kaggle dataset slugs and include checksums/manifests
instead of uploading raw medical images or model weights directly to GitHub.
