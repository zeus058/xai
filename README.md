# Pneumonia XAI Research Repository

This repository contains the reproducibility package for an LNCS-style research
study on mask-guided explainability for pediatric chest X-ray pneumonia
classification.

Canonical public repository:

- https://github.com/zeus058/xai
- Paper availability tag: `v1.0-paper`

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

## Regenerate Paper Evidence Assets

After the locked Kaggle outputs are present, regenerate the paper-ready figures
and SHA-256 evidence manifests with:

```bash
python tools/regenerate_paper_assets.py
```

This CPU-only command reads from `results/locked_final/` and writes:

- `paper/figures/classification_p_minus_c0_delta_ci.png`
- `paper/figures/xai_p_minus_c0_localization_delta_ci.png`
- `results/locked_final/paper_asset_manifest.csv`
- `results/locked_final/paper_asset_manifest.json`
- `results/locked_final/paper_asset_regeneration_report.md`

## Component Ablation Extension

The locked final paper evidence compares C0 against the full proposed
configuration. To support component-level claims, run the additional Kaggle
notebooks:

- `notebooks/kaggle/KAGGLE_F01` through `KAGGLE_F12` for CBAM-only and
  mask-loss-only matched runs.
- `notebooks/kaggle/KAGGLE_G01_component_ablation_statistics.ipynb` after B01-B12
  and F01-F12 outputs are available.

The ablation separates four components per backbone and seed: C0, CBAM-only,
mask-loss-only, and CBAM plus mask loss.

## GitHub Hygiene

Do not commit raw CXR data, lung-mask files, checkpoint weights, Kaggle staging
datasets, or temporary QA render images. They are intentionally ignored in
`.gitignore`.

For a public repository, cite Kaggle dataset slugs and include checksums/manifests
instead of uploading raw medical images or model weights directly to GitHub.
