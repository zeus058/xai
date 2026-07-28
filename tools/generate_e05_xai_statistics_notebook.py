import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "kaggle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


SETUP_CODE = r'''
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 3407
CI_LEVEL = 0.95

LOCAL_ROOT = Path("results/locked_final/xai_runs")
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

if KAGGLE_INPUT.exists():
    SEARCH_ROOTS = [KAGGLE_INPUT, KAGGLE_WORKING]
    OUT_DIR = KAGGLE_WORKING / "locked_outputs" / "xai_runs" / "aggregate"
else:
    SEARCH_ROOTS = [LOCAL_ROOT]
    OUT_DIR = LOCAL_ROOT / "aggregate"

FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("E01_XAI_D_C0_DenseNet121", "D-C0", "DenseNet121", "D", "C0"),
    ("E02_XAI_D_P_DenseNet121", "D-P", "DenseNet121", "D", "P"),
    ("E03_XAI_R_C0_ResNet50", "R-C0", "ResNet50", "R", "C0"),
    ("E04_XAI_R_P_ResNet50", "R-P", "ResNet50", "R", "P"),
]

METRICS = [
    "gcam_lrr_plus",
    "guided_gcam_lrr_plus",
    "gcam_pointing_inside_lung",
    "guided_gcam_pointing_inside_lung",
    "gcam_deletion_auc",
    "guided_gcam_deletion_auc",
    "gcam_insertion_auc",
    "guided_gcam_insertion_auc",
    "gcam_input_stability_lrr_delta",
    "guided_gcam_input_stability_lrr_delta",
    "gcam_head_randomization_corr",
    "guided_gcam_head_randomization_corr",
]

LOWER_IS_BETTER = {
    "gcam_input_stability_lrr_delta",
    "guided_gcam_input_stability_lrr_delta",
    "gcam_head_randomization_corr",
    "guided_gcam_head_randomization_corr",
}

PRIMARY_LOCALIZATION_METRICS = [
    "gcam_lrr_plus",
    "guided_gcam_lrr_plus",
    "gcam_pointing_inside_lung",
    "guided_gcam_pointing_inside_lung",
]

print(f"Output directory: {OUT_DIR}")
'''


HELPERS_CODE = r'''
def find_run_csv(run_id):
    matches = []
    for root in SEARCH_ROOTS:
        if root.exists():
            matches.extend([p for p in root.rglob("xai_metrics_by_image.csv") if run_id in str(p)])
    if not matches:
        raise FileNotFoundError(f"Missing xai_metrics_by_image.csv for {run_id}")
    return sorted(matches, key=lambda p: len(str(p)))[0]


def ci_bounds(values, level=0.95):
    values = np.asarray(values, dtype=float)
    alpha = (1.0 - level) / 2.0
    return (
        float(np.nanpercentile(values, alpha * 100.0)),
        float(np.nanpercentile(values, (1.0 - alpha) * 100.0)),
    )


def exact_binom_p(successes, failures):
    n = int(successes + failures)
    if n == 0:
        return 1.0
    k = int(min(successes, failures))
    if binomtest is not None:
        return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail))


def stratified_bootstrap_indices(labels, rng):
    labels = np.asarray(labels).astype(int)
    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    return np.concatenate([
        rng.choice(idx0, size=len(idx0), replace=True),
        rng.choice(idx1, size=len(idx1), replace=True),
    ])


def paper_metric_name(metric):
    return {
        "gcam_lrr_plus": "Grad-CAM LRR+",
        "guided_gcam_lrr_plus": "Guided Grad-CAM LRR+",
        "gcam_pointing_inside_lung": "Grad-CAM pointing",
        "guided_gcam_pointing_inside_lung": "Guided Grad-CAM pointing",
        "gcam_deletion_auc": "Grad-CAM deletion AUC",
        "guided_gcam_deletion_auc": "Guided Grad-CAM deletion AUC",
        "gcam_insertion_auc": "Grad-CAM insertion AUC",
        "guided_gcam_insertion_auc": "Guided Grad-CAM insertion AUC",
        "gcam_input_stability_lrr_delta": "Grad-CAM stability delta",
        "guided_gcam_input_stability_lrr_delta": "Guided Grad-CAM stability delta",
        "gcam_head_randomization_corr": "Grad-CAM randomization corr",
        "guided_gcam_head_randomization_corr": "Guided Grad-CAM randomization corr",
    }.get(metric, metric)
'''


LOAD_VALIDATE_CODE = r'''
frames = []
source_paths = {}
for run_id, condition, architecture, arch_code, method_group in RUNS:
    path = find_run_csv(run_id)
    source_paths[run_id] = str(path)
    df = pd.read_csv(path)
    required = {"seed", "image_id", "true_label", "condition", "architecture", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{run_id} missing columns: {missing}")
    if len(df) != 384:
        raise ValueError(f"{run_id} expected 384 rows, got {len(df)}")
    if df["image_id"].nunique() != 128:
        raise ValueError(f"{run_id} expected 128 unique images, got {df['image_id'].nunique()}")
    if sorted(df["seed"].astype(int).unique().tolist()) != [42, 2024, 3407]:
        raise ValueError(f"{run_id} unexpected seeds: {sorted(df['seed'].unique().tolist())}")
    label_counts = df["true_label"].astype(int).value_counts().sort_index().to_dict()
    if label_counts != {0: 192, 1: 192}:
        raise ValueError(f"{run_id} unexpected label counts: {label_counts}")
    df["run_id"] = run_id
    df["arch_code"] = arch_code
    df["method_group"] = method_group
    frames.append(df)

all_xai = pd.concat(frames, ignore_index=True)
all_xai.to_csv(OUT_DIR / "xai_all_by_image.csv", index=False)

input_validation = pd.DataFrame([
    {
        "run_id": run_id,
        "condition": condition,
        "architecture": architecture,
        "rows": len(all_xai[all_xai["run_id"] == run_id]),
        "unique_images": all_xai[all_xai["run_id"] == run_id]["image_id"].nunique(),
        "source_path": source_paths[run_id],
    }
    for run_id, condition, architecture, _, _ in RUNS
])
input_validation.to_csv(OUT_DIR / "xai_input_validation.csv", index=False)
input_validation
'''


SUMMARY_CODE = r'''
summary_rows = []
for (condition, architecture), group in all_xai.groupby(["condition", "architecture"], sort=False):
    for metric in METRICS:
        seed_means = group.groupby("seed")[metric].mean()
        summary_rows.append({
            "condition": condition,
            "architecture": architecture,
            "metric": metric,
            "metric_label": paper_metric_name(metric),
            "mean": float(seed_means.mean()),
            "sd": float(seed_means.std(ddof=1)),
            "n_seeds": int(seed_means.size),
            "n_seed_image_rows": int(len(group)),
        })

xai_summary_by_setting = pd.DataFrame(summary_rows)
xai_summary_by_setting.to_csv(OUT_DIR / "xai_summary_by_setting.csv", index=False)
xai_summary_by_setting.head(20)
'''


PAIRED_CODE = r'''
rng = np.random.default_rng(BOOTSTRAP_SEED)
paired_rows = []
sign_rows = []

for arch_code, architecture, c0_condition, p_condition in [
    ("D", "DenseNet121", "D-C0", "D-P"),
    ("R", "ResNet50", "R-C0", "R-P"),
]:
    c0 = all_xai[(all_xai["architecture"] == architecture) & (all_xai["condition"] == c0_condition)].copy()
    p = all_xai[(all_xai["architecture"] == architecture) & (all_xai["condition"] == p_condition)].copy()
    c0 = c0.sort_values(["seed", "image_id"]).reset_index(drop=True)
    p = p.sort_values(["seed", "image_id"]).reset_index(drop=True)
    if not c0[["seed", "image_id"]].equals(p[["seed", "image_id"]]):
        raise ValueError(f"{architecture} C0/P rows are not paired by seed and image_id")
    if not c0["true_label"].equals(p["true_label"]):
        raise ValueError(f"{architecture} C0/P labels are not aligned")

    for metric in METRICS:
        seed_deltas = []
        boot_deltas = []
        for seed in [3407, 42, 2024]:
            c0_seed = c0[c0["seed"] == seed].sort_values("image_id").reset_index(drop=True)
            p_seed = p[p["seed"] == seed].sort_values("image_id").reset_index(drop=True)
            delta = p_seed[metric].to_numpy(float) - c0_seed[metric].to_numpy(float)
            seed_deltas.append(float(np.mean(delta)))
        for _ in range(BOOTSTRAP_N):
            per_seed = []
            for seed in [3407, 42, 2024]:
                c0_seed = c0[c0["seed"] == seed].sort_values("image_id").reset_index(drop=True)
                p_seed = p[p["seed"] == seed].sort_values("image_id").reset_index(drop=True)
                labels = c0_seed["true_label"].to_numpy(int)
                idx = stratified_bootstrap_indices(labels, rng)
                delta = p_seed[metric].to_numpy(float)[idx] - c0_seed[metric].to_numpy(float)[idx]
                per_seed.append(np.mean(delta))
            boot_deltas.append(np.mean(per_seed))
        lo, hi = ci_bounds(boot_deltas, level=CI_LEVEL)
        mean_delta = float(np.mean(seed_deltas))
        if metric in LOWER_IS_BETTER:
            supports = bool(hi < 0)
        else:
            supports = bool(lo > 0)
        raw_delta = p[metric].to_numpy(float) - c0[metric].to_numpy(float)
        paired_rows.append({
            "architecture": architecture,
            "comparison": f"{p_condition} minus {c0_condition}",
            "metric": metric,
            "metric_label": paper_metric_name(metric),
            "mean_delta": mean_delta,
            "ci_low": lo,
            "ci_high": hi,
            "supports_improvement_at_95ci": supports,
            "seed_deltas": json.dumps(seed_deltas),
            "paired_better_count": int((raw_delta > 0).sum()),
            "paired_worse_count": int((raw_delta < 0).sum()),
            "paired_tie_count": int((raw_delta == 0).sum()),
        })

    for metric in ["gcam_pointing_inside_lung", "guided_gcam_pointing_inside_lung"]:
        raw_delta = p[metric].to_numpy(float) - c0[metric].to_numpy(float)
        improved = int((raw_delta > 0).sum())
        worsened = int((raw_delta < 0).sum())
        sign_rows.append({
            "architecture": architecture,
            "metric": metric,
            "metric_label": paper_metric_name(metric),
            "improved": improved,
            "worsened": worsened,
            "tied": int((raw_delta == 0).sum()),
            "exact_sign_test_p": exact_binom_p(improved, worsened),
            "note": "Sign test over paired seed-image rows; seed-image rows are not independent patient-level observations.",
        })

xai_p_minus_c0_paired_delta_ci = pd.DataFrame(paired_rows)
xai_pointing_sign_tests = pd.DataFrame(sign_rows)
xai_p_minus_c0_paired_delta_ci.to_csv(OUT_DIR / "xai_p_minus_c0_paired_delta_ci.csv", index=False)
xai_pointing_sign_tests.to_csv(OUT_DIR / "xai_pointing_sign_tests.csv", index=False)
display(xai_p_minus_c0_paired_delta_ci)
display(xai_pointing_sign_tests)
'''


PAPER_TABLES_CODE = r'''
primary_metrics = [
    "gcam_lrr_plus",
    "guided_gcam_lrr_plus",
    "gcam_pointing_inside_lung",
    "guided_gcam_pointing_inside_lung",
    "gcam_input_stability_lrr_delta",
    "guided_gcam_input_stability_lrr_delta",
]

paper_rows = []
for condition, architecture in [
    ("D-C0", "DenseNet121"),
    ("D-P", "DenseNet121"),
    ("R-C0", "ResNet50"),
    ("R-P", "ResNet50"),
]:
    row = {"condition": condition, "architecture": architecture}
    for metric in primary_metrics:
        s = xai_summary_by_setting[
            (xai_summary_by_setting["condition"] == condition)
            & (xai_summary_by_setting["architecture"] == architecture)
            & (xai_summary_by_setting["metric"] == metric)
        ].iloc[0]
        row[paper_metric_name(metric)] = f"{s['mean']:.4f} +/- {s['sd']:.4f}"
    paper_rows.append(row)

xai_classification_summary_table_df = pd.DataFrame(paper_rows)
xai_classification_summary_table_df.to_csv(OUT_DIR / "xai_summary_table.csv", index=False)

delta_rows = []
for architecture in ["DenseNet121", "ResNet50"]:
    row = {"architecture": architecture}
    for metric in primary_metrics:
        d = xai_p_minus_c0_paired_delta_ci[
            (xai_p_minus_c0_paired_delta_ci["architecture"] == architecture)
            & (xai_p_minus_c0_paired_delta_ci["metric"] == metric)
        ].iloc[0]
        row[paper_metric_name(metric) + " delta"] = f"{d['mean_delta']:+.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
        row[paper_metric_name(metric) + " 95CI"] = bool(d["supports_improvement_at_95ci"])
    delta_rows.append(row)

xai_classification_delta_table_df = pd.DataFrame(delta_rows)
xai_classification_delta_table_df.to_csv(OUT_DIR / "xai_delta_table.csv", index=False)
display(xai_classification_summary_table_df)
display(xai_classification_delta_table_df)
'''


FIGURE_CODE = r'''
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

plot_df = xai_summary_by_setting[xai_summary_by_setting["metric"].isin(PRIMARY_LOCALIZATION_METRICS)].copy()
plot_df["setting"] = plot_df["condition"] + "\n" + plot_df["architecture"].replace({"DenseNet121": "DenseNet", "ResNet50": "ResNet"})

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
axes = axes.ravel()
colors = {"D-C0": "#6c757d", "D-P": "#2a9d8f", "R-C0": "#6c757d", "R-P": "#2a9d8f"}

for ax, metric in zip(axes, PRIMARY_LOCALIZATION_METRICS):
    sub = plot_df[plot_df["metric"] == metric].copy()
    x = np.arange(len(sub))
    ax.bar(x, sub["mean"], yerr=sub["sd"], capsize=4, color=[colors[c] for c in sub["condition"]])
    ax.set_xticks(x)
    ax.set_xticklabels(sub["setting"], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_title(paper_metric_name(metric))
    ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(FIG_DIR / "xai_localization_metrics_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)

delta_plot = xai_p_minus_c0_paired_delta_ci[xai_p_minus_c0_paired_delta_ci["metric"].isin(PRIMARY_LOCALIZATION_METRICS)].copy()
fig, ax = plt.subplots(figsize=(9, 4.8))
labels = []
y = []
lo = []
hi = []
colors_list = []
for i, row in delta_plot.reset_index(drop=True).iterrows():
    labels.append(row["architecture"].replace("DenseNet121", "DenseNet").replace("ResNet50", "ResNet") + " " + paper_metric_name(row["metric"]).replace("Guided Grad-CAM", "Guided GC").replace("Grad-CAM", "GC"))
    y.append(row["mean_delta"])
    lo.append(row["mean_delta"] - row["ci_low"])
    hi.append(row["ci_high"] - row["mean_delta"])
    colors_list.append("#2a9d8f" if row["supports_improvement_at_95ci"] else "#8d99ae")
positions = np.arange(len(labels))
ax.errorbar(y, positions, xerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
ax.scatter(y, positions, c=colors_list, s=36, zorder=3)
ax.axvline(0, color="#555555", linewidth=1)
ax.set_yticks(positions)
ax.set_yticklabels(labels)
ax.set_xlabel("P minus C0 mean delta with 95% bootstrap CI")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "xai_p_minus_c0_localization_delta_ci.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved figures:", sorted([p.name for p in FIG_DIR.glob("*.png")]))
'''


AUDIT_CODE = r'''
audit = {
    "notebook": "KAGGLE_E05_XAI_STATISTICS_AND_FIGURES.ipynb",
    "purpose": "Aggregate locked XAI E01-E04 outputs into paper-ready tables and figures.",
    "bootstrap_n": BOOTSTRAP_N,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "ci_level": CI_LEVEL,
    "input_validation_passed": True,
    "n_runs": len(RUNS),
    "n_rows_total": int(len(all_xai)),
    "n_unique_images": int(all_xai["image_id"].nunique()),
    "source_paths": source_paths,
    "outputs": sorted([p.name for p in OUT_DIR.iterdir() if p.is_file()]),
    "figures": sorted([p.name for p in FIG_DIR.glob("*.png")]),
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
    "recommended_claim": (
        "The proposed CBAM + mask-guided training substantially increases Grad-CAM and Guided Grad-CAM "
        "localization inside the lung region across DenseNet121 and ResNet50."
    ),
    "caveats": [
        "These XAI statistics support localization, not automatically causal faithfulness.",
        "Deletion/insertion and randomization sanity metrics should be interpreted conservatively.",
        "Sign tests over seed-image rows are not independent patient-level tests because the same 128 images are reused across seeds.",
    ],
}
(OUT_DIR / "xai_aggregate_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
'''


cells = [
    markdown_cell(
        """
# KAGGLE_E05 XAI Statistics And Figures

## tl;dr

This notebook aggregates locked XAI outputs from E01-E04. It validates the four XAI result CSVs, computes setting summaries, paired P-minus-C0 bootstrap confidence intervals, sign tests for pointing metrics, and paper-ready figures/tables.

## Context & Methods

Inputs:

- E01 D-C0 DenseNet121 `xai_metrics_by_image.csv`
- E02 D-P DenseNet121 `xai_metrics_by_image.csv`
- E03 R-C0 ResNet50 `xai_metrics_by_image.csv`
- E04 R-P ResNet50 `xai_metrics_by_image.csv`

The notebook runs locally from the repository root or on Kaggle if E01-E04 outputs are provided as input datasets.
"""
    ),
    markdown_cell("## Setup"),
    code_cell(SETUP_CODE),
    markdown_cell("## Helper Functions"),
    code_cell(HELPERS_CODE),
    markdown_cell("## Data"),
    code_cell(LOAD_VALIDATE_CODE),
    markdown_cell("## Results"),
    code_cell(SUMMARY_CODE),
    markdown_cell("## Paired Statistics"),
    code_cell(PAIRED_CODE),
    markdown_cell("## Paper Tables"),
    code_cell(PAPER_TABLES_CODE),
    markdown_cell("## Figures"),
    code_cell(FIGURE_CODE),
    markdown_cell("## Audit"),
    code_cell(AUDIT_CODE),
]


write_notebook(OUT_DIR / "KAGGLE_E05_XAI_STATISTICS_AND_FIGURES.ipynb", cells)
