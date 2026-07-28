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
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


NOTEBOOK_CODE = r'''
import json
import math
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

RUNS = [
    ("B01_D_C0_seed3407", "D-C0", "DenseNet121", 3407),
    ("B02_D_C0_seed42", "D-C0", "DenseNet121", 42),
    ("B03_D_C0_seed2024", "D-C0", "DenseNet121", 2024),
    ("B04_D_P_seed3407", "D-P", "DenseNet121", 3407),
    ("B05_D_P_seed42", "D-P", "DenseNet121", 42),
    ("B06_D_P_seed2024", "D-P", "DenseNet121", 2024),
    ("B07_R_C0_seed3407", "R-C0", "ResNet50", 3407),
    ("B08_R_C0_seed42", "R-C0", "ResNet50", 42),
    ("B09_R_C0_seed2024", "R-C0", "ResNet50", 2024),
    ("B10_R_P_seed3407", "R-P", "ResNet50", 3407),
    ("B11_R_P_seed42", "R-P", "ResNet50", 42),
    ("B12_R_P_seed2024", "R-P", "ResNet50", 2024),
]

METRICS = [
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
    "brier",
    "ece",
]

LOWER_IS_BETTER = {"brier", "ece"}
BOOTSTRAP_N = 5000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 3407

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
LOCAL_FINAL_BATTERY = Path("results/locked_final/classification_runs")

if KAGGLE_INPUT.exists():
    SEARCH_ROOTS = [KAGGLE_INPUT, KAGGLE_WORKING]
    OUT_DIR = KAGGLE_WORKING / "locked_outputs" / "statistics"
else:
    SEARCH_ROOTS = [LOCAL_FINAL_BATTERY]
    OUT_DIR = Path("results/locked_final/classification_statistics")

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUT_DIR}")
'''


FUNCTIONS_CODE = r'''
def compute_ece(labels, probs, n_bins=15):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def metric_bundle(labels, probs, threshold):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": compute_ece(labels, probs, n_bins=15),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def ci_bounds(values, level=0.95):
    values = np.asarray(values, dtype=float)
    alpha = (1.0 - level) / 2.0
    return (
        float(np.nanpercentile(values, alpha * 100.0)),
        float(np.nanpercentile(values, (1.0 - alpha) * 100.0)),
    )


def format_mean_sd_ci(mean, sd, lo, hi, digits=3):
    return f"{mean:.{digits}f} +/- {sd:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def format_delta_ci(mean, lo, hi, digits=3):
    sign = "+" if mean >= 0 else ""
    return f"{sign}{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def exact_mcnemar_p(c0_only, p_only):
    n = int(c0_only + p_only)
    if n == 0:
        return 1.0
    if binomtest is not None:
        return float(binomtest(min(c0_only, p_only), n, p=0.5, alternative="two-sided").pvalue)
    # Fallback exact two-sided binomial calculation.
    k = int(min(c0_only, p_only))
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail))


def stratified_bootstrap_indices(labels, rng):
    labels = np.asarray(labels).astype(int)
    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    sample0 = rng.choice(idx0, size=len(idx0), replace=True)
    sample1 = rng.choice(idx1, size=len(idx1), replace=True)
    return np.concatenate([sample0, sample1])
'''


LOAD_CODE = r'''
def candidate_prediction_files():
    files = []
    for root in SEARCH_ROOTS:
        if root.exists():
            files.extend(root.rglob("test_predictions.csv"))
    return sorted(set(files))


def load_expected_predictions():
    expected_run_ids = {run_id for run_id, _, _, _ in RUNS}
    by_run = {}
    source_paths = {}
    for path in candidate_prediction_files():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "run_id" not in df.columns:
            continue
        run_ids = sorted(df["run_id"].dropna().astype(str).unique())
        if len(run_ids) != 1:
            continue
        run_id = run_ids[0]
        if run_id in expected_run_ids and run_id not in by_run:
            by_run[run_id] = df
            source_paths[run_id] = str(path)

    missing = sorted(expected_run_ids - set(by_run))
    extra = sorted(set(by_run) - expected_run_ids)
    if missing:
        raise FileNotFoundError(
            "Missing required B-run prediction files: "
            + ", ".join(missing)
            + "\nOn Kaggle, add a dataset containing the 12 B01-B12 locked outputs, "
            + "or run this notebook locally from the repository root."
        )
    if extra:
        print(f"Ignoring unexpected runs: {extra}")
    return by_run, source_paths


predictions, source_paths = load_expected_predictions()
print(json.dumps(source_paths, indent=2))
'''


VALIDATE_CODE = r'''
validation_rows = []
reference_order = None
for run_id, condition, architecture, seed in RUNS:
    df = predictions[run_id].copy()
    required_cols = {"image_id", "true_label", "prob_pneumonia", "threshold", "pred_label", "is_correct", "condition", "architecture", "seed"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"{run_id} missing columns: {missing_cols}")
    if len(df) != 624:
        raise ValueError(f"{run_id} expected 624 predictions, found {len(df)}")
    if df["image_id"].nunique() != 624:
        raise ValueError(f"{run_id} expected 624 unique image IDs, found {df['image_id'].nunique()}")
    label_counts = df["true_label"].astype(int).value_counts().sort_index().to_dict()
    if label_counts != {0: 234, 1: 390}:
        raise ValueError(f"{run_id} unexpected label counts: {label_counts}")
    if sorted(df["condition"].astype(str).unique()) != [condition]:
        raise ValueError(f"{run_id} condition mismatch")
    if sorted(df["architecture"].astype(str).unique()) != [architecture]:
        raise ValueError(f"{run_id} architecture mismatch")
    if sorted(df["seed"].astype(int).unique()) != [seed]:
        raise ValueError(f"{run_id} seed mismatch")
    ordered_ids = tuple(df.sort_values("image_id")["image_id"].tolist())
    if reference_order is None:
        reference_order = ordered_ids
    elif ordered_ids != reference_order:
        raise ValueError(f"{run_id} is not image-aligned with the reference run")
    validation_rows.append({
        "run_id": run_id,
        "condition": condition,
        "architecture": architecture,
        "seed": seed,
        "n_predictions": len(df),
        "n_unique_images": df["image_id"].nunique(),
        "n_normal": int(label_counts[0]),
        "n_pneumonia": int(label_counts[1]),
        "threshold": float(df["threshold"].iloc[0]),
        "source_path": source_paths[run_id],
    })

input_validation = pd.DataFrame(validation_rows)
input_validation.to_csv(OUT_DIR / "input_validation.csv", index=False)
input_validation
'''


POINT_METRICS_CODE = r'''
point_rows = []
for run_id, condition, architecture, seed in RUNS:
    df = predictions[run_id].sort_values("image_id").reset_index(drop=True)
    threshold = float(df["threshold"].iloc[0])
    metrics = metric_bundle(df["true_label"], df["prob_pneumonia"], threshold)
    row = {
        "run_id": run_id,
        "condition": condition,
        "architecture": architecture,
        "seed": seed,
        "n_test": len(df),
        "threshold": threshold,
    }
    row.update(metrics)
    point_rows.append(row)

by_run_point = pd.DataFrame(point_rows)
by_run_point.to_csv(OUT_DIR / "by_run_point_metrics.csv", index=False)
by_run_point
'''


BOOTSTRAP_CODE = r'''
rng = np.random.default_rng(BOOTSTRAP_SEED)
bootstrap_rows = []

for run_id, condition, architecture, seed in RUNS:
    df = predictions[run_id].sort_values("image_id").reset_index(drop=True)
    labels = df["true_label"].to_numpy(dtype=int)
    probs = df["prob_pneumonia"].to_numpy(dtype=float)
    threshold = float(df["threshold"].iloc[0])
    samples = {metric: [] for metric in METRICS}
    for _ in range(BOOTSTRAP_N):
        idx = stratified_bootstrap_indices(labels, rng)
        sampled = metric_bundle(labels[idx], probs[idx], threshold)
        for metric in METRICS:
            samples[metric].append(sampled[metric])
    point = metric_bundle(labels, probs, threshold)
    for metric in METRICS:
        ci_low, ci_high = ci_bounds(samples[metric], level=CI_LEVEL)
        bootstrap_rows.append({
            "run_id": run_id,
            "condition": condition,
            "architecture": architecture,
            "seed": seed,
            "metric": metric,
            "point": point[metric],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_n": BOOTSTRAP_N,
            "ci_level": CI_LEVEL,
        })

by_run_bootstrap_ci = pd.DataFrame(bootstrap_rows)
by_run_bootstrap_ci.to_csv(OUT_DIR / "by_run_bootstrap_ci.csv", index=False)
by_run_bootstrap_ci.head(20)
'''


SUMMARY_CODE = r'''
summary_rows = []
for (condition, architecture), group in by_run_point.groupby(["condition", "architecture"], sort=False):
    for metric in METRICS:
        values = group[metric].astype(float).values
        ci_group = by_run_bootstrap_ci[
            (by_run_bootstrap_ci["condition"] == condition)
            & (by_run_bootstrap_ci["architecture"] == architecture)
            & (by_run_bootstrap_ci["metric"] == metric)
        ]
        # Conservative paper CI: summarize the run-level bootstrap centers by seed-level mean
        # and report seed SD separately. Paired delta CIs are computed in the next section.
        summary_rows.append({
            "condition": condition,
            "architecture": architecture,
            "metric": metric,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "min_seed_value": float(np.min(values)),
            "max_seed_value": float(np.max(values)),
            "n_seeds": int(len(values)),
        })

setting_summary = pd.DataFrame(summary_rows)
setting_summary.to_csv(OUT_DIR / "setting_mean_sd.csv", index=False)
setting_summary.head(20)
'''


PAIRED_CODE = r'''
paired_rows = []
mcnemar_rows = []

for architecture in ["DenseNet121", "ResNet50"]:
    if architecture == "DenseNet121":
        c0_condition, p_condition = "D-C0", "D-P"
    else:
        c0_condition, p_condition = "R-C0", "R-P"
    seeds = [3407, 42, 2024]

    per_seed_deltas = {metric: [] for metric in METRICS}
    bootstrap_deltas = {metric: [] for metric in METRICS}
    pooled_c0_only = 0
    pooled_p_only = 0

    aligned_pairs = []
    for seed in seeds:
        c0_run = [r for r in RUNS if r[1] == c0_condition and r[2] == architecture and r[3] == seed][0][0]
        p_run = [r for r in RUNS if r[1] == p_condition and r[2] == architecture and r[3] == seed][0][0]
        c0 = predictions[c0_run].sort_values("image_id").reset_index(drop=True)
        p = predictions[p_run].sort_values("image_id").reset_index(drop=True)
        if not c0["image_id"].equals(p["image_id"]):
            raise ValueError(f"Image order mismatch for {architecture}, seed {seed}")
        if not c0["true_label"].equals(p["true_label"]):
            raise ValueError(f"Label mismatch for {architecture}, seed {seed}")

        c0_threshold = float(c0["threshold"].iloc[0])
        p_threshold = float(p["threshold"].iloc[0])
        labels = c0["true_label"].to_numpy(dtype=int)
        c0_probs = c0["prob_pneumonia"].to_numpy(dtype=float)
        p_probs = p["prob_pneumonia"].to_numpy(dtype=float)
        c0_metrics = metric_bundle(labels, c0_probs, c0_threshold)
        p_metrics = metric_bundle(labels, p_probs, p_threshold)
        for metric in METRICS:
            per_seed_deltas[metric].append(p_metrics[metric] - c0_metrics[metric])

        c0_correct = c0["is_correct"].astype(bool).to_numpy()
        p_correct = p["is_correct"].astype(bool).to_numpy()
        c0_only = int(np.sum(c0_correct & ~p_correct))
        p_only = int(np.sum(~c0_correct & p_correct))
        pooled_c0_only += c0_only
        pooled_p_only += p_only
        mcnemar_rows.append({
            "architecture": architecture,
            "seed": seed,
            "comparison": f"{p_condition} minus {c0_condition}",
            "c0_only_correct": c0_only,
            "p_only_correct": p_only,
            "n_discordant": c0_only + p_only,
            "exact_mcnemar_p": exact_mcnemar_p(c0_only, p_only),
        })
        aligned_pairs.append((labels, c0_probs, p_probs, c0_threshold, p_threshold))

    for _ in range(BOOTSTRAP_N):
        seed_level_delta = {metric: [] for metric in METRICS}
        for labels, c0_probs, p_probs, c0_threshold, p_threshold in aligned_pairs:
            idx = stratified_bootstrap_indices(labels, rng)
            c0_boot = metric_bundle(labels[idx], c0_probs[idx], c0_threshold)
            p_boot = metric_bundle(labels[idx], p_probs[idx], p_threshold)
            for metric in METRICS:
                seed_level_delta[metric].append(p_boot[metric] - c0_boot[metric])
        for metric in METRICS:
            bootstrap_deltas[metric].append(np.mean(seed_level_delta[metric]))

    for metric in METRICS:
        point_delta = float(np.mean(per_seed_deltas[metric]))
        ci_low, ci_high = ci_bounds(bootstrap_deltas[metric], level=CI_LEVEL)
        better_direction = -1 if metric in LOWER_IS_BETTER else 1
        supports_improvement = (ci_low * better_direction) > 0 if better_direction == 1 else (ci_high < 0)
        paired_rows.append({
            "architecture": architecture,
            "comparison": f"{p_condition} minus {c0_condition}",
            "metric": metric,
            "mean_delta": point_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_n": BOOTSTRAP_N,
            "ci_level": CI_LEVEL,
            "n_seeds": len(seeds),
            "all_seed_deltas": json.dumps([float(x) for x in per_seed_deltas[metric]]),
            "supports_improvement_at_95ci": bool(supports_improvement),
        })

    mcnemar_rows.append({
        "architecture": architecture,
        "seed": "pooled_seed_runs",
        "comparison": f"{p_condition} minus {c0_condition}",
        "c0_only_correct": pooled_c0_only,
        "p_only_correct": pooled_p_only,
        "n_discordant": pooled_c0_only + pooled_p_only,
        "exact_mcnemar_p": exact_mcnemar_p(pooled_c0_only, pooled_p_only),
        "note": "Pooled over seed-runs; interpret with caution because the same test images are reused across seeds.",
    })

paired_delta_ci = pd.DataFrame(paired_rows)
mcnemar_tests = pd.DataFrame(mcnemar_rows)
paired_delta_ci.to_csv(OUT_DIR / "paired_delta_bootstrap_ci.csv", index=False)
mcnemar_tests.to_csv(OUT_DIR / "mcnemar_tests.csv", index=False)
display(paired_delta_ci)
display(mcnemar_tests)
'''


PAPER_TABLE_CODE = r'''
metric_label = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "npv": "NPV",
    "f1": "F1",
    "brier": "Brier score",
    "ece": "ECE",
}

paper_rows = []
for condition, architecture in [
    ("D-C0", "DenseNet121"),
    ("D-P", "DenseNet121"),
    ("R-C0", "ResNet50"),
    ("R-P", "ResNet50"),
]:
    row = {"condition": condition, "architecture": architecture}
    for metric in METRICS:
        summary_row = setting_summary[
            (setting_summary["condition"] == condition)
            & (setting_summary["architecture"] == architecture)
            & (setting_summary["metric"] == metric)
        ].iloc[0]
        row[metric_label[metric]] = f"{summary_row['mean']:.4f} +/- {summary_row['sd']:.4f}"
    paper_rows.append(row)

classification_summary_table_df = pd.DataFrame(paper_rows)
classification_summary_table_df.to_csv(OUT_DIR / "classification_summary_table.csv", index=False)

delta_rows = []
for architecture in ["DenseNet121", "ResNet50"]:
    row = {"architecture": architecture}
    for metric in METRICS:
        delta_row = paired_delta_ci[
            (paired_delta_ci["architecture"] == architecture)
            & (paired_delta_ci["metric"] == metric)
        ].iloc[0]
        row[metric_label[metric] + " delta"] = format_delta_ci(delta_row["mean_delta"], delta_row["ci_low"], delta_row["ci_high"], digits=4)
        row[metric_label[metric] + " supports 95CI"] = bool(delta_row["supports_improvement_at_95ci"])
    delta_rows.append(row)

classification_delta_table_df = pd.DataFrame(delta_rows)
classification_delta_table_df.to_csv(OUT_DIR / "classification_delta_table.csv", index=False)

display(classification_summary_table_df)
display(classification_delta_table_df)
'''


AUDIT_CODE = r'''
audit = {
    "notebook": "KAGGLE_D01_statistical_validation.ipynb",
    "purpose": "Locked final statistical validation for B01-B12 classification battery.",
    "bootstrap_n": BOOTSTRAP_N,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "ci_level": CI_LEVEL,
    "n_runs": len(RUNS),
    "n_test_per_run": 624,
    "input_validation_passed": True,
    "source_paths": source_paths,
    "outputs": sorted([p.name for p in OUT_DIR.iterdir() if p.is_file()]),
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
    "caveats": [
        "Pooled McNemar results reuse the same locked test images across seeds; interpret as seed-run evidence, not independent patient-level evidence.",
        "Bootstrap CIs are empirical and conditional on the locked 624-image test set.",
        "These statistics validate classification only; shortcut/localization claims still require the Guided Grad-CAM validity battery.",
    ],
}
(OUT_DIR / "statistics_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print(json.dumps(audit, indent=2))
print(f"Saved outputs to: {OUT_DIR}")
'''


cells = [
    markdown_cell(
        """
# KAGGLE_D01 Statistical Validation

## tl;dr

This notebook validates the locked B01-B12 classification battery after all 12 model runs have completed. It reads the saved prediction CSVs, verifies that all runs use the same locked 624-image test set, recomputes metrics, estimates bootstrap confidence intervals, and performs paired C0-vs-P statistical comparisons.

## Context & Methods

Add one Kaggle dataset containing the 12 B01-B12 locked output folders, or run this notebook locally from the repository root after all B outputs have been pulled into `results/locked_final/classification_runs`.

Expected inputs per run:

- `test_predictions.csv`
- image-level columns: `image_id`, `true_label`, `prob_pneumonia`, `threshold`, `pred_label`, `is_correct`

Expected outputs:

- `input_validation.csv`
- `by_run_point_metrics.csv`
- `by_run_bootstrap_ci.csv`
- `setting_mean_sd.csv`
- `paired_delta_bootstrap_ci.csv`
- `mcnemar_tests.csv`
- `classification_summary_table.csv`
- `classification_delta_table.csv`
- `statistics_audit.json`

### Key Assumptions

- The 624-case locked test set is the final held-out test population.
- The B-run thresholds were selected on validation data and are treated as fixed during test-set statistical validation.
- Bootstrap intervals are conditional on the locked test set.
- Pooled McNemar rows reuse the same test images across seeds and should be interpreted cautiously.
"""
    ),
    markdown_cell("## Data"),
    code_cell(NOTEBOOK_CODE),
    markdown_cell("### Helper Functions"),
    code_cell(FUNCTIONS_CODE),
    markdown_cell("### 1. Locate Inputs"),
    code_cell(LOAD_CODE),
    markdown_cell("### 2. Validate Inputs"),
    code_cell(VALIDATE_CODE),
    markdown_cell("## Results"),
    markdown_cell("### 3. Recompute Point Metrics"),
    code_cell(POINT_METRICS_CODE),
    markdown_cell("### 4. Bootstrap Run-Level Confidence Intervals"),
    code_cell(BOOTSTRAP_CODE),
    markdown_cell("### 5. Summarize Across Seeds"),
    code_cell(SUMMARY_CODE),
    markdown_cell("### 6. Paired C0-vs-P Comparisons"),
    code_cell(PAIRED_CODE),
    markdown_cell("### 7. Paper-Ready Tables"),
    code_cell(PAPER_TABLE_CODE),
    markdown_cell("## Takeaways"),
    code_cell(AUDIT_CODE),
]


write_notebook(OUT_DIR / "KAGGLE_D01_statistical_validation.ipynb", cells)
