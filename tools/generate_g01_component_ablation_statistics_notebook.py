"""Generate the component-ablation statistics notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "kaggle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


SETUP_CODE = r'''
import json
import math
import platform
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

RUNS = [
    ("B01_D_C0_seed3407", "D-C0", "DenseNet121", "C0", 3407),
    ("B02_D_C0_seed42", "D-C0", "DenseNet121", "C0", 42),
    ("B03_D_C0_seed2024", "D-C0", "DenseNet121", "C0", 2024),
    ("F01_D_CBAM_seed3407", "D-CBAM", "DenseNet121", "CBAM-only", 3407),
    ("F02_D_CBAM_seed42", "D-CBAM", "DenseNet121", "CBAM-only", 42),
    ("F03_D_CBAM_seed2024", "D-CBAM", "DenseNet121", "CBAM-only", 2024),
    ("F04_D_MASK_seed3407", "D-MASK", "DenseNet121", "mask-loss-only", 3407),
    ("F05_D_MASK_seed42", "D-MASK", "DenseNet121", "mask-loss-only", 42),
    ("F06_D_MASK_seed2024", "D-MASK", "DenseNet121", "mask-loss-only", 2024),
    ("B04_D_P_seed3407", "D-P", "DenseNet121", "CBAM+mask-loss", 3407),
    ("B05_D_P_seed42", "D-P", "DenseNet121", "CBAM+mask-loss", 42),
    ("B06_D_P_seed2024", "D-P", "DenseNet121", "CBAM+mask-loss", 2024),
    ("B07_R_C0_seed3407", "R-C0", "ResNet50", "C0", 3407),
    ("B08_R_C0_seed42", "R-C0", "ResNet50", "C0", 42),
    ("B09_R_C0_seed2024", "R-C0", "ResNet50", "C0", 2024),
    ("F07_R_CBAM_seed3407", "R-CBAM", "ResNet50", "CBAM-only", 3407),
    ("F08_R_CBAM_seed42", "R-CBAM", "ResNet50", "CBAM-only", 42),
    ("F09_R_CBAM_seed2024", "R-CBAM", "ResNet50", "CBAM-only", 2024),
    ("F10_R_MASK_seed3407", "R-MASK", "ResNet50", "mask-loss-only", 3407),
    ("F11_R_MASK_seed42", "R-MASK", "ResNet50", "mask-loss-only", 42),
    ("F12_R_MASK_seed2024", "R-MASK", "ResNet50", "mask-loss-only", 2024),
    ("B10_R_P_seed3407", "R-P", "ResNet50", "CBAM+mask-loss", 3407),
    ("B11_R_P_seed42", "R-P", "ResNet50", "CBAM+mask-loss", 42),
    ("B12_R_P_seed2024", "R-P", "ResNet50", "CBAM+mask-loss", 2024),
]

PAIRWISE_COMPARISONS = [
    ("CBAM-only minus C0", "CBAM-only", "C0"),
    ("mask-loss-only minus C0", "mask-loss-only", "C0"),
    ("CBAM+mask-loss minus C0", "CBAM+mask-loss", "C0"),
    ("CBAM+mask-loss minus CBAM-only", "CBAM+mask-loss", "CBAM-only"),
    ("CBAM+mask-loss minus mask-loss-only", "CBAM+mask-loss", "mask-loss-only"),
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
BOOTSTRAP_SEED = 3407
CI_LEVEL = 0.95

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
LOCAL_ROOT = Path("results/locked_final/classification_runs")

if KAGGLE_INPUT.exists():
    SEARCH_ROOTS = [KAGGLE_INPUT, KAGGLE_WORKING]
    OUT_DIR = KAGGLE_WORKING / "locked_outputs" / "component_ablation_statistics"
else:
    SEARCH_ROOTS = [LOCAL_ROOT]
    OUT_DIR = Path("results/locked_final/component_ablation_statistics")

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUT_DIR}")
'''


HELPERS_CODE = r'''
def compute_ece(labels, probs, n_bins=15):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs <= hi) if hi == 1.0 else (probs >= lo) & (probs < hi)
        if np.any(mask):
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


def stratified_bootstrap_indices(labels, rng):
    labels = np.asarray(labels).astype(int)
    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    return np.concatenate([
        rng.choice(idx0, size=len(idx0), replace=True),
        rng.choice(idx1, size=len(idx1), replace=True),
    ])


def ci_bounds(values, level=0.95):
    values = np.asarray(values, dtype=float)
    alpha = (1.0 - level) / 2.0
    return (
        float(np.nanpercentile(values, alpha * 100.0)),
        float(np.nanpercentile(values, (1.0 - alpha) * 100.0)),
    )


def exact_mcnemar_p(reference_only, candidate_only):
    n = int(reference_only + candidate_only)
    if n == 0:
        return 1.0
    k = int(min(reference_only, candidate_only))
    if binomtest is not None:
        return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail))


def supports_improvement(metric, ci_low, ci_high):
    if metric in LOWER_IS_BETTER:
        return bool(ci_high < 0.0)
    return bool(ci_low > 0.0)
'''


LOAD_CODE = r'''
def candidate_prediction_files():
    files = []
    for root in SEARCH_ROOTS:
        if root.exists():
            files.extend(root.rglob("test_predictions.csv"))
    return sorted(set(files))


def load_expected_predictions():
    expected = {run_id for run_id, _, _, _, _ in RUNS}
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
        if run_id in expected and run_id not in by_run:
            by_run[run_id] = df
            source_paths[run_id] = str(path)
    missing = sorted(expected - set(by_run))
    if missing:
        raise FileNotFoundError(
            "Missing component ablation prediction files: "
            + ", ".join(missing)
            + "\nAdd Kaggle datasets containing B01-B12 and F01-F12 locked outputs, "
            + "or run locally after pulling all outputs into results/locked_final/classification_runs."
        )
    return by_run, source_paths


predictions, source_paths = load_expected_predictions()
print(json.dumps(source_paths, indent=2))
'''


VALIDATION_CODE = r'''
rows = []
reference_ids = None
for run_id, condition, architecture, component, seed in RUNS:
    df = predictions[run_id].copy()
    required = {"image_id", "true_label", "prob_pneumonia", "threshold", "pred_label", "is_correct", "condition", "architecture", "seed"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"{run_id} missing columns: {missing_cols}")
    if len(df) != 624 or df["image_id"].nunique() != 624:
        raise ValueError(f"{run_id} expected 624 unique test predictions, found rows={len(df)}, unique={df['image_id'].nunique()}")
    label_counts = df["true_label"].astype(int).value_counts().sort_index().to_dict()
    if label_counts != {0: 234, 1: 390}:
        raise ValueError(f"{run_id} unexpected label counts: {label_counts}")
    if sorted(df["condition"].astype(str).unique()) != [condition]:
        raise ValueError(f"{run_id} condition mismatch")
    if sorted(df["architecture"].astype(str).unique()) != [architecture]:
        raise ValueError(f"{run_id} architecture mismatch")
    if sorted(df["seed"].astype(int).unique()) != [seed]:
        raise ValueError(f"{run_id} seed mismatch")
    sorted_ids = tuple(df.sort_values("image_id")["image_id"].tolist())
    if reference_ids is None:
        reference_ids = sorted_ids
    elif sorted_ids != reference_ids:
        raise ValueError(f"{run_id} test image IDs are not aligned with other runs")
    rows.append({
        "run_id": run_id,
        "condition": condition,
        "architecture": architecture,
        "component": component,
        "seed": seed,
        "n_predictions": len(df),
        "threshold": float(df["threshold"].iloc[0]),
        "source_path": source_paths[run_id],
    })

input_validation = pd.DataFrame(rows)
input_validation.to_csv(OUT_DIR / "component_ablation_input_validation.csv", index=False)
input_validation
'''


POINT_CODE = r'''
point_rows = []
for run_id, condition, architecture, component, seed in RUNS:
    df = predictions[run_id].sort_values("image_id").reset_index(drop=True)
    threshold = float(df["threshold"].iloc[0])
    metrics = metric_bundle(df["true_label"], df["prob_pneumonia"], threshold)
    row = {
        "run_id": run_id,
        "condition": condition,
        "architecture": architecture,
        "component": component,
        "seed": seed,
        "n_test": len(df),
        "threshold": threshold,
    }
    row.update(metrics)
    point_rows.append(row)

point_metrics = pd.DataFrame(point_rows)
point_metrics.to_csv(OUT_DIR / "component_ablation_by_run_metrics.csv", index=False)
point_metrics
'''


SUMMARY_CODE = r'''
summary_rows = []
for (architecture, component, condition), group in point_metrics.groupby(["architecture", "component", "condition"], sort=False):
    row = {
        "architecture": architecture,
        "component": component,
        "condition": condition,
        "n_seeds": int(group["seed"].nunique()),
    }
    for metric in METRICS:
        values = group[metric].astype(float).to_numpy()
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_sd"] = float(np.std(values, ddof=1))
        row[f"{metric}"] = f"{np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}"
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "component_ablation_summary_table.csv", index=False)
summary
'''


PAIRWISE_CODE = r'''
rng = np.random.default_rng(BOOTSTRAP_SEED)
paired_rows = []
mcnemar_rows = []

run_lookup = {(architecture, component, seed): run_id for run_id, _, architecture, component, seed in RUNS}

for architecture in ["DenseNet121", "ResNet50"]:
    for comparison_name, candidate_component, reference_component in PAIRWISE_COMPARISONS:
        per_seed_deltas = {metric: [] for metric in METRICS}
        bootstrap_deltas = {metric: [] for metric in METRICS}
        pair_data = []
        pooled_ref_only = 0
        pooled_candidate_only = 0

        for seed in [3407, 42, 2024]:
            candidate_run = run_lookup[(architecture, candidate_component, seed)]
            reference_run = run_lookup[(architecture, reference_component, seed)]
            cand = predictions[candidate_run].sort_values("image_id").reset_index(drop=True)
            ref = predictions[reference_run].sort_values("image_id").reset_index(drop=True)
            if not cand["image_id"].equals(ref["image_id"]):
                raise ValueError(f"Image mismatch: {architecture} {comparison_name} seed {seed}")
            if not cand["true_label"].equals(ref["true_label"]):
                raise ValueError(f"Label mismatch: {architecture} {comparison_name} seed {seed}")

            labels = cand["true_label"].to_numpy(dtype=int)
            cand_probs = cand["prob_pneumonia"].to_numpy(dtype=float)
            ref_probs = ref["prob_pneumonia"].to_numpy(dtype=float)
            cand_threshold = float(cand["threshold"].iloc[0])
            ref_threshold = float(ref["threshold"].iloc[0])
            cand_metrics = metric_bundle(labels, cand_probs, cand_threshold)
            ref_metrics = metric_bundle(labels, ref_probs, ref_threshold)

            for metric in METRICS:
                per_seed_deltas[metric].append(cand_metrics[metric] - ref_metrics[metric])

            cand_correct = cand["is_correct"].astype(bool).to_numpy()
            ref_correct = ref["is_correct"].astype(bool).to_numpy()
            ref_only = int(np.sum(ref_correct & ~cand_correct))
            cand_only = int(np.sum(~ref_correct & cand_correct))
            pooled_ref_only += ref_only
            pooled_candidate_only += cand_only
            mcnemar_rows.append({
                "architecture": architecture,
                "comparison": comparison_name,
                "candidate_component": candidate_component,
                "reference_component": reference_component,
                "seed": seed,
                "reference_only_correct": ref_only,
                "candidate_only_correct": cand_only,
                "n_discordant": ref_only + cand_only,
                "exact_mcnemar_p": exact_mcnemar_p(ref_only, cand_only),
            })
            pair_data.append((labels, cand_probs, ref_probs, cand_threshold, ref_threshold))

        for _ in range(BOOTSTRAP_N):
            seed_level_delta = {metric: [] for metric in METRICS}
            for labels, cand_probs, ref_probs, cand_threshold, ref_threshold in pair_data:
                idx = stratified_bootstrap_indices(labels, rng)
                cand_metrics = metric_bundle(labels[idx], cand_probs[idx], cand_threshold)
                ref_metrics = metric_bundle(labels[idx], ref_probs[idx], ref_threshold)
                for metric in METRICS:
                    seed_level_delta[metric].append(cand_metrics[metric] - ref_metrics[metric])
            for metric in METRICS:
                bootstrap_deltas[metric].append(np.mean(seed_level_delta[metric]))

        for metric in METRICS:
            point_delta = float(np.mean(per_seed_deltas[metric]))
            ci_low, ci_high = ci_bounds(bootstrap_deltas[metric], level=CI_LEVEL)
            paired_rows.append({
                "architecture": architecture,
                "comparison": comparison_name,
                "candidate_component": candidate_component,
                "reference_component": reference_component,
                "metric": metric,
                "mean_delta": point_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_n": BOOTSTRAP_N,
                "ci_level": CI_LEVEL,
                "n_seeds": 3,
                "all_seed_deltas": json.dumps([float(x) for x in per_seed_deltas[metric]]),
                "supports_improvement_at_95ci": supports_improvement(metric, ci_low, ci_high),
            })

        mcnemar_rows.append({
            "architecture": architecture,
            "comparison": comparison_name,
            "candidate_component": candidate_component,
            "reference_component": reference_component,
            "seed": "pooled_seed_runs",
            "reference_only_correct": pooled_ref_only,
            "candidate_only_correct": pooled_candidate_only,
            "n_discordant": pooled_ref_only + pooled_candidate_only,
            "exact_mcnemar_p": exact_mcnemar_p(pooled_ref_only, pooled_candidate_only),
            "note": "Pooled over seed-runs; interpret cautiously because the same test images are reused across seeds.",
        })

pairwise_delta_ci = pd.DataFrame(paired_rows)
mcnemar = pd.DataFrame(mcnemar_rows)
pairwise_delta_ci.to_csv(OUT_DIR / "component_ablation_pairwise_delta_ci.csv", index=False)
mcnemar.to_csv(OUT_DIR / "component_ablation_mcnemar_tests.csv", index=False)
display(pairwise_delta_ci)
display(mcnemar)
'''


AUDIT_CODE = r'''
audit = {
    "notebook": "KAGGLE_G01_component_ablation_statistics.ipynb",
    "purpose": "Matched component ablation over C0, CBAM-only, mask-loss-only, and CBAM+mask-loss.",
    "n_runs": len(RUNS),
    "n_test_per_run": 624,
    "n_seeds_per_component": 3,
    "bootstrap_n": BOOTSTRAP_N,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "ci_level": CI_LEVEL,
    "components": ["C0", "CBAM-only", "mask-loss-only", "CBAM+mask-loss"],
    "ablation_definition": {
        "C0": "No CBAM and no mask-guided auxiliary loss.",
        "CBAM-only": "CBAM feature gating enabled; no mask-guided auxiliary loss.",
        "mask-loss-only": "Auxiliary spatial-attention head receives mask loss; features are not gated by CBAM.",
        "CBAM+mask-loss": "CBAM feature gating enabled and CBAM spatial map receives mask-guided auxiliary loss.",
    },
    "outputs": sorted([p.name for p in OUT_DIR.iterdir() if p.is_file()]),
    "source_paths": source_paths,
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
    "paper_claim_rule": [
        "Only claim a component contribution when candidate-minus-reference paired 95% CI excludes zero in the beneficial direction.",
        "If CBAM-only or mask-loss-only is weaker than CBAM+mask-loss, describe the proposed method as a combined configuration unless interaction is explicitly supported.",
        "Classification ablation does not establish lesion localization or clinical utility.",
    ],
}
(OUT_DIR / "component_ablation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
'''


def main() -> None:
    cells = [
        markdown_cell(
            """
# KAGGLE_G01 Component Ablation Statistics

This notebook aggregates the full matched component-ablation battery:

- C0 baseline
- CBAM-only
- mask-loss-only
- CBAM plus mask loss

It requires the existing B01-B12 outputs plus the new F01-F12 outputs. It does
not train models. It validates that all 24 runs share the same 624-image sealed
test set, recomputes metrics, and estimates paired bootstrap confidence
intervals for component contrasts.
"""
        ),
        code_cell(SETUP_CODE),
        markdown_cell("## Helpers"),
        code_cell(HELPERS_CODE),
        markdown_cell("## Locate Inputs"),
        code_cell(LOAD_CODE),
        markdown_cell("## Validate Inputs"),
        code_cell(VALIDATION_CODE),
        markdown_cell("## Point Metrics"),
        code_cell(POINT_CODE),
        markdown_cell("## Component Summary"),
        code_cell(SUMMARY_CODE),
        markdown_cell("## Paired Component Contrasts"),
        code_cell(PAIRWISE_CODE),
        markdown_cell("## Audit"),
        code_cell(AUDIT_CODE),
    ]
    write_notebook(OUT_DIR / "KAGGLE_G01_component_ablation_statistics.ipynb", cells)


if __name__ == "__main__":
    main()
