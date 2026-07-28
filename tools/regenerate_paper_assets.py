"""Regenerate paper-ready tables, figures, and evidence manifests.

This CPU-only script intentionally starts from locked CSV/JSON artifacts under
``results/locked_final``. It does not require raw CXR images, masks, model
weights, Kaggle credentials, or GPU access.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "results" / "locked_final"
PAPER_FIGURES = ROOT / "paper" / "figures"

CLASSIFICATION_SUMMARY = LOCKED / "classification_statistics" / "classification_summary_table.csv"
CLASSIFICATION_DELTA = LOCKED / "classification_statistics" / "paired_delta_bootstrap_ci.csv"
XAI_SUMMARY = LOCKED / "xai_runs" / "aggregate" / "xai_summary_table.csv"
XAI_DELTA = LOCKED / "xai_runs" / "aggregate" / "xai_p_minus_c0_paired_delta_ci.csv"
QUALITATIVE_XAI = PAPER_FIGURES / "xai_qualitative_densenet_guided_cases.png"

MANIFEST_CSV = LOCKED / "paper_asset_manifest.csv"
MANIFEST_JSON = LOCKED / "paper_asset_manifest.json"
REPORT_MD = LOCKED / "paper_asset_regeneration_report.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path.relative_to(ROOT)}")


def plot_delta_ci(
    df: pd.DataFrame,
    rows: list[dict[str, str]],
    output_path: Path,
    xlabel: str,
) -> None:
    plot_rows = []
    for row in rows:
        subset = df[
            (df["architecture"] == row["architecture"])
            & (df["metric"] == row["metric"])
        ]
        if subset.empty:
            raise ValueError(f"Missing metric row: {row}")
        item = subset.iloc[0].to_dict()
        item["label"] = row["label"]
        item["invert_benefit"] = row.get("invert_benefit", "false") == "true"
        plot_rows.append(item)

    plot_df = pd.DataFrame(plot_rows)
    plot_df["mean_delta"] = plot_df["mean_delta"].astype(float)
    plot_df["ci_low"] = plot_df["ci_low"].astype(float)
    plot_df["ci_high"] = plot_df["ci_high"].astype(float)

    colors = ["#2f6f8f" if arch == "DenseNet121" else "#8a5a44" for arch in plot_df["architecture"]]
    y_positions = list(range(len(plot_df)))[::-1]

    fig_height = max(3.0, 0.34 * len(plot_df) + 0.85)
    fig, ax = plt.subplots(figsize=(7.1, fig_height))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y = y_positions[idx]
        ax.errorbar(
            row["mean_delta"],
            y,
            xerr=[
                [row["mean_delta"] - row["ci_low"]],
                [row["ci_high"] - row["mean_delta"]],
            ],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=1.6,
            capsize=3,
            markersize=5.5,
        )

    ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["label"], fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def regenerate_figures() -> list[Path]:
    classification = pd.read_csv(CLASSIFICATION_DELTA)
    xai = pd.read_csv(XAI_DELTA)

    classification_rows = [
        {"architecture": "DenseNet121", "metric": "auroc", "label": "DenseNet121 AUROC"},
        {"architecture": "DenseNet121", "metric": "balanced_accuracy", "label": "DenseNet121 balanced accuracy"},
        {"architecture": "DenseNet121", "metric": "specificity", "label": "DenseNet121 specificity"},
        {"architecture": "DenseNet121", "metric": "f1", "label": "DenseNet121 F1"},
        {"architecture": "DenseNet121", "metric": "brier", "label": "DenseNet121 Brier"},
        {"architecture": "DenseNet121", "metric": "ece", "label": "DenseNet121 ECE"},
        {"architecture": "ResNet50", "metric": "auroc", "label": "ResNet50 AUROC"},
        {"architecture": "ResNet50", "metric": "balanced_accuracy", "label": "ResNet50 balanced accuracy"},
        {"architecture": "ResNet50", "metric": "specificity", "label": "ResNet50 specificity"},
        {"architecture": "ResNet50", "metric": "f1", "label": "ResNet50 F1"},
        {"architecture": "ResNet50", "metric": "brier", "label": "ResNet50 Brier"},
        {"architecture": "ResNet50", "metric": "ece", "label": "ResNet50 ECE"},
    ]
    classification_plot = PAPER_FIGURES / "classification_p_minus_c0_delta_ci.png"
    plot_delta_ci(
        classification,
        classification_rows,
        classification_plot,
        "P minus C0 delta; negative is better for Brier and ECE",
    )

    xai_rows = [
        {"architecture": "DenseNet121", "metric": "gcam_lrr_plus", "label": "DenseNet121 Grad-CAM LRR+"},
        {
            "architecture": "DenseNet121",
            "metric": "guided_gcam_lrr_plus",
            "label": "DenseNet121 Guided Grad-CAM LRR+",
        },
        {"architecture": "DenseNet121", "metric": "gcam_pointing_inside_lung", "label": "DenseNet121 Grad-CAM pointing"},
        {
            "architecture": "DenseNet121",
            "metric": "guided_gcam_pointing_inside_lung",
            "label": "DenseNet121 Guided Grad-CAM pointing",
        },
        {"architecture": "ResNet50", "metric": "gcam_lrr_plus", "label": "ResNet50 Grad-CAM LRR+"},
        {
            "architecture": "ResNet50",
            "metric": "guided_gcam_lrr_plus",
            "label": "ResNet50 Guided Grad-CAM LRR+",
        },
        {"architecture": "ResNet50", "metric": "gcam_pointing_inside_lung", "label": "ResNet50 Grad-CAM pointing"},
        {
            "architecture": "ResNet50",
            "metric": "guided_gcam_pointing_inside_lung",
            "label": "ResNet50 Guided Grad-CAM pointing",
        },
    ]
    xai_plot = PAPER_FIGURES / "xai_p_minus_c0_localization_delta_ci.png"
    plot_delta_ci(
        xai,
        xai_rows,
        xai_plot,
        "P minus C0 delta; positive means stronger lung-field concentration",
    )

    return [classification_plot, xai_plot, QUALITATIVE_XAI]


def build_manifest(paths: list[Path]) -> list[dict[str, str | int]]:
    manifest_paths = [
        CLASSIFICATION_SUMMARY,
        CLASSIFICATION_DELTA,
        XAI_SUMMARY,
        XAI_DELTA,
        *paths,
        LOCKED / "retained_artifact_manifest.csv",
        LOCKED / "paper_evidence_audit.md",
    ]

    rows = []
    for path in manifest_paths:
        require_file(path)
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str | int]]) -> None:
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "entries": rows,
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(rows: list[dict[str, str | int]]) -> None:
    report = [
        "# Paper Asset Regeneration Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This report was generated from locked final artifacts only. It does not require raw images, masks, checkpoints, GPU access, or Kaggle credentials.",
        "",
        "## Regenerated Paper Figures",
        "",
        "- `paper/figures/classification_p_minus_c0_delta_ci.png`",
        "- `paper/figures/xai_p_minus_c0_localization_delta_ci.png`",
        "",
        "## Verified Evidence Inputs",
        "",
    ]
    for row in rows:
        report.append(f"- `{row['path']}`: {row['bytes']} bytes, sha256 `{row['sha256']}`")
    report.append("")
    REPORT_MD.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    for path in [CLASSIFICATION_SUMMARY, CLASSIFICATION_DELTA, XAI_SUMMARY, XAI_DELTA, QUALITATIVE_XAI]:
        require_file(path)
    regenerated = regenerate_figures()
    rows = build_manifest(regenerated)
    write_manifest(rows)
    write_report(rows)
    print(f"Wrote {MANIFEST_CSV.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST_JSON.relative_to(ROOT)}")
    print(f"Wrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
