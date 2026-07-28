# Results Layout

Use `locked_final/` as the source of truth for the manuscript.

## `locked_final/`

Validated outputs from the locked Kaggle rerun protocol:

- `reference_reproduction/`: original-paper reproduction attempt, per-seed
  predictions, metrics, logs, and audit summaries.
- `classification_runs/`: 12 locked C0/P classification runs across two
  backbones and three seeds. Checkpoint weights are intentionally excluded from
  this clean GitHub package; metrics, predictions, logs, configs, and audits are
  retained.
- `classification_statistics/`: classification summary tables, paired deltas,
  bootstrap confidence intervals, McNemar tests, and validation summary.
- `xai_manifest/`: deterministic 128-case balanced XAI sample manifest from C00.
- `xai_runs/`: E01-E04 XAI metric outputs plus E05 aggregate statistics.
  Per-case rendered PNG panels were removed from the clean package; manuscript
  figures are retained in `paper/figures/`.
- `paper_evidence_audit.md`: final source-to-paper consistency audit.

## Excluded Files

Large raw data, model weights, Kaggle dataset staging folders, archives,
development-only evidence, legacy local artifacts, and QA preview renders are
excluded from this cleaned repository workspace.
