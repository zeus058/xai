# A01 Original Reproduction Audit

Source kernel: `hintrngia/kagglea01`

Downloaded artifacts:

- 6 run folders: DenseNet121 and ResNet50 across seeds 3407, 42, 2024
- Per-run checkpoint, train history, sealed-test metrics, and 624-row prediction CSV
- Reproduction config, environment, dataset audits, summary CSV/JSON, and Kaggle log

Dataset audit:

- Train: 4411
- Validation/tuning: 779
- Sealed test: 624
- Sealed test labels: 234 normal, 390 pneumonia

## Reproduction Result

| Architecture | Accuracy mean | Accuracy reference | Accuracy diff | AUROC mean | AUROC reference | AUROC diff | Pass +/-2% |
|---|---:|---:|---:|---:|---:|---:|---|
| DenseNet121 | 0.812500 | 0.891 | -0.078500 | 0.954562 | 0.980 | -0.025438 | No |
| ResNet50 | 0.815171 | 0.844 | -0.028829 | 0.954854 | 0.950 | +0.004854 | No |

## Interpretation

This A01 run is a completed best-effort/source-like reproduction under the cleaned dataset protocol, but it does not reproduce the reference paper within the predefined +/-2 percentage-point tolerance.

The final manuscript should not claim exact reproduction of the original paper. It can state that a best-effort reproduction was run and did not match the original reported performance under the cleaned protocol. Any improvement claim should therefore be framed against the locked C0 baselines from the final battery, not as a direct reproduced improvement over the reference paper.
