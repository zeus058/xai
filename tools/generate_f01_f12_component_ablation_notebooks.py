"""Generate Kaggle notebooks for the component ablation battery.

The locked B01-B12 battery already contains C0 and CBAM+mask-loss (P) runs.
This generator creates the 12 missing matched runs:

- DenseNet121 CBAM-only and mask-loss-only across seeds 3407, 42, 2024
- ResNet50 CBAM-only and mask-loss-only across seeds 3407, 42, 2024
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "kaggle"
B_GENERATOR = ROOT / "tools" / "generate_b01_b12_classification_notebooks.py"

RUNS = [
    {
        "filename": "KAGGLE_F01_D_CBAM_seed3407.ipynb",
        "run_id": "F01_D_CBAM_seed3407",
        "condition": "D-CBAM",
        "architecture": "DenseNet121",
        "seed": 3407,
        "use_cbam": True,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_F02_D_CBAM_seed42.ipynb",
        "run_id": "F02_D_CBAM_seed42",
        "condition": "D-CBAM",
        "architecture": "DenseNet121",
        "seed": 42,
        "use_cbam": True,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_F03_D_CBAM_seed2024.ipynb",
        "run_id": "F03_D_CBAM_seed2024",
        "condition": "D-CBAM",
        "architecture": "DenseNet121",
        "seed": 2024,
        "use_cbam": True,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_F04_D_MASK_seed3407.ipynb",
        "run_id": "F04_D_MASK_seed3407",
        "condition": "D-MASK",
        "architecture": "DenseNet121",
        "seed": 3407,
        "use_cbam": False,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_F05_D_MASK_seed42.ipynb",
        "run_id": "F05_D_MASK_seed42",
        "condition": "D-MASK",
        "architecture": "DenseNet121",
        "seed": 42,
        "use_cbam": False,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_F06_D_MASK_seed2024.ipynb",
        "run_id": "F06_D_MASK_seed2024",
        "condition": "D-MASK",
        "architecture": "DenseNet121",
        "seed": 2024,
        "use_cbam": False,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_F07_R_CBAM_seed3407.ipynb",
        "run_id": "F07_R_CBAM_seed3407",
        "condition": "R-CBAM",
        "architecture": "ResNet50",
        "seed": 3407,
        "use_cbam": True,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_F08_R_CBAM_seed42.ipynb",
        "run_id": "F08_R_CBAM_seed42",
        "condition": "R-CBAM",
        "architecture": "ResNet50",
        "seed": 42,
        "use_cbam": True,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_F09_R_CBAM_seed2024.ipynb",
        "run_id": "F09_R_CBAM_seed2024",
        "condition": "R-CBAM",
        "architecture": "ResNet50",
        "seed": 2024,
        "use_cbam": True,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_F10_R_MASK_seed3407.ipynb",
        "run_id": "F10_R_MASK_seed3407",
        "condition": "R-MASK",
        "architecture": "ResNet50",
        "seed": 3407,
        "use_cbam": False,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_F11_R_MASK_seed42.ipynb",
        "run_id": "F11_R_MASK_seed42",
        "condition": "R-MASK",
        "architecture": "ResNet50",
        "seed": 42,
        "use_cbam": False,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_F12_R_MASK_seed2024.ipynb",
        "run_id": "F12_R_MASK_seed2024",
        "condition": "R-MASK",
        "architecture": "ResNet50",
        "seed": 2024,
        "use_cbam": False,
        "use_mask_loss": True,
    },
]


def load_b_generator_namespace() -> dict[str, object]:
    source = B_GENERATOR.read_text(encoding="utf-8")
    source = re.sub(r"\nfor run in RUNS:\n    write_notebook\(OUT_DIR / run\[\"filename\"\], make_notebook\(run\)\)\n?$", "\n", source)
    namespace: dict[str, object] = {"__file__": str(B_GENERATOR)}
    exec(compile(source, str(B_GENERATOR), "exec"), namespace)
    return namespace


def main() -> None:
    namespace = load_b_generator_namespace()
    write_notebook = namespace["write_notebook"]
    make_notebook = namespace["make_notebook"]

    for run in RUNS:
        cells = make_notebook(run)
        for cell in cells:
            if cell["cell_type"] != "markdown":
                continue
            text = "".join(cell["source"])
            if "Purpose: locked final classification run" in text:
                text = text.replace(
                    "Purpose: locked final classification run",
                    "Purpose: locked component-ablation classification run",
                )
                text += (
                    "\nAblation definition: `CBAM-only` uses CBAM feature gating without "
                    "mask loss; `mask-loss-only` uses an auxiliary spatial-attention head "
                    "for mask supervision without CBAM feature gating.\n"
                )
                cell["source"] = [line + "\n" for line in text.strip("\n").split("\n")]
                break
        write_notebook(OUT_DIR / run["filename"], cells)


if __name__ == "__main__":
    main()
