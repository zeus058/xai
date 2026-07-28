import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "pipeline" / "src"
OUT_DIR = ROOT / "notebooks" / "kaggle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


MODULES = [
    "02_config.py",
    "04_dataset.py",
    "05_dataloader.py",
    "06_models.py",
    "07_training.py",
    "08_validation.py",
    "09_calibration.py",
]


RUNS = [
    {
        "filename": "KAGGLE_B01_D_C0_seed3407.ipynb",
        "run_id": "B01_D_C0_seed3407",
        "condition": "D-C0",
        "architecture": "DenseNet121",
        "seed": 3407,
        "use_cbam": False,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_B04_D_P_seed3407.ipynb",
        "run_id": "B04_D_P_seed3407",
        "condition": "D-P",
        "architecture": "DenseNet121",
        "seed": 3407,
        "use_cbam": True,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_B02_D_C0_seed42.ipynb",
        "run_id": "B02_D_C0_seed42",
        "condition": "D-C0",
        "architecture": "DenseNet121",
        "seed": 42,
        "use_cbam": False,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_B05_D_P_seed42.ipynb",
        "run_id": "B05_D_P_seed42",
        "condition": "D-P",
        "architecture": "DenseNet121",
        "seed": 42,
        "use_cbam": True,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_B03_D_C0_seed2024.ipynb",
        "run_id": "B03_D_C0_seed2024",
        "condition": "D-C0",
        "architecture": "DenseNet121",
        "seed": 2024,
        "use_cbam": False,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_B06_D_P_seed2024.ipynb",
        "run_id": "B06_D_P_seed2024",
        "condition": "D-P",
        "architecture": "DenseNet121",
        "seed": 2024,
        "use_cbam": True,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_B07_R_C0_seed3407.ipynb",
        "run_id": "B07_R_C0_seed3407",
        "condition": "R-C0",
        "architecture": "ResNet50",
        "seed": 3407,
        "use_cbam": False,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_B10_R_P_seed3407.ipynb",
        "run_id": "B10_R_P_seed3407",
        "condition": "R-P",
        "architecture": "ResNet50",
        "seed": 3407,
        "use_cbam": True,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_B08_R_C0_seed42.ipynb",
        "run_id": "B08_R_C0_seed42",
        "condition": "R-C0",
        "architecture": "ResNet50",
        "seed": 42,
        "use_cbam": False,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_B11_R_P_seed42.ipynb",
        "run_id": "B11_R_P_seed42",
        "condition": "R-P",
        "architecture": "ResNet50",
        "seed": 42,
        "use_cbam": True,
        "use_mask_loss": True,
    },
    {
        "filename": "KAGGLE_B09_R_C0_seed2024.ipynb",
        "run_id": "B09_R_C0_seed2024",
        "condition": "R-C0",
        "architecture": "ResNet50",
        "seed": 2024,
        "use_cbam": False,
        "use_mask_loss": False,
    },
    {
        "filename": "KAGGLE_B12_R_P_seed2024.ipynb",
        "run_id": "B12_R_P_seed2024",
        "condition": "R-P",
        "architecture": "ResNet50",
        "seed": 2024,
        "use_cbam": True,
        "use_mask_loss": True,
    },
]


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


def module_cells():
    cells = []
    for module in MODULES:
        cells.append(markdown_cell(f"## Embedded Module: {module}"))
        cells.append(code_cell((SRC_DIR / module).read_text(encoding="utf-8")))
    return cells


MAIN_TEMPLATE = r'''
import os
import json
import time
import random
import shutil
import hashlib
import zipfile
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix

RUN_ID = "{run_id}"
CONDITION = "{condition}"
ARCHITECTURE = "{architecture}"
SEED = {seed}
USE_CBAM = {use_cbam}
USE_MASK_LOSS = {use_mask_loss}

INPUT_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
RUNTIME_DIR = Path("/tmp/xai_locked_runtime") / RUN_ID
OUT_DIR = WORK_DIR / "locked_outputs" / "classification_runs" / CONDITION / ARCHITECTURE / f"seed_{{SEED}}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_REQUIRED = [
    "gate46-development-only-cxr",
    "gate7-test-seal",
    "cxr-lung-masks",
]


def setup_locked_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def sha256_file(path, chunk_size=1 << 20):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_required_datasets():
    found = {{name: len(list(INPUT_DIR.rglob(name))) > 0 for name in DATASETS_REQUIRED}}
    missing = [name for name, ok in found.items() if not ok]
    audit = {{"required": DATASETS_REQUIRED, "found": found, "missing": missing}}
    if missing:
        raise FileNotFoundError(f"Missing required Kaggle datasets: {{missing}}")
    return audit


def environment_snapshot():
    return {{
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "kaggle_input_children": sorted([p.name for p in INPUT_DIR.iterdir()]) if INPUT_DIR.exists() else [],
    }}


def extract_archives_to_tmp():
    dev_blob_matches = list(INPUT_DIR.rglob("GATE46_DEVELOPMENT.g46blob"))
    if not dev_blob_matches:
        raise FileNotFoundError("GATE46_DEVELOPMENT.g46blob not found in Kaggle input.")
    dev_root = RUNTIME_DIR / "gate46_development_extracted"
    existing_dev = len(list(dev_root.rglob("*.jpg"))) + len(list(dev_root.rglob("*.jpeg"))) if dev_root.exists() else 0
    if existing_dev < 100:
        if dev_root.exists():
            shutil.rmtree(dev_root)
        dev_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dev_blob_matches[0], "r") as zf:
            zf.extractall(dev_root)

    test_root = RUNTIME_DIR / "opaque_test_images_extracted"
    test_zip_matches = list(INPUT_DIR.rglob("opaque_test_images.zip"))
    if test_zip_matches:
        existing_test = len(list(test_root.rglob("*.jpg"))) + len(list(test_root.rglob("*.jpeg"))) if test_root.exists() else 0
        if existing_test < 100:
            if test_root.exists():
                shutil.rmtree(test_root)
            test_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(test_zip_matches[0], "r") as zf:
                zf.extractall(test_root)
    else:
        print("[AUDIT] opaque_test_images.zip not found; using images directly from /kaggle/input.")
    return dev_root, test_root


def label_values(df):
    if "label_index" in df.columns:
        return df["label_index"].astype(int).values
    if "label" in df.columns:
        if df["label"].dtype == object:
            return (df["label"].str.strip().str.upper() == "PNEUMONIA").astype(int).values
        return df["label"].astype(int).values
    raise KeyError("No label column found.")


def image_names_from_df(df):
    if "archive_member" in df.columns:
        return df["archive_member"].apply(lambda x: Path(str(x)).name).values
    if "source_relative_path" in df.columns:
        return df["source_relative_path"].apply(lambda x: Path(str(x)).name).values
    if "image" in df.columns:
        return df["image"].apply(lambda x: Path(str(x)).name).values
    if "image_id" in df.columns:
        return df["image_id"].apply(lambda x: Path(str(x)).name).values
    if "filename" in df.columns:
        return df["filename"].apply(lambda x: Path(str(x)).name).values
    raise KeyError("No image id/path column found.")


def load_manifests(dev_root):
    test_label_matches = list(INPUT_DIR.rglob("SEALED_test_label_key.csv"))
    if not test_label_matches:
        raise FileNotFoundError("SEALED_test_label_key.csv not found.")
    test_df = pd.read_csv(test_label_matches[0])

    dev_csvs = list(dev_root.rglob("*.csv")) + list(INPUT_DIR.rglob("*development*.csv"))
    dev_matches = [p for p in dev_csvs if "clean" in p.name.lower() and "development" in p.name.lower()]
    if not dev_matches:
        dev_matches = [p for p in dev_csvs if "development" in p.name.lower()]
    if not dev_matches:
        raise FileNotFoundError("Development manifest not found.")

    dev_df = pd.read_csv(dev_matches[0])
    if "final_split" in dev_df.columns:
        train_df = dev_df[dev_df["final_split"] == "train"].reset_index(drop=True)
        val_df = dev_df[dev_df["final_split"] == "tuning"].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        labels = label_values(dev_df)
        train_df, val_df = train_test_split(dev_df, test_size=0.15, random_state=3407, stratify=labels)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df, test_df, dev_matches[0], test_label_matches[0]


def build_image_map(*roots):
    image_map = {{}}
    valid_exts = {{".jpg", ".jpeg", ".png"}}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_exts and "mask" not in path.parent.name.lower():
                image_map[path.name] = path
    return image_map


def build_mask_map(*roots):
    mask_map = {{}}
    valid_exts = {{".jpg", ".jpeg", ".png"}}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in valid_exts:
                continue
            is_mask = "mask" in path.name.lower() or "mask" in path.parent.name.lower()
            if is_mask:
                mask_map[path.name] = path
    return mask_map


def find_mask_for_image(image_name, mask_map):
    stem = Path(image_name).stem
    candidates = [
        f"{{stem}}.png",
        f"{{stem}}.jpg",
        f"{{stem}}.jpeg",
        f"{{stem}}_mask.png",
        f"{{stem}}_lung_mask.png",
        f"{{stem}}_lungs.png",
    ]
    for candidate in candidates:
        if candidate in mask_map:
            return mask_map[candidate]
    stem_lower = stem.lower()
    for name, path in mask_map.items():
        normalized = Path(name).stem.lower().replace("_mask", "").replace("_lung", "").replace("_lungs", "")
        if normalized == stem_lower:
            return path
    return None


def make_canonical_mask_dir(all_image_names, mask_map):
    canonical_dir = RUNTIME_DIR / "lung_masks_canonical"
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    coverage = {{"total": len(all_image_names), "found": 0, "missing": []}}
    for image_name in sorted(set(all_image_names)):
        mask_path = find_mask_for_image(image_name, mask_map)
        if mask_path is None:
            coverage["missing"].append(image_name)
            continue
        target = canonical_dir / image_name
        try:
            os.symlink(mask_path, target)
        except Exception:
            shutil.copy2(mask_path, target)
        coverage["found"] += 1
    coverage["missing_first_20"] = coverage["missing"][:20]
    coverage["missing_count"] = len(coverage["missing"])
    return canonical_dir, coverage


def expected_image_coverage(df, image_map):
    names = image_names_from_df(df)
    missing = [name for name in names if name not in image_map]
    return {{"total": int(len(names)), "found": int(len(names) - len(missing)), "missing_count": int(len(missing)), "missing_first_20": missing[:20]}}


def compute_ece(labels, probs, n_bins=15):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def train_with_history(model, train_loader, val_loader, device, config, seed, class_weights=None):
    setup_locked_seed(seed)
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = get_criterion(config, class_weights)
    scaler = GradScaler(device="cuda", enabled=config.mixed_precision and torch.cuda.is_available())
    history = []

    optimizer_stage1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.stage1_lr, weight_decay=config.weight_decay)
    for epoch in range(config.stage1_epochs):
        current_mask_weight = config.mask_loss_weight * ((epoch + 1) / config.stage1_epochs) if config.use_mask_loss else 0.0
        train_loss, train_probs, train_labels = train_one_epoch(model, train_loader, optimizer_stage1, criterion, scaler, device, config, current_mask_weight)
        val_loss, val_probs, val_labels = validate(model, val_loader, criterion, device, config, current_mask_weight)
        val_metrics = calculate_metrics(val_labels.squeeze(), val_probs.squeeze(), threshold=0.5)
        row = {{"stage": 1, "epoch": epoch + 1, "train_loss": float(train_loss), "val_loss": float(val_loss), "mask_weight": float(current_mask_weight)}}
        row.update({{f"val_{{k}}": v for k, v in val_metrics.items() if k != "confusion_matrix"}})
        history.append(row)
        print(f"[{{RUN_ID}}] Stage 1 epoch {{epoch+1}}/{{config.stage1_epochs}} | train_loss={{train_loss:.4f}} | val_loss={{val_loss:.4f}} | val_auroc={{val_metrics['auroc']:.4f}}")

    model.unfreeze_all()
    optimizer_stage2 = AdamW(model.parameters(), lr=config.stage2_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer_stage2, T_max=config.stage2_epochs, eta_min=config.stage2_min_lr)
    best_val_loss = float("inf")
    best_weights = {{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.stage2_epochs):
        current_mask_weight = config.mask_loss_weight if config.use_mask_loss else 0.0
        train_loss, train_probs, train_labels = train_one_epoch(model, train_loader, optimizer_stage2, criterion, scaler, device, config, current_mask_weight)
        val_loss, val_probs, val_labels = validate(model, val_loader, criterion, device, config, current_mask_weight)
        scheduler.step()
        val_metrics = calculate_metrics(val_labels.squeeze(), val_probs.squeeze(), threshold=0.5)
        row = {{"stage": 2, "epoch": epoch + 1, "train_loss": float(train_loss), "val_loss": float(val_loss), "lr": float(scheduler.get_last_lr()[0]), "mask_weight": float(current_mask_weight)}}
        row.update({{f"val_{{k}}": v for k, v in val_metrics.items() if k != "confusion_matrix"}})
        history.append(row)
        print(f"[{{RUN_ID}}] Stage 2 epoch {{epoch+1}}/{{config.stage2_epochs}} | train_loss={{train_loss:.4f}} | val_loss={{val_loss:.4f}} | val_auroc={{val_metrics['auroc']:.4f}}")
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = float(val_loss)
            best_weights = {{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[{{RUN_ID}}] Early stopping at stage 2 epoch {{epoch+1}}.")
                break

    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history), {{"best_stage2_epoch": int(best_epoch), "best_val_loss": float(best_val_loss)}}


def predict_with_logits(model, dataloader, device, config, temperature):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels, names, masks in dataloader:
            images = images.to(device)
            labels_np = labels.cpu().numpy().astype(int)
            use_autocast = config.mixed_precision and torch.cuda.is_available()
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=use_autocast):
                logits = model(images, return_attention=False).detach().cpu().numpy().reshape(-1)
                probs = torch.sigmoid(torch.tensor(logits / temperature)).numpy().reshape(-1)
                prob_list = [probs]
                logit_list = [logits]
                if config.use_tta:
                    images_hf = torch.flip(images, dims=[3])
                    logits_hf = model(images_hf, return_attention=False).detach().cpu().numpy().reshape(-1)
                    probs_hf = torch.sigmoid(torch.tensor(logits_hf / temperature)).numpy().reshape(-1)
                    prob_list.append(probs_hf)
                    logit_list.append(logits_hf)
                avg_probs = np.mean(prob_list, axis=0)
                avg_logits = np.mean(logit_list, axis=0)
            for name, label, raw_logit, prob in zip(names, labels_np, avg_logits, avg_probs):
                rows.append({{"image_id": name, "true_label": int(label), "logit_raw": float(raw_logit), "temperature": float(temperature), "prob_pneumonia": float(prob)}})
    return pd.DataFrame(rows)


def normalize_metrics(metrics, labels, probs, threshold, temperature):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    cm = metrics["confusion_matrix"]
    normalized = {{
        "n_test": int(len(labels)),
        "n_normal": int((labels == 0).sum()),
        "n_pneumonia": int((labels == 1).sum()),
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "precision": float(metrics["precision"]),
        "npv": float(cm["tn"] / (cm["tn"] + cm["fn"])) if (cm["tn"] + cm["fn"]) else 0.0,
        "f1": float(metrics["f1"]),
        "brier": float(metrics["brier_score"]),
        "ece": compute_ece(labels, probs, n_bins=15),
        "log_loss": float(metrics["log_loss"]),
        "mcc": float(metrics["mcc"]),
        "threshold": float(threshold),
        "temperature": float(temperature),
        "tn": int(cm["tn"]),
        "fp": int(cm["fp"]),
        "fn": int(cm["fn"]),
        "tp": int(cm["tp"]),
    }}
    return normalized


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


run_started = time.time()
setup_locked_seed(SEED)
status = {{"run_id": RUN_ID, "status": "STARTED", "started_unix": run_started}}
save_json(OUT_DIR / "run_status.json", status)

try:
    presence_audit = audit_required_datasets()
    dev_root, test_root = extract_archives_to_tmp()
    train_df, val_df, test_df, dev_manifest_path, test_label_path = load_manifests(dev_root)
    image_map = build_image_map(INPUT_DIR, dev_root, test_root)
    mask_map = build_mask_map(INPUT_DIR, dev_root)

    all_names = list(image_names_from_df(train_df)) + list(image_names_from_df(val_df)) + list(image_names_from_df(test_df))
    canonical_mask_dir, mask_coverage = make_canonical_mask_dir(all_names, mask_map)

    dataset_audit = {{
        "dataset_presence": presence_audit,
        "development_manifest": str(dev_manifest_path),
        "test_label_key": str(test_label_path),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_label_counts": pd.Series(label_values(train_df)).value_counts().sort_index().astype(int).to_dict(),
        "val_label_counts": pd.Series(label_values(val_df)).value_counts().sort_index().astype(int).to_dict(),
        "test_label_counts": pd.Series(label_values(test_df)).value_counts().sort_index().astype(int).to_dict(),
        "image_map_size": int(len(image_map)),
        "train_image_coverage": expected_image_coverage(train_df, image_map),
        "val_image_coverage": expected_image_coverage(val_df, image_map),
        "test_image_coverage": expected_image_coverage(test_df, image_map),
        "mask_map_size": int(len(mask_map)),
        "canonical_mask_dir": str(canonical_mask_dir),
        "mask_coverage_all_splits": mask_coverage,
        "runtime_extraction_dir": str(RUNTIME_DIR),
    }}
    save_json(OUT_DIR / "dataset_audit.json", dataset_audit)

    if dataset_audit["test_image_coverage"]["found"] != 624:
        raise RuntimeError(f"Expected 624 test images, found {{dataset_audit['test_image_coverage']['found']}}")
    if USE_MASK_LOSS and mask_coverage["missing_count"] > 0:
        raise RuntimeError(f"P run requires masks for all train/val/test images; missing {{mask_coverage['missing_count']}}")

    config = PipelineConfig()
    config.seeds = [SEED]
    config.models_to_train = [ARCHITECTURE]
    config.batch_size = 32
    config.num_workers = 2
    config.input_resolution = 256
    config.skip_training_if_checkpoint_exists = False
    config.use_cbam = USE_CBAM
    config.use_mask_loss = USE_MASK_LOSS
    config.apply_clahe = True
    config.loss_type = "Focal"
    config.use_tta = True
    config.mixed_precision = True
    config.stage1_epochs = 5
    config.stage2_epochs = 25
    config.patience = 5

    run_config = {{
        "run_id": RUN_ID,
        "condition": CONDITION,
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "use_cbam": USE_CBAM,
        "use_mask_loss": USE_MASK_LOSS,
        "config": config.__dict__,
        "dataset_slugs": DATASETS_REQUIRED,
        "protocol": "locked_final_battery_B",
    }}
    save_json(OUT_DIR / "run_config.json", run_config)
    save_json(OUT_DIR / "environment.json", environment_snapshot())

    train_ds = ChestXRayDataset(train_df, image_map, config, split="train", mask_dir=canonical_mask_dir)
    val_ds = ChestXRayDataset(val_df, image_map, config, split="val", mask_dir=canonical_mask_dir)
    test_ds = ChestXRayDataset(test_df, image_map, config, split="test", mask_dir=canonical_mask_dir)
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, config, use_sampler=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{{RUN_ID}}] Device: {{device}}")

    model = ClassificationModel(
        ARCHITECTURE,
        freeze_percent=config.densenet_freeze_percent if ARCHITECTURE == "DenseNet121" else config.resnet_freeze_percent,
        use_cbam=config.use_cbam,
        use_aux_attention=(config.use_mask_loss and not config.use_cbam),
    )
    class_weights = get_class_weights(train_df)
    model, history_df, training_summary = train_with_history(model, train_loader, val_loader, device, config, SEED, class_weights=class_weights)
    history_df.to_csv(OUT_DIR / "train_history.csv", index=False)

    temperature = calibrate_model(model, val_loader, device)
    _, val_probs, val_labels = validate(model, val_loader, get_criterion(config, class_weights.to(device)), device, config)
    val_probs_flat = val_probs.squeeze()
    val_labels_flat = val_labels.squeeze()
    threshold = find_optimal_threshold(val_labels_flat, val_probs_flat, metric="youden")
    save_json(OUT_DIR / "calibration.json", {{"temperature": float(temperature), "method": "temperature_scaling_on_validation"}})
    save_json(OUT_DIR / "threshold.json", {{"threshold": float(threshold), "source": "validation_youden", "temperature": float(temperature)}})

    pred_df = predict_with_logits(model, test_loader, device, config, temperature)
    pred_df["threshold"] = float(threshold)
    pred_df["pred_label"] = (pred_df["prob_pneumonia"] >= threshold).astype(int)
    pred_df["is_correct"] = pred_df["pred_label"] == pred_df["true_label"]
    pred_df["architecture"] = ARCHITECTURE
    pred_df["condition"] = CONDITION
    pred_df["seed"] = SEED
    pred_df["dataset_split"] = "sealed_test"
    pred_df["run_id"] = RUN_ID
    pred_df["logit_calibrated"] = np.log(np.clip(pred_df["prob_pneumonia"].values, 1e-7, 1 - 1e-7) / np.clip(1 - pred_df["prob_pneumonia"].values, 1e-7, 1 - 1e-7))
    pred_df["image_path_or_archive_member"] = pred_df["image_id"].map(lambda x: str(image_map[x]))
    pred_df["image_sha256"] = pred_df["image_path_or_archive_member"].map(lambda p: sha256_file(Path(p)))
    pred_df["mask_id"] = pred_df["image_id"]
    pred_df["mask_path"] = pred_df["image_id"].map(lambda x: str(canonical_mask_dir / x) if (canonical_mask_dir / x).exists() else "")
    pred_df["mask_sha256"] = pred_df["mask_path"].map(lambda p: sha256_file(Path(p)) if p else "")
    ordered_prediction_cols = [
        "image_id", "image_path_or_archive_member", "image_sha256",
        "mask_id", "mask_path", "mask_sha256", "true_label",
        "logit_raw", "temperature", "logit_calibrated", "prob_pneumonia",
        "threshold", "pred_label", "is_correct", "architecture", "condition",
        "seed", "dataset_split", "run_id",
    ]
    pred_df = pred_df[ordered_prediction_cols]
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    metrics_raw = calculate_metrics(pred_df["true_label"].values, pred_df["prob_pneumonia"].values, threshold=threshold)
    test_metrics = normalize_metrics(metrics_raw, pred_df["true_label"].values, pred_df["prob_pneumonia"].values, threshold, temperature)
    test_metrics.update({{"run_id": RUN_ID, "architecture": ARCHITECTURE, "condition": CONDITION, "seed": SEED}})
    save_json(OUT_DIR / "test_metrics.json", test_metrics)
    save_json(OUT_DIR / "confusion_matrix.json", {{"tn": test_metrics["tn"], "fp": test_metrics["fp"], "fn": test_metrics["fn"], "tp": test_metrics["tp"]}})
    torch.save(model.state_dict(), OUT_DIR / "best_checkpoint.pt")

    status = {{
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "runtime_seconds": float(time.time() - run_started),
        "n_test_predictions": int(len(pred_df)),
        "test_auroc": test_metrics["auroc"],
        "test_accuracy": test_metrics["accuracy"],
    }}
    save_json(OUT_DIR / "run_status.json", status)
    print(json.dumps(status, indent=2))
    print(pd.DataFrame([test_metrics]))

except Exception as exc:
    import traceback
    status = {{
        "run_id": RUN_ID,
        "status": "FAILED",
        "runtime_seconds": float(time.time() - run_started),
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }}
    save_json(OUT_DIR / "run_status.json", status)
    print(json.dumps(status, indent=2))
    raise
'''


def make_notebook(run):
    title = f"# {run['filename'].replace('.ipynb', '')}"
    desc = f"""
{title}

Purpose: locked final classification run for `{run['condition']}`, `{run['architecture']}`, seed `{run['seed']}`.

Add these Kaggle datasets:

- `hintrngia/gate46-development-only-cxr`
- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`

Outputs are written to:

`/kaggle/working/locked_outputs/classification_runs/{run['condition']}/{run['architecture']}/seed_{run['seed']}/`

This notebook extracts temporary image data into `/tmp/xai_locked_runtime/` so Kaggle output does not include thousands of source images.
"""
    cells = [
        markdown_cell(desc),
        code_cell("!pip install -q albumentations grad-cam"),
    ]
    cells.extend(module_cells())
    cells.append(markdown_cell("## Locked B-Run Orchestrator"))
    cells.append(code_cell(MAIN_TEMPLATE.format(**run)))
    return cells


for run in RUNS:
    write_notebook(OUT_DIR / run["filename"], make_notebook(run))
