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
    notebook = {
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
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


A01_CELLS = [
    markdown_cell(
        """
# KAGGLE_A01 Original Paper Reproduction

Purpose: reproduce the reference-paper baseline before claiming improvement.

Add these Kaggle datasets to the notebook:

- `hintrngia/gate46-development-only-cxr`
- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`

This notebook writes outputs to `/kaggle/working/locked_outputs/reproduction/`.

Important: this is a best-effort source-like reproduction under the cleaned dataset protocol. If the reference paper's exact original train/test split cannot be reconstructed from the available Kaggle datasets, the paper must report this as a limitation instead of claiming exact reproduction.
"""
    ),
    code_cell(
        """
!pip install -q albumentations
"""
    ),
    code_cell(
        """
import os
import json
import time
import random
import hashlib
import zipfile
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torchvision.models import densenet121, DenseNet121_Weights, resnet50, ResNet50_Weights
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)

INPUT_DIR = Path('/kaggle/input')
WORK_DIR = Path('/kaggle/working')
OUT_DIR = WORK_DIR / 'locked_outputs' / 'reproduction'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_REQUIRED = [
    'gate46-development-only-cxr',
    'gate7-test-seal',
    'cxr-lung-masks',
]

REFERENCE_TARGETS = {
    'DenseNet121': {'accuracy': 0.891, 'auroc': 0.980},
    'ResNet50': {'accuracy': 0.844, 'auroc': 0.950},
}

SEEDS = [3407, 42, 2024]
ARCHITECTURES = ['DenseNet121', 'ResNet50']
TOLERANCE = 0.02

CONFIG = {
    'task': 'original_reference_reproduction_best_effort',
    'seeds': SEEDS,
    'architectures': ARCHITECTURES,
    'batch_size': 32,
    'num_workers': 2,
    'input_resolution': 224,
    'stage1_epochs': 5,
    'stage2_epochs': 20,
    'stage1_lr': 1e-3,
    'stage2_lr': 1e-4,
    'weight_decay': 1e-4,
    'patience': 5,
    'loss_type': 'BCEWithLogitsLoss',
    'use_cbam': False,
    'use_mask_loss': False,
    'apply_clahe': False,
    'use_tta': False,
    'threshold_source': 'validation_youden',
    'reference_targets': REFERENCE_TARGETS,
    'reproduction_tolerance_abs': TOLERANCE,
}

print(json.dumps(CONFIG, indent=2))
"""
    ),
    code_cell(
        """
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def environment_snapshot():
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'kaggle_input_children': sorted([p.name for p in INPUT_DIR.iterdir()]) if INPUT_DIR.exists() else [],
    }


def audit_required_datasets():
    found = {name: len(list(INPUT_DIR.rglob(name))) > 0 for name in DATASETS_REQUIRED}
    missing = [name for name, ok in found.items() if not ok]
    audit = {'required': DATASETS_REQUIRED, 'found': found, 'missing': missing}
    (OUT_DIR / 'dataset_presence_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if missing:
        raise FileNotFoundError(f'Missing required Kaggle datasets: {missing}')
    return audit


audit_required_datasets()
(OUT_DIR / 'original_reproduction_config.json').write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
(OUT_DIR / 'environment.json').write_text(json.dumps(environment_snapshot(), indent=2), encoding='utf-8')
print('Environment and dataset audit saved.')
"""
    ),
    code_cell(
        """
def extract_archives_once():
    dev_blob_matches = list(INPUT_DIR.rglob('GATE46_DEVELOPMENT.g46blob'))
    if not dev_blob_matches:
        raise FileNotFoundError('GATE46_DEVELOPMENT.g46blob not found in Kaggle input.')
    dev_root = WORK_DIR / 'gate46_development_extracted'
    if not dev_root.exists() or len(list(dev_root.rglob('*.jpg'))) + len(list(dev_root.rglob('*.jpeg'))) < 100:
        if dev_root.exists():
            import shutil
            shutil.rmtree(dev_root)
        dev_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dev_blob_matches[0], 'r') as zf:
            zf.extractall(dev_root)
    test_zip_matches = list(INPUT_DIR.rglob('opaque_test_images.zip'))
    test_root = WORK_DIR / 'opaque_test_images_extracted'
    if test_zip_matches:
        if not test_root.exists() or len(list(test_root.rglob('*.jpg'))) + len(list(test_root.rglob('*.jpeg'))) < 100:
            if test_root.exists():
                import shutil
                shutil.rmtree(test_root)
            test_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(test_zip_matches[0], 'r') as zf:
                zf.extractall(test_root)
    else:
        print('[AUDIT] opaque_test_images.zip not found; using images directly from /kaggle/input.')
    return dev_root, test_root


def load_manifests(dev_root):
    test_label_matches = list(INPUT_DIR.rglob('SEALED_test_label_key.csv'))
    if not test_label_matches:
        raise FileNotFoundError('SEALED_test_label_key.csv not found.')
    test_df = pd.read_csv(test_label_matches[0])
    dev_csvs = list(dev_root.rglob('*.csv')) + list(INPUT_DIR.rglob('*development*.csv'))
    dev_matches = [p for p in dev_csvs if 'clean' in p.name.lower() and 'development' in p.name.lower()]
    if not dev_matches:
        dev_matches = [p for p in dev_csvs if 'development' in p.name.lower()]
    if not dev_matches:
        raise FileNotFoundError('Development manifest not found.')
    dev_df = pd.read_csv(dev_matches[0])
    if 'final_split' in dev_df.columns:
        train_df = dev_df[dev_df['final_split'] == 'train'].reset_index(drop=True)
        val_df = dev_df[dev_df['final_split'] == 'tuning'].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        label_col = 'label_index' if 'label_index' in dev_df.columns else 'label'
        train_df, val_df = train_test_split(dev_df, test_size=0.15, random_state=3407, stratify=dev_df[label_col])
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
    audit = {
        'development_manifest': str(dev_matches[0]),
        'test_label_key': str(test_label_matches[0]),
        'n_train': int(len(train_df)),
        'n_val': int(len(val_df)),
        'n_test': int(len(test_df)),
    }
    (OUT_DIR / 'dataset_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(audit)
    return train_df, val_df, test_df


def build_image_map(*roots):
    image_map = {}
    valid_exts = {'.jpg', '.jpeg', '.png'}
    for root in roots:
        for path in Path(root).rglob('*'):
            if path.is_file() and path.suffix.lower() in valid_exts and 'mask' not in path.parent.name.lower():
                image_map[path.name] = path
    return image_map


def label_values(df):
    if 'label_index' in df.columns:
        return df['label_index'].astype(int).values
    if 'label' in df.columns:
        if df['label'].dtype == object:
            return (df['label'].str.strip().str.upper() == 'PNEUMONIA').astype(int).values
        return df['label'].astype(int).values
    raise KeyError('No label column found.')


def image_ids(df):
    for col in ['image_id', 'image', 'filename', 'archive_member', 'source_relative_path']:
        if col in df.columns:
            return df[col].apply(lambda x: Path(str(x)).name).values
    raise KeyError('No image id/path column found.')


dev_root, test_root = extract_archives_once()
train_df, val_df, test_df = load_manifests(dev_root)
image_map = build_image_map(INPUT_DIR, dev_root, test_root)
print(f'Image map size: {len(image_map)}')
"""
    ),
    code_cell(
        """
class CXRDataset(Dataset):
    def __init__(self, df, image_map, split):
        self.df = df.reset_index(drop=True)
        self.image_map = image_map
        self.split = split
        self.names = image_ids(self.df)
        self.labels = label_values(self.df)
        self.res = CONFIG['input_resolution']
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = self.image_map.get(name)
        if path is None:
            raise FileNotFoundError(f'Image not found in image_map: {name}')
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f'Could not read image: {path}')
        img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA)
        img = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        if self.split == 'train':
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label, name, str(path)


class BaselineModel(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        if arch == 'DenseNet121':
            self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            n = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(n, 1)
        elif arch == 'ResNet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            n = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(n, 1)
        else:
            raise ValueError(arch)

    def forward(self, x):
        return self.backbone(x).flatten()


def freeze_backbone_head_only(model):
    for p in model.parameters():
        p.requires_grad = False
    if model.arch == 'DenseNet121':
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.backbone.fc.parameters():
            p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def make_loaders(seed):
    train_ds = CXRDataset(train_df, image_map, 'train')
    val_ds = CXRDataset(val_df, image_map, 'val')
    test_ds = CXRDataset(test_df, image_map, 'test')
    labels = label_values(train_df)
    class_counts = np.bincount(labels, minlength=2)
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=CONFIG['num_workers'], generator=generator)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    return train_loader, val_loader, test_loader
"""
    ),
    code_cell(
        """
def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'n': int(len(y_true)),
        'n_normal': int((y_true == 0).sum()),
        'n_pneumonia': int((y_true == 1).sum()),
        'auroc': float(roc_auc_score(y_true, probs)),
        'auprc': float(average_precision_score(y_true, probs)),
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'sensitivity': float(recall_score(y_true, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) else 0.0,
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'brier': float(brier_score_loss(y_true, probs)),
        'threshold': float(threshold),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def find_youden_threshold(y_true, probs):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        score = sens + spec - 1
        if score > best_score:
            best_t, best_score = t, score
    return float(best_t)


def predict(model, loader, device):
    model.eval()
    logits_all, probs_all, labels_all, rows = [], [], [], []
    with torch.no_grad():
        for images, labels, names, paths in loader:
            images = images.to(device)
            logits = model(images).detach().cpu().numpy()
            probs = sigmoid_np(logits)
            logits_all.extend(logits.tolist())
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.numpy().astype(int).tolist())
            for name, path in zip(names, paths):
                rows.append({'image_id': name, 'image_path_or_archive_member': path})
    return np.array(logits_all), np.array(probs_all), np.array(labels_all), rows


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def fit_model(arch, seed):
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = make_loaders(seed)
    model = BaselineModel(arch).to(device)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    history = []
    best_auc, best_state, best_epoch, patience_count = -1, None, -1, 0

    freeze_backbone_head_only(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['stage1_lr'], weight_decay=CONFIG['weight_decay'])
    for epoch in range(CONFIG['stage1_epochs']):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        _, val_probs, val_labels, _ = predict(model, val_loader, device)
        metrics = compute_metrics(val_labels, val_probs, 0.5)
        history.append({'stage': 1, 'epoch': epoch + 1, 'train_loss': loss, **metrics})
        print(arch, seed, 'stage1', epoch + 1, metrics['auroc'])

    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['stage2_lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['stage2_epochs'])
    for epoch in range(CONFIG['stage2_epochs']):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        _, val_probs, val_labels, _ = predict(model, val_loader, device)
        metrics = compute_metrics(val_labels, val_probs, 0.5)
        history.append({'stage': 2, 'epoch': epoch + 1, 'train_loss': loss, **metrics})
        print(arch, seed, 'stage2', epoch + 1, metrics['auroc'])
        if metrics['auroc'] > best_auc + 1e-4:
            best_auc = metrics['auroc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= CONFIG['patience']:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_probs, val_labels, _ = predict(model, val_loader, device)
    threshold = find_youden_threshold(val_labels, val_probs)
    test_logits, test_probs, test_labels, test_rows = predict(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs, threshold)
    test_metrics.update({'architecture': arch, 'seed': seed, 'best_val_auroc': float(best_auc), 'best_epoch_stage2': int(best_epoch)})

    run_dir = OUT_DIR / f'{arch}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / 'train_history.csv', index=False)
    preds = pd.DataFrame(test_rows)
    preds['image_sha256'] = preds['image_path_or_archive_member'].apply(lambda p: sha256_file(Path(p)))
    preds['true_label'] = test_labels
    preds['logit_raw'] = test_logits
    preds['temperature'] = 1.0
    preds['logit_calibrated'] = test_logits
    preds['prob_pneumonia'] = test_probs
    preds['threshold'] = threshold
    preds['pred_label'] = (test_probs >= threshold).astype(int)
    preds['is_correct'] = preds['pred_label'].values == preds['true_label'].values
    preds['architecture'] = arch
    preds['condition'] = 'R0_original_reproduction'
    preds['seed'] = seed
    preds['dataset_split'] = 'sealed_test'
    preds['run_id'] = f'R0_{arch}_seed{seed}'
    preds.to_csv(run_dir / f'original_reproduction_predictions_seed{seed}.csv', index=False)
    (run_dir / 'test_metrics.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf-8')
    torch.save(model.state_dict(), run_dir / 'best_checkpoint.pt')
    return test_metrics
"""
    ),
    code_cell(
        """
all_metrics = []
start = time.time()
for arch in ARCHITECTURES:
    for seed in SEEDS:
        metrics = fit_model(arch, seed)
        all_metrics.append(metrics)
        pd.DataFrame(all_metrics).to_csv(OUT_DIR / 'original_reproduction_metrics_by_seed.csv', index=False)

metrics_df = pd.DataFrame(all_metrics)
summary_rows = []
for arch, group in metrics_df.groupby('architecture'):
    target = REFERENCE_TARGETS[arch]
    row = {'architecture': arch, 'n_runs': int(len(group))}
    for metric in ['accuracy', 'auroc']:
        mean_val = float(group[metric].mean())
        sd_val = float(group[metric].std(ddof=1))
        diff = mean_val - target[metric]
        row[f'{metric}_mean'] = mean_val
        row[f'{metric}_sd'] = sd_val
        row[f'{metric}_reference'] = target[metric]
        row[f'{metric}_diff'] = float(diff)
        row[f'{metric}_within_tolerance'] = bool(abs(diff) <= TOLERANCE)
    row['overall_pass'] = bool(row['accuracy_within_tolerance'] and row['auroc_within_tolerance'])
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / 'original_reproduction_summary.csv', index=False)
summary = {
    'runtime_seconds': time.time() - start,
    'summary': summary_rows,
    'paper_rule': 'If overall_pass is false, describe reproduction as best-effort and do not claim exact reproduced improvement.',
}
(OUT_DIR / 'original_reproduction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
summary_df
"""
    ),
]


C00_CELLS = [
    markdown_cell(
        """
# KAGGLE_C00 Locked XAI Sample Manifest 128

Purpose: create the single fixed 128-case manifest used by every Grad-CAM and Guided Grad-CAM XAI notebook.

Add these Kaggle datasets to the notebook:

- `hintrngia/gate46-development-only-cxr`
- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`

This notebook writes `/kaggle/working/locked_outputs/xai/xai_sample_manifest_128.csv`.
"""
    ),
    code_cell(
        """
import os
import json
import random
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path('/kaggle/input')
WORK_DIR = Path('/kaggle/working')
OUT_DIR = WORK_DIR / 'locked_outputs' / 'xai'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_REQUIRED = [
    'gate46-development-only-cxr',
    'gate7-test-seal',
    'cxr-lung-masks',
]

SAMPLE_SEED = 3407
N_PER_CLASS = 64
"""
    ),
    code_cell(
        """
def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_required_datasets():
    found = {name: len(list(INPUT_DIR.rglob(name))) > 0 for name in DATASETS_REQUIRED}
    missing = [name for name, ok in found.items() if not ok]
    audit = {'required': DATASETS_REQUIRED, 'found': found, 'missing': missing}
    (OUT_DIR / 'xai_manifest_dataset_presence_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if missing:
        raise FileNotFoundError(f'Missing required Kaggle datasets: {missing}')
    return audit


def extract_test_images_once():
    test_zip_matches = list(INPUT_DIR.rglob('opaque_test_images.zip'))
    test_root = WORK_DIR / 'opaque_test_images_extracted'
    if test_zip_matches:
        if not test_root.exists() or len(list(test_root.rglob('*.jpg'))) + len(list(test_root.rglob('*.jpeg'))) < 100:
            if test_root.exists():
                import shutil
                shutil.rmtree(test_root)
            test_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(test_zip_matches[0], 'r') as zf:
                zf.extractall(test_root)
    else:
        print('[AUDIT] opaque_test_images.zip not found; using images directly from /kaggle/input.')
    return test_root


def build_file_map(roots, include_masks=False):
    file_map = {}
    valid_exts = {'.jpg', '.jpeg', '.png'}
    for root in roots:
        for path in Path(root).rglob('*'):
            if not path.is_file() or path.suffix.lower() not in valid_exts:
                continue
            is_mask_path = 'mask' in path.name.lower() or 'mask' in path.parent.name.lower()
            if include_masks != is_mask_path:
                continue
            file_map[path.name] = path
    return file_map


def get_image_id(row):
    for col in ['image_id', 'image', 'filename', 'archive_member', 'source_relative_path']:
        if col in row.index and pd.notna(row[col]):
            return Path(str(row[col])).name
    raise KeyError('No image id/path column found in test label key.')


def get_label(row):
    if 'label_index' in row.index:
        return int(row['label_index'])
    if 'label' in row.index:
        value = row['label']
        if isinstance(value, str):
            return int(value.strip().upper() == 'PNEUMONIA')
        return int(value)
    raise KeyError('No label column found in test label key.')


def find_mask_for_image(image_name, mask_map):
    stem = Path(image_name).stem
    candidates = [
        f'{stem}.png',
        f'{stem}.jpg',
        f'{stem}.jpeg',
        f'{stem}_mask.png',
        f'{stem}_lung_mask.png',
        f'{stem}_lungs.png',
    ]
    for c in candidates:
        if c in mask_map:
            return mask_map[c]
    stem_lower = stem.lower()
    for name, path in mask_map.items():
        if Path(name).stem.lower().replace('_mask', '').replace('_lung', '').replace('_lungs', '') == stem_lower:
            return path
    return None


audit_required_datasets()
test_root = extract_test_images_once()
test_label_matches = list(INPUT_DIR.rglob('SEALED_test_label_key.csv'))
if not test_label_matches:
    raise FileNotFoundError('SEALED_test_label_key.csv not found.')
test_df = pd.read_csv(test_label_matches[0]).copy()
image_map = build_file_map([INPUT_DIR, test_root], include_masks=False)
mask_map = build_file_map([INPUT_DIR], include_masks=True)
print({'n_test_labels': len(test_df), 'n_images': len(image_map), 'n_masks': len(mask_map), 'label_key': str(test_label_matches[0])})
"""
    ),
    code_cell(
        """
rows = []
missing_images = []
missing_masks = []

for _, row in test_df.iterrows():
    image_id = get_image_id(row)
    label = get_label(row)
    image_path = image_map.get(image_id)
    mask_path = find_mask_for_image(image_id, mask_map)
    if image_path is None:
        missing_images.append(image_id)
        continue
    if mask_path is None:
        missing_masks.append(image_id)
        continue
    rows.append({
        'image_id': image_id,
        'image_path_or_archive_member': str(image_path),
        'image_sha256': sha256_file(image_path),
        'mask_id': mask_path.name,
        'mask_path': str(mask_path),
        'mask_sha256': sha256_file(mask_path),
        'true_label': int(label),
    })

eligible = pd.DataFrame(rows)
audit = {
    'n_test_labels': int(len(test_df)),
    'n_eligible_with_image_and_mask': int(len(eligible)),
    'n_missing_images': int(len(missing_images)),
    'n_missing_masks': int(len(missing_masks)),
    'missing_images_first_20': missing_images[:20],
    'missing_masks_first_20': missing_masks[:20],
}
(OUT_DIR / 'xai_manifest_eligibility_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))

if len(eligible) < 128:
    raise RuntimeError('Fewer than 128 test cases have both image and lung mask.')

counts = eligible['true_label'].value_counts().to_dict()
if counts.get(0, 0) < N_PER_CLASS or counts.get(1, 0) < N_PER_CLASS:
    raise RuntimeError(f'Not enough cases per class for balanced 64/64 sample: {counts}')

sampled = (
    eligible.groupby('true_label', group_keys=False)
    .apply(lambda g: g.sample(n=N_PER_CLASS, random_state=SAMPLE_SEED))
    .sort_values(['true_label', 'image_id'])
    .reset_index(drop=True)
)
sampled['sampling_stratum'] = sampled['true_label'].map({0: 'normal', 1: 'pneumonia'})
sampled['sample_index'] = np.arange(len(sampled))
sampled['sample_seed'] = SAMPLE_SEED
sampled['manifest_id'] = 'xai_sample_manifest_128_seed3407_balanced_64_64'

ordered_cols = [
    'manifest_id', 'sample_seed', 'sample_index', 'sampling_stratum',
    'image_id', 'image_path_or_archive_member', 'image_sha256',
    'mask_id', 'mask_path', 'mask_sha256', 'true_label'
]
sampled = sampled[ordered_cols]
out_path = OUT_DIR / 'xai_sample_manifest_128.csv'
sampled.to_csv(out_path, index=False)

manifest_summary = {
    'manifest_file': str(out_path),
    'n_rows': int(len(sampled)),
    'class_counts': sampled['true_label'].value_counts().sort_index().astype(int).to_dict(),
    'sha256': sha256_file(out_path),
    'rule': 'Balanced deterministic sample: 64 normal and 64 pneumonia, seed 3407; independent of model correctness/confidence.',
}
(OUT_DIR / 'xai_sample_manifest_128_summary.json').write_text(json.dumps(manifest_summary, indent=2), encoding='utf-8')
print(json.dumps(manifest_summary, indent=2))
sampled.head()
"""
    ),
]


write_notebook(OUT_DIR / "KAGGLE_A01_original_reproduction.ipynb", A01_CELLS)
write_notebook(OUT_DIR / "KAGGLE_C00_make_xai_sample_manifest_128.ipynb", C00_CELLS)
