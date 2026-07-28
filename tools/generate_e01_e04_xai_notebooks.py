import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "kaggle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = [
    {
        "filename": "KAGGLE_E01_XAI_D_C0_DenseNet121.ipynb",
        "xai_id": "E01_XAI_D_C0_DenseNet121",
        "condition": "D-C0",
        "architecture": "DenseNet121",
        "use_cbam": False,
        "seeds": [3407, 42, 2024],
    },
    {
        "filename": "KAGGLE_E02_XAI_D_P_DenseNet121.ipynb",
        "xai_id": "E02_XAI_D_P_DenseNet121",
        "condition": "D-P",
        "architecture": "DenseNet121",
        "use_cbam": True,
        "seeds": [3407, 42, 2024],
    },
    {
        "filename": "KAGGLE_E03_XAI_R_C0_ResNet50.ipynb",
        "xai_id": "E03_XAI_R_C0_ResNet50",
        "condition": "R-C0",
        "architecture": "ResNet50",
        "use_cbam": False,
        "seeds": [3407, 42, 2024],
    },
    {
        "filename": "KAGGLE_E04_XAI_R_P_ResNet50.ipynb",
        "xai_id": "E04_XAI_R_P_ResNet50",
        "condition": "R-P",
        "architecture": "ResNet50",
        "use_cbam": True,
        "seeds": [3407, 42, 2024],
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
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


SETUP_CODE = r'''
!pip install -q grad-cam
'''


MAIN_CODE = r'''
import json
import os
import random
import time
import hashlib
import platform
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, resnet50

from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

RUN_CONFIG = json.loads(r"""__RUN_CONFIG_JSON__""")

XAI_ID = RUN_CONFIG["xai_id"]
CONDITION = RUN_CONFIG["condition"]
ARCHITECTURE = RUN_CONFIG["architecture"]
USE_CBAM = bool(RUN_CONFIG["use_cbam"])
SEEDS = [int(x) for x in RUN_CONFIG["seeds"]]

INPUT_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
LOCAL_FINAL_ROOT = Path("results/locked_final")

if INPUT_DIR.exists():
    ZIP_EXTRACT_ROOT = Path("/tmp/xai_locked_inputs")
    ZIP_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    for zip_path in INPUT_DIR.rglob("*.zip"):
        if zip_path.name in {"D-C0.zip", "D-P.zip", "R-C0.zip", "R-P.zip"}:
            extract_target = ZIP_EXTRACT_ROOT / zip_path.stem
            marker = extract_target / ".extract_complete"
            if not marker.exists():
                extract_target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_target)
                marker.write_text(str(zip_path), encoding="utf-8")
                print(f"[AUDIT] Extracted {zip_path} -> {extract_target}")
    SEARCH_ROOTS = [INPUT_DIR, ZIP_EXTRACT_ROOT, WORK_DIR]
    OUT_DIR = WORK_DIR / "locked_outputs" / "xai_runs" / XAI_ID
else:
    SEARCH_ROOTS = [LOCAL_FINAL_ROOT]
    OUT_DIR = LOCAL_FINAL_ROOT / "xai_runs" / XAI_ID

FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_RESOLUTION = 256
XAI_SAMPLE_SIZE = 128
DI_STEPS = 8
MAX_FIGURES_PER_SEED = 8
STABILITY_ENABLED = True
RANDOMIZATION_ENABLED = True

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print(json.dumps({
    "xai_id": XAI_ID,
    "condition": CONDITION,
    "architecture": ARCHITECTURE,
    "use_cbam": USE_CBAM,
    "seeds": SEEDS,
    "out_dir": str(OUT_DIR),
}, indent=2))
'''


MODEL_CODE = r'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.output_node = nn.Identity()

    def forward(self, x):
        x_out = x * self.ca(x)
        spatial_att = self.sa(x_out)
        x_out = x_out * spatial_att
        return self.output_node(x_out), spatial_att


class ClassificationModel(nn.Module):
    def __init__(self, architecture, freeze_percent=0.0, use_cbam=False):
        super().__init__()
        self.architecture = architecture
        self.use_cbam = use_cbam
        if architecture == "DenseNet121":
            self.backbone = densenet121(weights=None)
            in_features = self.backbone.classifier.in_features
            if use_cbam:
                self.cbam = CBAM(in_features)
            self.backbone.classifier = nn.Sequential(nn.Linear(in_features, 1))
        elif architecture == "ResNet50":
            self.backbone = resnet50(weights=None)
            in_features = self.backbone.fc.in_features
            if use_cbam:
                self.cbam = CBAM(in_features)
            self.backbone.fc = nn.Sequential(nn.Linear(in_features, 1))
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x, return_attention=False):
        if self.architecture == "DenseNet121":
            features = self.backbone.features(x)
            features = F.relu(features, inplace=False)
            spatial_att = None
            if self.use_cbam:
                features, spatial_att = self.cbam(features)
            out = F.adaptive_avg_pool2d(features, (1, 1))
            out = torch.flatten(out, 1)
            logits = self.backbone.classifier(out)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)
            spatial_att = None
            if self.use_cbam:
                features, spatial_att = self.cbam(features)
            out = self.backbone.avgpool(features)
            out = torch.flatten(out, 1)
            logits = self.backbone.fc(out)
        if return_attention:
            return logits, spatial_att
        return logits


class TwoClassWrapper(nn.Module):
    def __init__(self, binary_model):
        super().__init__()
        self.binary_model = binary_model

    def forward(self, x):
        logit = self.binary_model(x, return_attention=False)
        return torch.cat([-logit, logit], dim=1)
'''


HELPERS_CODE = r'''
def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_manifest():
    matches = []
    for root in SEARCH_ROOTS:
        if root.exists():
            matches.extend(root.rglob("xai_sample_manifest_128.csv"))
    if not matches:
        raise FileNotFoundError("xai_sample_manifest_128.csv not found. Add C00 output or results/locked_final/xai_manifest as Kaggle input.")
    return sorted(matches, key=lambda p: len(str(p)))[0]


def build_image_index():
    image_index = {}
    mask_index = {}
    valid_exts = {".jpg", ".jpeg", ".png"}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in valid_exts:
                continue
            lower = str(path).lower()
            if "mask" in lower or "lung" in lower:
                mask_index[path.name] = path
            else:
                image_index[path.name] = path
    return image_index, mask_index


def find_checkpoint_and_threshold(seed):
    checkpoint_candidates = []
    threshold_candidates = []
    seed_token = f"seed_{seed}"
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("best_checkpoint.pt"):
            parts = {part.lower() for part in path.parts}
            path_str = str(path).lower()
            if CONDITION.lower() in path_str and ARCHITECTURE.lower() in path_str and seed_token.lower() in parts:
                checkpoint_candidates.append(path)
        for path in root.rglob("threshold.json"):
            parts = {part.lower() for part in path.parts}
            path_str = str(path).lower()
            if CONDITION.lower() in path_str and ARCHITECTURE.lower() in path_str and seed_token.lower() in parts:
                threshold_candidates.append(path)
    if not checkpoint_candidates:
        raise FileNotFoundError(f"Missing checkpoint for {CONDITION}/{ARCHITECTURE}/seed_{seed}. Add B-run locked outputs as Kaggle input.")
    if not threshold_candidates:
        raise FileNotFoundError(f"Missing threshold.json for {CONDITION}/{ARCHITECTURE}/seed_{seed}.")
    return sorted(checkpoint_candidates, key=lambda p: len(str(p)))[0], sorted(threshold_candidates, key=lambda p: len(str(p)))[0]


def load_and_preprocess_image(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (INPUT_RESOLUTION, INPUT_RESOLUTION), interpolation=cv2.INTER_AREA)
    rgb_float = rgb.astype(np.float32) / 255.0
    normalized = (rgb_float - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, rgb_float


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"OpenCV could not read mask: {path}")
    mask = cv2.resize(mask, (INPUT_RESOLUTION, INPUT_RESOLUTION), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


def compute_lrr(heatmap, lung_mask):
    if heatmap.shape != lung_mask.shape:
        lung_mask = cv2.resize(lung_mask, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_NEAREST)
    lung_mask = (lung_mask > 0).astype(np.float32)
    positive = np.maximum(heatmap, 0)
    total = float(positive.sum())
    inside = float((positive * lung_mask).sum())
    chance = float(lung_mask.mean())
    lrr_plus = inside / total if total > 0 else 0.0
    enrichment = lrr_plus / chance if chance > 0 else 0.0
    abs_total = float(np.abs(heatmap).sum())
    lrr_abs = float((np.abs(heatmap) * lung_mask).sum() / abs_total) if abs_total > 0 else 0.0
    return lrr_plus, lrr_abs, chance, enrichment


def pointing_inside(heatmap, lung_mask):
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return int(lung_mask[y, x] > 0), int(x), int(y)


def target_probability(logit, target_label):
    signed = logit if int(target_label) == 1 else -logit
    return float(1.0 / (1.0 + np.exp(-signed)))


def deletion_insertion_auc(model, image, heatmap, target_label, steps=8):
    heatmap = cv2.resize(heatmap, (image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
    flat = heatmap.reshape(-1)
    order = np.argsort(flat)[::-1]
    n_pixels = len(order)
    step_size = max(1, n_pixels // steps)
    blurred = torch.from_numpy(cv2.GaussianBlur(image[0].detach().cpu().numpy().transpose(1, 2, 0), (21, 21), 5.0).transpose(2, 0, 1)).float().unsqueeze(0).to(image.device)
    deletion = image.clone().detach()
    insertion = blurred.clone().detach()
    del_probs = []
    ins_probs = []
    with torch.no_grad():
        for i in range(0, steps + 1):
            if i > 0:
                selected = order[(i - 1) * step_size:min(i * step_size, n_pixels)]
                rows = selected // heatmap.shape[1]
                cols = selected % heatmap.shape[1]
                deletion[0, :, rows, cols] = blurred[0, :, rows, cols]
                insertion[0, :, rows, cols] = image[0, :, rows, cols]
            del_logit = float(model(deletion, return_attention=False).item())
            ins_logit = float(model(insertion, return_attention=False).item())
            del_probs.append(target_probability(del_logit, target_label))
            ins_probs.append(target_probability(ins_logit, target_label))
    x = np.linspace(0, 1, steps + 1)
    return float(np.trapz(del_probs, x)), float(np.trapz(ins_probs, x))


def find_target_layers(model):
    if getattr(model, "use_cbam", False):
        return [model.cbam.output_node]
    if model.architecture == "DenseNet121":
        return [model.backbone.features[-1]]
    return [model.backbone.layer4[-1]]


def make_guided_gradcam(guided_backprop, cam_map):
    if guided_backprop.ndim == 3:
        gb_gray = np.max(np.abs(guided_backprop), axis=-1)
    else:
        gb_gray = np.abs(guided_backprop)
    if gb_gray.shape != cam_map.shape:
        gb_gray = cv2.resize(gb_gray, (cam_map.shape[1], cam_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    guided = gb_gray * cam_map
    if np.max(guided) > 0:
        guided = guided / np.max(guided)
    return guided.astype(np.float32)


def parameter_randomization_corr(base_model, wrapped_model, image, original_cam, original_guided, target_label, target_layers, device):
    head = base_model.backbone.classifier if base_model.architecture == "DenseNet121" else base_model.backbone.fc
    original_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.apply(lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)
    with GradCAM(model=wrapped_model, target_layers=target_layers) as rand_cam:
        rand_map = rand_cam(input_tensor=image, targets=[ClassifierOutputTarget(int(target_label))])[0, :]
    gb = GuidedBackpropReLUModel(model=wrapped_model, device=device)
    rand_gb = gb(image, target_category=int(target_label))
    rand_guided = make_guided_gradcam(rand_gb, rand_map)
    head.load_state_dict(original_state)
    corr_cam = float(np.corrcoef(original_cam.reshape(-1), rand_map.reshape(-1))[0, 1])
    corr_guided = float(np.corrcoef(original_guided.reshape(-1), rand_guided.reshape(-1))[0, 1])
    if not np.isfinite(corr_cam):
        corr_cam = 0.0
    if not np.isfinite(corr_guided):
        corr_guided = 0.0
    return corr_cam, corr_guided
'''


RUN_CODE = r'''
manifest_path = find_manifest()
manifest = pd.read_csv(manifest_path).head(XAI_SAMPLE_SIZE).copy()
image_index, mask_index = build_image_index()

missing_images = [name for name in manifest["image_id"] if name not in image_index]
missing_masks = [name for name in manifest["mask_id"] if name not in mask_index]
if missing_images or missing_masks:
    raise FileNotFoundError(json.dumps({
        "missing_images_count": len(missing_images),
        "missing_masks_count": len(missing_masks),
        "missing_images_first_10": missing_images[:10],
        "missing_masks_first_10": missing_masks[:10],
    }, indent=2))

presence = {
    "manifest_path": str(manifest_path),
    "n_manifest_rows": int(len(manifest)),
    "class_counts": manifest["true_label"].astype(int).value_counts().sort_index().astype(int).to_dict(),
    "n_indexed_images": int(len(image_index)),
    "n_indexed_masks": int(len(mask_index)),
    "missing_images": 0,
    "missing_masks": 0,
}
(OUT_DIR / "input_audit.json").write_text(json.dumps(presence, indent=2), encoding="utf-8")
print(json.dumps(presence, indent=2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_rows = []
summary_rows = []

for seed in SEEDS:
    setup_seed(seed)
    checkpoint_path, threshold_path = find_checkpoint_and_threshold(seed)
    threshold_payload = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    threshold = float(threshold_payload["threshold"])
    model = ClassificationModel(ARCHITECTURE, use_cbam=USE_CBAM).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    wrapped = TwoClassWrapper(model).to(device).eval()
    target_layers = find_target_layers(model)
    figures_saved = 0
    started = time.time()

    print(f"[{XAI_ID}] seed={seed} checkpoint={checkpoint_path} threshold={threshold}")

    with GradCAM(model=wrapped, target_layers=target_layers) as cam:
        guided_model = GuidedBackpropReLUModel(model=wrapped, device=device)
        for row_idx, item in manifest.reset_index(drop=True).iterrows():
            image_id = item["image_id"]
            true_label = int(item["true_label"])
            image_tensor, image_vis = load_and_preprocess_image(image_index[image_id])
            mask = load_mask(mask_index[item["mask_id"]])
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                logit = float(model(image_tensor, return_attention=False).item())
                prob = float(1.0 / (1.0 + np.exp(-logit)))
                pred_label = int(prob >= threshold)
                correct = int(pred_label == true_label)

            target = [ClassifierOutputTarget(pred_label)]
            cam_map = cam(input_tensor=image_tensor, targets=target)[0, :]
            guided_raw = guided_model(image_tensor, target_category=pred_label)
            guided_map = make_guided_gradcam(guided_raw, cam_map)

            g_lrr, g_abs, chance, g_enrich = compute_lrr(cam_map, mask)
            gg_lrr, gg_abs, _, gg_enrich = compute_lrr(guided_map, mask)
            g_point, g_x, g_y = pointing_inside(cam_map, mask)
            gg_point, gg_x, gg_y = pointing_inside(guided_map, mask)
            g_del, g_ins = deletion_insertion_auc(model, image_tensor, cam_map, pred_label, steps=DI_STEPS)
            gg_del, gg_ins = deletion_insertion_auc(model, image_tensor, guided_map, pred_label, steps=DI_STEPS)

            if STABILITY_ENABLED:
                noise = torch.randn_like(image_tensor) * 0.01
                noisy = torch.clamp(image_tensor + noise, -3.0, 3.0).detach()
                noisy_cam = cam(input_tensor=noisy, targets=target)[0, :]
                noisy_guided = make_guided_gradcam(guided_model(noisy, target_category=pred_label), noisy_cam)
                noisy_g_lrr = compute_lrr(noisy_cam, mask)[0]
                noisy_gg_lrr = compute_lrr(noisy_guided, mask)[0]
                g_stability_delta = float(abs(g_lrr - noisy_g_lrr))
                gg_stability_delta = float(abs(gg_lrr - noisy_gg_lrr))
            else:
                g_stability_delta = np.nan
                gg_stability_delta = np.nan

            if RANDOMIZATION_ENABLED:
                g_rand_corr, gg_rand_corr = parameter_randomization_corr(
                    model, wrapped, image_tensor, cam_map, guided_map, pred_label, target_layers, device
                )
            else:
                g_rand_corr = np.nan
                gg_rand_corr = np.nan

            out = {
                "xai_id": XAI_ID,
                "condition": CONDITION,
                "architecture": ARCHITECTURE,
                "seed": int(seed),
                "sample_index": int(item["sample_index"]),
                "sampling_stratum": item["sampling_stratum"],
                "image_id": image_id,
                "true_label": true_label,
                "prob_pneumonia": prob,
                "threshold": threshold,
                "pred_label": pred_label,
                "is_correct": correct,
                "gcam_lrr_plus": g_lrr,
                "gcam_lrr_abs": g_abs,
                "gcam_lrr_chance": chance,
                "gcam_lrr_enrichment": g_enrich,
                "guided_gcam_lrr_plus": gg_lrr,
                "guided_gcam_lrr_abs": gg_abs,
                "guided_gcam_lrr_enrichment": gg_enrich,
                "gcam_pointing_inside_lung": g_point,
                "guided_gcam_pointing_inside_lung": gg_point,
                "gcam_argmax_x": g_x,
                "gcam_argmax_y": g_y,
                "guided_gcam_argmax_x": gg_x,
                "guided_gcam_argmax_y": gg_y,
                "gcam_deletion_auc": g_del,
                "gcam_insertion_auc": g_ins,
                "guided_gcam_deletion_auc": gg_del,
                "guided_gcam_insertion_auc": gg_ins,
                "gcam_input_stability_lrr_delta": g_stability_delta,
                "guided_gcam_input_stability_lrr_delta": gg_stability_delta,
                "gcam_head_randomization_corr": g_rand_corr,
                "guided_gcam_head_randomization_corr": gg_rand_corr,
                "checkpoint_path": str(checkpoint_path),
            }
            all_rows.append(out)

            if figures_saved < MAX_FIGURES_PER_SEED:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(image_vis)
                axes[0].set_title(f"True {true_label} / Pred {pred_label}")
                axes[0].axis("off")
                axes[1].imshow(mask, cmap="gray")
                axes[1].set_title("Lung mask")
                axes[1].axis("off")
                axes[2].imshow(show_cam_on_image(image_vis, cam_map, use_rgb=True))
                axes[2].set_title(f"Grad-CAM LRR {g_lrr:.2f}")
                axes[2].axis("off")
                axes[3].imshow(show_cam_on_image(image_vis, guided_map, use_rgb=True))
                axes[3].set_title(f"Guided GC LRR {gg_lrr:.2f}")
                axes[3].axis("off")
                fig.tight_layout()
                fig.savefig(FIG_DIR / f"{XAI_ID}_seed{seed}_{row_idx:03d}_{Path(image_id).stem}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                figures_saved += 1

            if (row_idx + 1) % 16 == 0:
                partial = pd.DataFrame(all_rows)
                partial.to_csv(OUT_DIR / "xai_metrics_by_image_partial.csv", index=False)
                print(f"[{XAI_ID}] seed={seed} processed {row_idx + 1}/{len(manifest)}")

            del image_tensor
            torch.cuda.empty_cache()

    seed_rows = [r for r in all_rows if r["seed"] == seed]
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT_DIR / f"xai_metrics_seed_{seed}.csv", index=False)
    summary = {
        "xai_id": XAI_ID,
        "condition": CONDITION,
        "architecture": ARCHITECTURE,
        "seed": int(seed),
        "n_samples": int(len(seed_df)),
        "accuracy_on_xai_sample": float(seed_df["is_correct"].mean()),
        "gcam_lrr_plus_mean": float(seed_df["gcam_lrr_plus"].mean()),
        "guided_gcam_lrr_plus_mean": float(seed_df["guided_gcam_lrr_plus"].mean()),
        "gcam_pointing_inside_lung_mean": float(seed_df["gcam_pointing_inside_lung"].mean()),
        "guided_gcam_pointing_inside_lung_mean": float(seed_df["guided_gcam_pointing_inside_lung"].mean()),
        "gcam_deletion_auc_mean": float(seed_df["gcam_deletion_auc"].mean()),
        "guided_gcam_deletion_auc_mean": float(seed_df["guided_gcam_deletion_auc"].mean()),
        "gcam_insertion_auc_mean": float(seed_df["gcam_insertion_auc"].mean()),
        "guided_gcam_insertion_auc_mean": float(seed_df["guided_gcam_insertion_auc"].mean()),
        "gcam_head_randomization_corr_mean": float(seed_df["gcam_head_randomization_corr"].mean()),
        "guided_gcam_head_randomization_corr_mean": float(seed_df["guided_gcam_head_randomization_corr"].mean()),
        "runtime_seconds": float(time.time() - started),
    }
    summary_rows.append(summary)
    print(json.dumps(summary, indent=2))

all_df = pd.DataFrame(all_rows)
summary_df = pd.DataFrame(summary_rows)
all_df.to_csv(OUT_DIR / "xai_metrics_by_image.csv", index=False)
summary_df.to_csv(OUT_DIR / "xai_summary_by_seed.csv", index=False)

audit = {
    "xai_id": XAI_ID,
    "condition": CONDITION,
    "architecture": ARCHITECTURE,
    "use_cbam": USE_CBAM,
    "seeds": SEEDS,
    "n_expected_rows": int(len(SEEDS) * len(manifest)),
    "n_output_rows": int(len(all_df)),
    "n_manifest_rows": int(len(manifest)),
    "manifest_path": str(manifest_path),
    "di_steps": DI_STEPS,
    "max_figures_per_seed": MAX_FIGURES_PER_SEED,
    "stability_enabled": STABILITY_ENABLED,
    "randomization_enabled": RANDOMIZATION_ENABLED,
    "input_validation_passed": bool(len(all_df) == len(SEEDS) * len(manifest)),
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "outputs": sorted([p.name for p in OUT_DIR.iterdir() if p.is_file()]),
}
(OUT_DIR / "xai_run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
'''


for run in RUNS:
    title = run["filename"].replace(".ipynb", "")
    run_json = json.dumps(run)
    cells = [
        markdown_cell(
            f"""
# {title}

Purpose: locked Guided Grad-CAM validity run for `{run['condition']}` / `{run['architecture']}` across seeds `{run['seeds']}`.

Add these Kaggle inputs:

- `hintrngia/gate7-test-seal`
- `hintrngia/cxr-lung-masks`
- a dataset containing `xai_sample_manifest_128.csv` from C00
- a dataset containing B-run outputs with `best_checkpoint.pt` and `threshold.json` for `{run['condition']}` / `{run['architecture']}` / seeds `{run['seeds']}`

Outputs are written to:

`/kaggle/working/locked_outputs/xai_runs/{run['xai_id']}/`

The notebook evaluates all 128 locked XAI samples for all three seeds and saves a limited figure subset per seed to avoid excessive Kaggle output size.
"""
        ),
        markdown_cell("## Setup"),
        code_cell(SETUP_CODE),
        markdown_cell("## Configuration"),
        code_cell(MAIN_CODE.replace("__RUN_CONFIG_JSON__", run_json)),
        markdown_cell("## Model Definition"),
        code_cell(MODEL_CODE),
        markdown_cell("## XAI Helper Functions"),
        code_cell(HELPERS_CODE),
        markdown_cell("## Run Locked XAI Battery"),
        code_cell(RUN_CODE),
    ]
    write_notebook(OUT_DIR / run["filename"], cells)
