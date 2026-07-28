import json
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
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import densenet121, resnet50


ROOT = Path(__file__).resolve().parents[1]
XAI_CSV = ROOT / "results/locked_final/xai_runs/aggregate/xai_all_by_image.csv"
FINAL_ROOT = ROOT / "results/locked_final/classification_runs"
IMAGE_DIR = ROOT / "pipeline/input/opaque_test_images/images"
MASK_DIR = ROOT / "pipeline/input/lung_masks"
OUT_DIR = ROOT / "paper/figures"
AUDIT_DIR = ROOT / "results/locked_final/xai_runs/aggregate/selected_qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 3407
INPUT_RESOLUTION = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    def __init__(self, architecture, use_cbam=False):
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
            raise ValueError(architecture)

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


def load_image(image_id):
    gray = cv2.imread(str(IMAGE_DIR / image_id), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(IMAGE_DIR / image_id)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (INPUT_RESOLUTION, INPUT_RESOLUTION), interpolation=cv2.INTER_AREA)
    vis = rgb.astype(np.float32) / 255.0
    normalized = (vis - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, vis


def load_mask(image_id):
    mask = cv2.imread(str(MASK_DIR / image_id), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(MASK_DIR / image_id)
    mask = cv2.resize(mask, (INPUT_RESOLUTION, INPUT_RESOLUTION), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


def load_model(architecture, condition, device):
    use_cbam = condition.endswith("-P")
    model = ClassificationModel(architecture, use_cbam=use_cbam).to(device)
    ckpt = FINAL_ROOT / condition / architecture / f"seed_{SEED}" / "best_checkpoint.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    threshold = json.loads((ckpt.parent / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    return model, float(threshold), ckpt


def target_layers(model):
    if model.use_cbam:
        return [model.cbam.output_node]
    if model.architecture == "DenseNet121":
        return [model.backbone.features[-1]]
    return [model.backbone.layer4[-1]]


def guided_map_for(model, image_tensor, pred_label, device):
    wrapped = TwoClassWrapper(model).to(device).eval()
    with GradCAM(model=wrapped, target_layers=target_layers(model)) as cam:
        cam_map = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(int(pred_label))])[0, :]
    guided_model = GuidedBackpropReLUModel(model=wrapped, device=device)
    guided_raw = guided_model(image_tensor, target_category=int(pred_label))
    gb_gray = np.max(np.abs(guided_raw), axis=-1) if guided_raw.ndim == 3 else np.abs(guided_raw)
    if gb_gray.shape != cam_map.shape:
        gb_gray = cv2.resize(gb_gray, (cam_map.shape[1], cam_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    guided = gb_gray * cam_map
    if np.max(guided) > 0:
        guided = guided / np.max(guided)
    return cam_map.astype(np.float32), guided.astype(np.float32)


def lrr(heatmap, mask):
    positive = np.maximum(heatmap, 0)
    total = float(positive.sum())
    return float((positive * mask).sum() / total) if total > 0 else 0.0


def select_cases(architecture, control, proposed):
    df = pd.read_csv(XAI_CSV)
    c0 = df[(df.architecture == architecture) & (df.condition == control) & (df.seed == SEED)].set_index("image_id")
    pp = df[(df.architecture == architecture) & (df.condition == proposed) & (df.seed == SEED)].set_index("image_id")
    rows = []
    for image_id in c0.index.intersection(pp.index):
        r = pp.loc[image_id]
        outcome = ("T" if int(r.pred_label) == int(r.true_label) else "F") + ("P" if int(r.pred_label) == 1 else "N")
        rows.append(
            {
                "image_id": image_id,
                "true_label": int(r.true_label),
                "pred_label": int(r.pred_label),
                "outcome": outcome,
                "control_gcam_lrr": float(c0.loc[image_id].gcam_lrr_plus),
                "control_guided_lrr": float(c0.loc[image_id].guided_gcam_lrr_plus),
                "proposed_gcam_lrr": float(r.gcam_lrr_plus),
                "proposed_guided_lrr": float(r.guided_gcam_lrr_plus),
                "gcam_delta": float(r.gcam_lrr_plus - c0.loc[image_id].gcam_lrr_plus),
                "guided_delta": float(r.guided_gcam_lrr_plus - c0.loc[image_id].guided_gcam_lrr_plus),
                "control_gcam_pointing": int(c0.loc[image_id].gcam_pointing_inside_lung),
                "control_guided_pointing": int(c0.loc[image_id].guided_gcam_pointing_inside_lung),
                "proposed_gcam_pointing": int(r.gcam_pointing_inside_lung),
                "proposed_guided_pointing": int(r.guided_gcam_pointing_inside_lung),
            }
        )
    candidates = pd.DataFrame(rows)
    selected = []
    for outcome in ["TN", "TP", "FP", "FN"]:
        sub = candidates[candidates.outcome == outcome].copy()
        if sub.empty:
            continue
        sub["both_p_point_inside"] = sub["proposed_gcam_pointing"] + sub["proposed_guided_pointing"]
        sub["min_p_lrr"] = sub[["proposed_gcam_lrr", "proposed_guided_lrr"]].min(axis=1)
        sub["total_delta"] = sub["gcam_delta"] + sub["guided_delta"]
        sub = sub.sort_values(
            ["both_p_point_inside", "min_p_lrr", "total_delta", "proposed_guided_lrr"],
            ascending=False,
        )
        selected.append(sub.iloc[0].to_dict())
    return candidates, pd.DataFrame(selected)


def build_panel(architecture, control, proposed, out_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates, selected = select_cases(architecture, control, proposed)
    models = {}
    audit_rows = []
    for condition in [control, proposed]:
        models[condition] = load_model(architecture, condition, device)

    nrows = len(selected)
    fig, axes = plt.subplots(nrows, 4, figsize=(12, 3.45 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, item in selected.reset_index(drop=True).iterrows():
        image_id = item["image_id"]
        image_tensor, image_vis = load_image(image_id)
        mask = load_mask(image_id)
        image_tensor = image_tensor.to(device)
        gcam_overlays = {}
        guided_overlays = {}
        measured = {}
        preds = {}
        for condition, (model, threshold, ckpt) in models.items():
            with torch.no_grad():
                prob = float(torch.sigmoid(model(image_tensor)).item())
            pred = int(prob >= threshold)
            cam_map, guided = guided_map_for(model, image_tensor, pred, device)
            gcam_overlays[condition] = show_cam_on_image(image_vis, cam_map, use_rgb=True)
            guided_overlays[condition] = show_cam_on_image(image_vis, guided, use_rgb=True)
            measured[f"{condition}_gcam"] = lrr(cam_map, mask)
            measured[f"{condition}_guided"] = lrr(guided, mask)
            preds[condition] = (pred, prob, threshold, str(ckpt))

        label = "normal" if int(item["true_label"]) == 0 else "pneumonia"
        axes[row_idx, 0].imshow(image_vis)
        axes[row_idx, 0].set_title(f"{item['outcome']} / true {label}", fontsize=9)
        axes[row_idx, 1].imshow(gcam_overlays[control])
        axes[row_idx, 1].set_title(f"{control} Grad-CAM\nLRR+ {measured[f'{control}_gcam']:.2f}", fontsize=9)
        axes[row_idx, 2].imshow(gcam_overlays[proposed])
        axes[row_idx, 2].set_title(f"{proposed} Grad-CAM\nLRR+ {measured[f'{proposed}_gcam']:.2f}", fontsize=9)
        axes[row_idx, 3].imshow(guided_overlays[proposed])
        axes[row_idx, 3].set_title(f"{proposed} Guided GC\nLRR+ {measured[f'{proposed}_guided']:.2f}", fontsize=9)
        for ax in axes[row_idx]:
            ax.axis("off")
        audit_rows.append(
            {
                **item,
                "architecture": architecture,
                "seed": SEED,
                "control": control,
                "proposed": proposed,
                "recomputed_control_gcam_lrr": measured[f"{control}_gcam"],
                "recomputed_control_guided_lrr": measured[f"{control}_guided"],
                "recomputed_proposed_gcam_lrr": measured[f"{proposed}_gcam"],
                "recomputed_proposed_guided_lrr": measured[f"{proposed}_guided"],
                "control_pred": preds[control][0],
                "proposed_pred": preds[proposed][0],
                "control_prob_pneumonia": preds[control][1],
                "proposed_prob_pneumonia": preds[proposed][1],
                "control_checkpoint": preds[control][3],
                "proposed_checkpoint": preds[proposed][3],
            }
        )

    fig.tight_layout(pad=0.7, h_pad=1.25, w_pad=0.35)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    selected.to_csv(AUDIT_DIR / f"{out_path.stem}_selected_cases.csv", index=False)
    candidates.to_csv(AUDIT_DIR / f"{out_path.stem}_candidate_pool.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT_DIR / f"{out_path.stem}_recomputed_audit.csv", index=False)
    return out_path


def main():
    outputs = [
        build_panel("DenseNet121", "D-C0", "D-P", "xai_qualitative_densenet_guided_cases.png"),
        build_panel("ResNet50", "R-C0", "R-P", "xai_qualitative_resnet_guided_cases.png"),
    ]
    summary = {
        "seed": SEED,
        "outputs": [str(p) for p in outputs],
        "selection_rule": (
            "For each available P-model outcome stratum in the 128-case XAI sample, "
            "select the case with both proposed Grad-CAM and proposed Guided Grad-CAM "
            "pointing inside lung, then largest minimum proposed LRR+ across the two "
            "methods, then largest total proposed-minus-control LRR+ delta."
        ),
        "note": "No FN stratum was present for the proposed models in seed 3407 XAI sample.",
    }
    (AUDIT_DIR / "selected_qualitative_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
