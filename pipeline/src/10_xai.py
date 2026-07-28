import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from pathlib import Path
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF

# XAI Libraries (Ensure these are installed in the environment: pip install grad-cam)
try:
    from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("[XAI] WARNING: pytorch-grad-cam is missing. XAI features will be disabled.")

class BinaryClassifierOutputTarget:
    """Custom target for binary classification to explain the True Class."""
    def __init__(self, category):
        self.category = category
        
    def __call__(self, model_output):
        if self.category == 1:
            return model_output[0]
        # For class 0 (Normal), we negate the logit to find features that pushed it negative
        return -model_output[0]

def compute_lrr(saliency_map, lung_mask):
    """
    Computes Lung Relevance Ratio (LRR+) and variants.
    """
    # Resize mask to match saliency map if needed
    if saliency_map.shape != lung_mask.shape:
        lung_mask = cv2.resize(lung_mask, (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Ensure binary mask
    lung_mask = (lung_mask > 0).astype(np.float32)
    
    # Calculate positive relevance only (LRR+)
    pos_relevance = np.maximum(saliency_map, 0)
    
    relevance_in_lungs = np.sum(pos_relevance * lung_mask)
    total_relevance = np.sum(pos_relevance)
    
    lrr_plus = relevance_in_lungs / total_relevance if total_relevance > 0 else 0.0
    
    # Absolute relevance
    abs_relevance = np.abs(saliency_map)
    abs_in_lungs = np.sum(abs_relevance * lung_mask)
    total_abs = np.sum(abs_relevance)
    lrr_abs = abs_in_lungs / total_abs if total_abs > 0 else 0.0
    
    # Random chance (Area of lungs / Area of image)
    lung_area = np.sum(lung_mask)
    total_area = lung_mask.shape[0] * lung_mask.shape[1]
    lrr_chance = lung_area / total_area if total_area > 0 else 0.0
    
    # Enrichment score
    lrr_enrichment = lrr_plus / lrr_chance if lrr_chance > 0 else 0.0
    
    return {
        "lrr_plus": float(lrr_plus),
        "lrr_abs": float(lrr_abs),
        "lrr_chance": float(lrr_chance),
        "lrr_enrichment": float(lrr_enrichment)
    }

def _target_probability_from_logit(logit, target_label):
    signed_logit = logit if target_label == 1 else -logit
    return 1 / (1 + np.exp(-signed_logit))

def compute_deletion_insertion_auc(model, image, heatmap, target_label, steps=10):
    # image: [1, C, H, W]
    # heatmap: [H, W] (already resized to image shape)
    
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1] # descending
    num_pixels = len(sorted_indices)
    
    blurred_image = TF.gaussian_blur(image, kernel_size=21, sigma=[5.0, 5.0])
    
    del_probs = []
    ins_probs = []
    
    del_img = image.clone().detach()
    ins_img = blurred_image.clone().detach()
    
    with torch.no_grad():
        orig_logit = model(del_img, return_attention=False).item()
        blur_logit = model(ins_img, return_attention=False).item()
        
        del_probs.append(_target_probability_from_logit(orig_logit, target_label))
        ins_probs.append(_target_probability_from_logit(blur_logit, target_label))
        
        step_size = num_pixels // steps
        
        for i in range(1, steps + 1):
            mask_indices = sorted_indices[:i * step_size]
            rows = mask_indices // heatmap.shape[1]
            cols = mask_indices % heatmap.shape[1]
            
            del_img[0, :, rows, cols] = blurred_image[0, :, rows, cols]
            ins_img[0, :, rows, cols] = image[0, :, rows, cols]
            
            d_logit = model(del_img, return_attention=False).item()
            i_logit = model(ins_img, return_attention=False).item()
            
            del_probs.append(_target_probability_from_logit(d_logit, target_label))
            ins_probs.append(_target_probability_from_logit(i_logit, target_label))
            
    x = np.linspace(0, 1, steps + 1)
    del_auc = np.trapz(del_probs, x)
    ins_auc = np.trapz(ins_probs, x)
    
    return float(del_auc), float(ins_auc)

def compute_input_stability(model, cam, gb_model, image, lung_mask, predicted_label, orig_gcam_lrr, orig_gb_lrr, target_class):
    noise = torch.randn_like(image) * 0.01
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, -2.5, 2.5).clone().detach().requires_grad_(True)
    
    noisy_gcam = cam(input_tensor=noisy_image, targets=target_class)[0, :]
    noisy_gcam_lrr = compute_lrr(noisy_gcam, lung_mask)['lrr_plus']
    
    try:
        gb_map = gb_model(noisy_image, target_category=predicted_label)
    except:
        try:
            gb_map = gb_model(noisy_image, targets=target_class)
        except:
            gb_map = gb_model(noisy_image)
            
    if isinstance(gb_map, list): gb_map = gb_map[0]
    if torch.is_tensor(gb_map): gb_map = gb_map.detach().cpu().numpy()
    if gb_map.ndim == 4: gb_map = gb_map[0]
    if gb_map.ndim == 3 and gb_map.shape[0] in [1, 3]: gb_map = np.transpose(gb_map, (1, 2, 0))
    
    gb_gray = np.max(np.abs(gb_map), axis=-1) if gb_map.ndim == 3 else np.abs(gb_map)
    if gb_gray.shape != noisy_gcam.shape:
        gb_gray = cv2.resize(gb_gray, (noisy_gcam.shape[1], noisy_gcam.shape[0]))
        
    noisy_guided = gb_gray * noisy_gcam
    noisy_guided_lrr = compute_lrr(noisy_guided, lung_mask)['lrr_plus']
    
    return float(abs(orig_gcam_lrr - noisy_gcam_lrr)), float(abs(orig_gb_lrr - noisy_guided_lrr))

def compute_parameter_randomization(model, cam, gb_model, image, orig_gcam_map, orig_guided_map, predicted_label, target_class):
    orig_state_dict = {}
    head_layers = []
    
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'fc'):
        head_layers.append(model.backbone.fc)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'classifier'):
        head_layers.append(model.backbone.classifier)
    if hasattr(model, 'head'):
        head_layers.append(model.head)
    if not head_layers:
        raise RuntimeError("No classifier head found for parameter randomization sanity check.")
        
    for layer in head_layers:
        orig_state_dict[layer] = {k: v.clone() for k, v in layer.state_dict().items()}
        layer.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
    rand_gcam = cam(input_tensor=image, targets=target_class)[0, :]
    
    try:
        gb_map = gb_model(image, target_category=predicted_label)
    except:
        try:
            gb_map = gb_model(image, targets=target_class)
        except:
            gb_map = gb_model(image)
            
    if isinstance(gb_map, list): gb_map = gb_map[0]
    if torch.is_tensor(gb_map): gb_map = gb_map.detach().cpu().numpy()
    if gb_map.ndim == 4: gb_map = gb_map[0]
    if gb_map.ndim == 3 and gb_map.shape[0] in [1, 3]: gb_map = np.transpose(gb_map, (1, 2, 0))
    
    gb_gray = np.max(np.abs(gb_map), axis=-1) if gb_map.ndim == 3 else np.abs(gb_map)
    if gb_gray.shape != rand_gcam.shape:
        gb_gray = cv2.resize(gb_gray, (rand_gcam.shape[1], rand_gcam.shape[0]))
        
    rand_guided = gb_gray * rand_gcam
    
    for layer in head_layers:
        layer.load_state_dict(orig_state_dict[layer])
        
    try:
        gcam_corr = pearsonr(orig_gcam_map.flatten(), rand_gcam.flatten())[0]
        guided_corr = pearsonr(orig_guided_map.flatten(), rand_guided.flatten())[0]
    except:
        gcam_corr, guided_corr = 0.0, 0.0
        
    return float(gcam_corr), float(guided_corr)

def generate_explanations(model, dataloader, device, lung_masks_dir: Path, save_figures: bool = False, output_dir: Path = None, max_figures: int = 20, threshold: float = 0.0, max_samples: int = 100):
    """
    Generates Grad-CAM and Guided Grad-CAM attributions and computes LRR metrics.
    Requires lung masks to be pre-generated in `lung_masks_dir`.
    Args:
        threshold: Logit threshold for binary prediction (default 0.0 = sigmoid 0.5).
                   Pass the Youden's J optimal_threshold (as logit) for accurate predictions.
        max_samples: Maximum test samples to evaluate for XAI (default 100 for fast CPU/GPU execution).
    """
    if not XAI_AVAILABLE:
        print("[XAI] Skipping explanations due to missing dependencies.")
        return []
        
    if save_figures and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from pytorch_grad_cam.utils.image import show_cam_on_image
            import matplotlib.pyplot as plt
        except ImportError:
            print("[XAI] WARNING: Cannot save figures without matplotlib and pytorch-grad-cam utils.")
            save_figures = False
            
    model.eval()
    results = []
    figures_saved = 0
    
    # Grad-CAM Setup
    # Note: Target layer depends on architecture (DenseNet vs ResNet) and whether CBAM is used
    if getattr(model, 'use_cbam', False):
        target_layers = [model.cbam.output_node] # Target the filtered features after Mask Loss!
    elif hasattr(model.backbone, 'features'):
        target_layers = [model.backbone.features[-1]] # DenseNet fallback
    else:
        target_layers = [model.backbone.layer4[-1]] # ResNet fallback
        
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Guided Backprop Setup
    try:
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
    except TypeError:
        try:
            gb_model = GuidedBackpropReLUModel(model=model, device=str(device))
        except TypeError:
            try:
                gb_model = GuidedBackpropReLUModel(model=model)
            except TypeError:
                gb_model = GuidedBackpropReLUModel(model=model, use_cuda=(device.type == 'cuda'))
    
    print(f"[XAI] Running Explainability Analysis (Grad-CAM & Guided Grad-CAM) for up to {max_samples} samples...")
    
    for i, batch in enumerate(dataloader):
        if max_samples > 0 and len(results) >= max_samples:
            break
            
        images_batch = batch[0].to(device)
        labels_batch = batch[1]
        img_names_batch = batch[2]
        
        for b in range(images_batch.size(0)):
            img_name = img_names_batch[b]
            mask_path = lung_masks_dir / img_name
            
            if not mask_path.exists():
                continue # Skip if no mask available
                
            lung_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if lung_mask is None:
                continue
                
            # Process 1 image at a time to prevent OOM
            image = images_batch[b:b+1].clone().detach().requires_grad_(True)
            label = int(labels_batch[b].item())
            
            # Get model's own prediction (XAI explains the model's decision, NOT the true label)
            # This avoids blank GradCAM on False Negatives (model predicts Normal but true label is Pneumonia)
            with torch.no_grad():
                raw_logit = model(image.detach(), return_attention=False)
            predicted_label = 1 if raw_logit.item() >= threshold else 0
            
            # 1. Grad-CAM - explain the predicted class
            target_class = [BinaryClassifierOutputTarget(predicted_label)]
            grayscale_cam = cam(input_tensor=image, targets=target_class)
            gcam_map = grayscale_cam[0, :]
            gcam_lrr = compute_lrr(gcam_map, lung_mask)
            
            # 2. Guided Grad-CAM - explain the predicted class
            # First, compute Guided Backprop
            try:
                gb_map_raw = gb_model(image, target_category=predicted_label)
            except Exception:
                try:
                    gb_map_raw = gb_model(image, targets=target_class)
                except Exception:
                    gb_map_raw = gb_model(image)

            if isinstance(gb_map_raw, list):
                gb_map_raw = gb_map_raw[0]
            if torch.is_tensor(gb_map_raw):
                gb_map_raw = gb_map_raw.detach().cpu().numpy()
            
            if gb_map_raw.ndim == 4:
                gb_map_raw = gb_map_raw[0]
            if gb_map_raw.ndim == 3 and gb_map_raw.shape[0] in [1, 3]:
                gb_map_raw = np.transpose(gb_map_raw, (1, 2, 0))
            
            # Convert Guided Backprop to grayscale by taking max across channels
            if gb_map_raw.ndim == 3:
                gb_grayscale = np.max(np.abs(gb_map_raw), axis=-1)
            else:
                gb_grayscale = np.abs(gb_map_raw)
            
            # Element-wise multiply Guided Backprop with Grad-CAM heatmap to get Guided Grad-CAM
            if gb_grayscale.shape != gcam_map.shape:
                gb_grayscale = cv2.resize(gb_grayscale, (gcam_map.shape[1], gcam_map.shape[0]))
            
            guided_gcam_map_raw = gb_grayscale * gcam_map
            guided_gcam_lrr = compute_lrr(guided_gcam_map_raw, lung_mask)
            
            # --- Advanced Robustness Metrics ---
            gcam_del_auc, gcam_ins_auc = compute_deletion_insertion_auc(model, image, gcam_map, predicted_label)
            
            guided_abs = np.abs(guided_gcam_map_raw)
            if guided_abs.shape != gcam_map.shape:
                guided_abs = cv2.resize(guided_abs, (gcam_map.shape[1], gcam_map.shape[0]))
            guided_del_auc, guided_ins_auc = compute_deletion_insertion_auc(model, image, guided_abs, predicted_label)
            
            gcam_stab, guided_stab = compute_input_stability(model, cam, gb_model, image, lung_mask, predicted_label, gcam_lrr['lrr_plus'], guided_gcam_lrr['lrr_plus'], target_class)
            
            gcam_rand, guided_rand = compute_parameter_randomization(model, cam, gb_model, image, gcam_map, guided_gcam_map_raw, predicted_label, target_class)
            
            # Aggressive memory cleanup
            del image
            torch.cuda.empty_cache()
            
            # Save visual heatmaps if requested
            if save_figures and figures_saved < max_figures:
                # Unnormalize image back to [0, 1] for visualization
                img_np = images_batch[b].detach().cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_unnorm = np.clip(img_np * std + mean, 0, 1).astype(np.float32)
                
                # Grad-CAM overlay
                gcam_vis = show_cam_on_image(img_unnorm, gcam_map, use_rgb=True)
                
                # Guided Grad-CAM visualization
                guided_gcam_map_norm = np.maximum(guided_gcam_map_raw, 0)
                if np.max(guided_gcam_map_norm) > 0:
                    guided_gcam_map_norm = guided_gcam_map_norm / np.max(guided_gcam_map_norm)
                # Guided Grad-CAM is usually displayed as a heatmap or superimposed. 
                guided_gcam_vis = show_cam_on_image(img_unnorm, guided_gcam_map_norm, use_rgb=True)
                
                plt.rcParams.update({
                    "font.family": "serif",
                    "font.serif": ["Times New Roman", "DejaVu Serif"],
                    "font.size": 14,
                    "axes.titlesize": 16
                })
                
                # Create side-by-side plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_unnorm)
                correct = "OK" if predicted_label == label else "WRONG"
                axes[0].set_title(f"True:{label} | Pred:{predicted_label} [{correct}]")
                axes[0].axis('off')
                
                axes[1].imshow(gcam_vis)
                axes[1].set_title(f"Grad-CAM (LRR+: {gcam_lrr['lrr_plus']:.3f})")
                axes[1].axis('off')
                
                axes[2].imshow(guided_gcam_vis)
                axes[2].set_title(f"Guided Grad-CAM (LRR+: {guided_gcam_lrr['lrr_plus']:.3f})")
                axes[2].axis('off')
                
                plt.tight_layout()
                save_stem = Path(img_name).stem
                plt.savefig(output_dir / f"{model.architecture}_{save_stem}.png", bbox_inches='tight')
                plt.close(fig)
                figures_saved += 1
            
            results.append({
                "image": img_name,
                "label": label,
                "gcam_lrr_plus": gcam_lrr['lrr_plus'],
                "guided_gcam_lrr_plus": guided_gcam_lrr['lrr_plus'],
                "lrr_chance": gcam_lrr['lrr_chance'],
                "gcam_del_auc": gcam_del_auc,
                "gcam_ins_auc": gcam_ins_auc,
                "guided_del_auc": guided_del_auc,
                "guided_ins_auc": guided_ins_auc,
                "gcam_stab": gcam_stab,
                "guided_stab": guided_stab,
                "gcam_rand": gcam_rand,
                "guided_rand": guided_rand
            })
            
    print(f"[XAI] Completed analysis for {len(results)} images. Saved {figures_saved} figures.")

    return results
