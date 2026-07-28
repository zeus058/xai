import torch
from torch.amp import autocast
import numpy as np

try:
    calculate_metrics
except NameError:
    import importlib.util
    from pathlib import Path
    validation_path = Path(__file__).with_name("08_validation.py")
    spec = importlib.util.spec_from_file_location("pipeline_validation", validation_path)
    validation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_module)
    calculate_metrics = validation_module.calculate_metrics

def evaluate_test_set(model, test_loader, device, config, threshold, temperature):
    """
    Evaluates the model on the TEST SET exactly once.
    Enforces freeze constraints (uses pre-calculated threshold and temperature).
    Applies Test Time Augmentation (TTA) if configured.
    """
    print("\n[GATE 7] Starting Single-Pass Test Evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            labels = labels.cpu().numpy()
            all_labels.extend(labels)
            
            # Move to device
            images = images.to(device)
            
            batch_probs = []
            
            use_autocast = (getattr(config, 'mixed_precision', False) and torch.cuda.is_available())
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=use_autocast):
                # 1. Standard Forward Pass
                logits = model(images)
                # Apply Temperature Scaling
                logits = logits / temperature
                probs = torch.sigmoid(logits).cpu().numpy()
                batch_probs.append(probs)
                
                # 2. Test Time Augmentation (Horizontal Flip)
                if config.use_tta:
                    images_hf = torch.flip(images, dims=[3]) # Flip width
                    logits_hf = model(images_hf)
                    logits_hf = logits_hf / temperature
                    probs_hf = torch.sigmoid(logits_hf).cpu().numpy()
                    batch_probs.append(probs_hf)
                    
            # Average probabilities across TTA augmentations
            avg_probs = np.mean(batch_probs, axis=0)
            all_preds.extend(avg_probs)
            
    all_preds = np.array(all_preds).squeeze()
    all_labels = np.array(all_labels)
    
    print(f"[GATE 7] Test Evaluation complete for {len(all_labels)} samples.")
    print(f"[GATE 7] Using frozen threshold: {threshold:.4f} and temperature: {temperature:.4f}")
    
    metrics = calculate_metrics(all_labels, all_preds, threshold=threshold)
    
    return metrics, all_preds, all_labels
