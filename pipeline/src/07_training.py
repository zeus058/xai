import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import time
import copy

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Calculate alpha properly for positive and negative classes
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_criterion(config, class_weights=None):
    if config.loss_type == "Focal":
        # If class weights are provided, use them as alpha (or a scaled version)
        alpha = config.focal_alpha
        if class_weights is not None:
            # simple mapping of pos_weight to focal alpha
            alpha = float(class_weights[0]) / (1.0 + float(class_weights[0]))
        return FocalLoss(alpha=alpha, gamma=config.focal_gamma)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, config, current_mask_weight=0.0):
    model.train()
    
    # CRITICAL FIX: Ensure frozen BatchNorm layers stay in eval mode to prevent running stats corruption
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'weight') and m.weight is not None and not m.weight.requires_grad:
            m.eval()
            
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for i, (images, labels, _, masks) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        masks = masks.to(device)
        
        use_autocast = (getattr(config, 'mixed_precision', False) and torch.cuda.is_available())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        with autocast(device_type=device_type, enabled=use_autocast):
            if config.use_mask_loss:
                outputs, spatial_att = model(images, return_attention=True)
                loss_clf = criterion(outputs, labels)
                
                # Resize spatial attention to mask size
                spatial_att = F.interpolate(spatial_att, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                # Mask guided loss: encourage attention inside the mask, penalize outside
                # simple BCE between spatial attention and lung mask
                with torch.amp.autocast(device_type, enabled=False):
                    loss_mask = F.binary_cross_entropy(torch.clamp(spatial_att.float(), 1e-7, 1.0 - 1e-7), torch.clamp(masks.float(), 0.0, 1.0))
                
                loss = loss_clf + current_mask_weight * loss_mask
            else:
                outputs = model(images, return_attention=False)
                loss = criterion(outputs, labels)
            
            # Normalize loss for gradient accumulation
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
                
        scaler.scale(loss).backward()
        
        if (i + 1) % config.gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
            if config.gradient_clipping_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping_max_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        running_loss += loss.item() * config.gradient_accumulation_steps
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
    return running_loss / len(dataloader), np.array(all_preds), np.array(all_labels)

def validate(model, dataloader, criterion, device, config, current_mask_weight=0.0):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    use_autocast = (getattr(config, 'mixed_precision', False) and torch.cuda.is_available())
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    with torch.no_grad():
        for images, labels, _, masks in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            masks = masks.to(device)
            
            with autocast(device_type=device_type, enabled=use_autocast):
                if config.use_mask_loss:
                    outputs, spatial_att = model(images, return_attention=True)
                    loss_clf = criterion(outputs, labels)
                    spatial_att = F.interpolate(spatial_att, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    with torch.amp.autocast(device_type, enabled=False):
                        loss_mask = F.binary_cross_entropy(torch.clamp(spatial_att.float(), 1e-7, 1.0 - 1e-7), torch.clamp(masks.float(), 0.0, 1.0))
                    loss = loss_clf + current_mask_weight * loss_mask
                else:
                    outputs = model(images, return_attention=False)
                    loss = criterion(outputs, labels)
                
            running_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(dataloader), np.array(all_preds), np.array(all_labels)

def train_pipeline(model, train_loader, val_loader, device, config, seed, class_weights=None):
    """
    Executes the Two-Stage transfer learning pipeline.
    """
    if getattr(config, 'skip_training_if_checkpoint_exists', False):
        import os
        from pathlib import Path
        model_name = f"{model.architecture}_seed{seed}_best.pt"
        
        # Search candidate directories
        candidate_dirs = [
            Path("/content/drive/MyDrive/q1_rebuild_outputs/models"),
            Path("/content/q1_rebuild_outputs/models"),
            Path("/content/models"),
            Path("/content"),
            Path("./outputs/models"),
            Path("./local_work/models"),
            Path("./models"),
            Path("."),
            Path("/kaggle/working/models"),
            Path("/kaggle/working")
        ]
        
        chkpt_path = None
        for cdir in candidate_dirs:
            if (cdir / model_name).exists():
                chkpt_path = cdir / model_name
                break
        
        if chkpt_path is None:
            # Fallback recursive search in /content and current dir
            for search_root in [Path("/content/drive/MyDrive"), Path("/content"), Path(".")]:
                if search_root.exists():
                    matches = list(search_root.rglob(model_name))
                    if matches:
                        chkpt_path = matches[0]
                        break
        
        if chkpt_path is not None and chkpt_path.exists():
            print(f"\n[TRAIN] Found existing checkpoint: {chkpt_path}")
            print("[TRAIN] skip_training_if_checkpoint_exists=True. Loading weights and skipping training...")
            model.load_state_dict(torch.load(chkpt_path, map_location=device, weights_only=True))
            model.to(device)
            return model

    print(f"\n--- Starting Training for {model.architecture} (Seed: {seed}) ---")
    model = model.to(device)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        
    criterion = get_criterion(config, class_weights)
    scaler = GradScaler(device="cuda", enabled=config.mixed_precision)
    
    # -----------------------
    # STAGE 1: Head Only
    # -----------------------
    print("\n[STAGE 1] Training Classifier Head Only")
    optimizer_stage1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=config.stage1_lr, weight_decay=config.weight_decay)
    
    for epoch in range(config.stage1_epochs):
        # Stage 1: Warm-up mask loss while backbone is frozen to safely train CBAM
        current_mask_weight = config.mask_loss_weight * ((epoch + 1) / config.stage1_epochs)
        
        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer_stage1, criterion, scaler, device, config, current_mask_weight)
        val_loss, _, _ = validate(model, val_loader, criterion, device, config, current_mask_weight)
        print(f"Stage 1 - Epoch {epoch+1}/{config.stage1_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mask W: {current_mask_weight:.2f}")
        
    # -----------------------
    # STAGE 2: Fine-Tuning
    # -----------------------
    print("\n[STAGE 2] Unfreezing Backbone for Fine-Tuning")
    model.unfreeze_all()
    optimizer_stage2 = AdamW(model.parameters(), lr=config.stage2_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer_stage2, T_max=config.stage2_epochs, eta_min=config.stage2_min_lr)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(config.stage2_epochs):
        # Stage 2: Mask loss is fully active
        current_mask_weight = config.mask_loss_weight
        
        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer_stage2, criterion, scaler, device, config, current_mask_weight)
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device, config, current_mask_weight)
        
        scheduler.step()
        
        print(f"Stage 2 - Epoch {epoch+1}/{config.stage2_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Mask W: {current_mask_weight:.3f}")
        
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  [*] Saved new best model!")
        else:
            patience_counter += 1
            print(f"  [!] Early stopping counter: {patience_counter} / {config.patience}")
            if patience_counter >= config.patience:
                print(f"  [!] Early stopping triggered. Training stopped at epoch {epoch+1}.")
                break
            
    # Restore best weights
    model.load_state_dict(best_weights)
    print("Training Complete. Restored best validation weights.")
    return model
