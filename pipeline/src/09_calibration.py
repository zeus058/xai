import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, f1_score
from scipy.optimize import minimize

class TemperatureScaling(nn.Module):
    """
    Applies temperature scaling on model logits to calibrate probabilities.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature
        
def find_optimal_threshold(labels, probs, metric='youden'):
    """
    Finds the optimal classification threshold using ROC Curve.
    Metric can be 'youden' or 'f1'.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])
    
    # Ensure threshold is sensible
    optimal_threshold = max(0.1, min(0.9, optimal_threshold))
    
    print(f"[CALIBRATION] Optimal Threshold: {optimal_threshold:.4f}")
        
    return optimal_threshold

def calibrate_model(model, val_loader, device):
    """
    Calibrates the model by fitting a temperature parameter on the validation set.
    """
    model.eval()
    nll_criterion = nn.BCEWithLogitsLoss()
    temperature_model = TemperatureScaling().to(device)
    
    # Collect all logits and labels from validation
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels.to(device).unsqueeze(1))
            
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Optimize temperature
    optimizer = torch.optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_model(all_logits), all_labels)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    
    temp = temperature_model.temperature.item()
    print(f"[CALIBRATION] Optimal Temperature found: {temp:.4f}")
    
    return temp
