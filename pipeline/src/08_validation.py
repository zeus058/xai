import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import brier_score_loss, log_loss

def calculate_metrics(labels, probs, threshold=0.5):
    """
    Computes all standard evaluation metrics required by the Gate 7 protocol.
    """
    preds = (probs >= threshold).astype(int)
    
    auc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0) # Sensitivity
    f1 = f1_score(labels, preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    brier = brier_score_loss(labels, probs)
    logloss = log_loss(labels, probs)
    
    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    
    metrics = {
        "auroc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(acc),
        "precision": float(precision),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "brier_score": float(brier),
        "log_loss": float(logloss),
        "mcc": float(mcc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold_used": float(threshold)
    }
    return metrics
