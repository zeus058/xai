import os
import random
import numpy as np
import torch
from pathlib import Path

def setup_environment(seed: int = 3407):
    """
    Sets deterministic seed for all frameworks and auto-detects Kaggle environment.
    """
    # 1. Deterministic Seeding
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN Determinism (sacrifices some speed for perfect reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 for Ampere (L4) / Turing (T4) GPUs if available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"[ENV] Set deterministic seed to {seed}")
    
    # 2. Auto-detect environment paths
    if Path("/content").exists():
        # Google Colab Environment
        is_kaggle = False
        input_dir = Path("/content/input")
        # Save results permanently to Google Drive to prevent data loss
        work_dir = Path("/content/drive/MyDrive/q1_rebuild_outputs")
        print("[ENV] Auto-detected Google Colab Environment.")
        input_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    elif Path("/kaggle/input").exists():
        # Kaggle Environment
        is_kaggle = True
        input_dir = Path("/kaggle/input")
        work_dir = Path("/kaggle/working")
        print("[ENV] Auto-detected Kaggle Environment.")
    else:
        # Local Environment
        is_kaggle = False
        if Path("./input").exists():
            input_dir = Path("./input")
        elif Path("../input").exists():
            input_dir = Path("../input")
        elif Path("D:/XAI/q1_rebuild/data/kaggle_mock_input").exists():
            input_dir = Path("D:/XAI/q1_rebuild/data/kaggle_mock_input")
        else:
            input_dir = Path("./input")
            
        work_dir = Path("./outputs")
        print(f"[ENV] Auto-detected Local Environment (input_dir={input_dir}, work_dir={work_dir}).")
        
        # Ensure local dirs exist
        input_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    
    return is_kaggle, input_dir, work_dir

def get_device():
    """Returns the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[ENV] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[ENV] WARNING: GPU not found. Falling back to CPU.")
    return device
