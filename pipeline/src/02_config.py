import dataclasses
from typing import List, Tuple

@dataclasses.dataclass
class PipelineConfig:
    # Environment & Seeds
    seeds: List[int] = dataclasses.field(default_factory=lambda: [3407, 42, 2024])
    
    # Dataset & Dataloader
    batch_size: int = 32
    num_workers: int = 2
    input_resolution: int = 256  # configurable to 224, 320, 384
    
    # Model configuration
    models_to_train: List[str] = dataclasses.field(default_factory=lambda: ["DenseNet121", "ResNet50"])
    densenet_freeze_percent: float = 0.75
    resnet_freeze_percent: float = 0.80
    
    # Early Stopping & Checkpoints
    patience: int = 5
    min_delta: float = 1e-4
    skip_training_if_checkpoint_exists: bool = True  # Automatically loads existing .pt checkpoints to skip training and execute XAI directly
    
    # Training (Stage 1 - Head Only)
    stage1_epochs: int = 5
    stage1_lr: float = 1e-3
    
    # Training (Stage 2 - Fine tuning)
    stage2_epochs: int = 25
    stage2_lr: float = 1e-4
    stage2_min_lr: float = 1e-6
    weight_decay: float = 1e-4
    
    # Optimization
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clipping_max_norm: float = 1.0
    
    # Data Augmentation (Albumentations)
    apply_clahe: bool = True
    
    # Thresholding & TTA
    use_tta: bool = True
    
    # XAI
    explainability_methods: List[str] = dataclasses.field(default_factory=lambda: ["Grad-CAM", "Guided Grad-CAM"])

    # Loss Configuration
    loss_type: str = "Focal" # Options: "BCE", "Focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Architecture Enhancements
    use_cbam: bool = True
    
    # Auxiliary Loss (Mask-Guided Attention)
    use_mask_loss: bool = True
    mask_loss_weight: float = 0.1
    
# Global Configuration instance
CONFIG = PipelineConfig()
