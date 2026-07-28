import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXRayDataset(Dataset):
    def __init__(self, df, image_map: dict, config, split: str = 'train', mask_dir: Path = None):
        """
        df: DataFrame containing dataset info.
        image_map: Dictionary mapping filename to absolute Path.
        split: 'train', 'val', or 'test'
        mask_dir: Optional path to lung masks directory.
        """
        self.df = df
        self.image_map = image_map
        self.config = config
        self.split = split
        self.mask_dir = mask_dir
        
        # Determine image path column
        if 'archive_member' in df.columns:
            self.image_filenames = df['archive_member'].apply(lambda x: Path(x).name).values
        elif 'source_relative_path' in df.columns:
            self.image_filenames = df['source_relative_path'].apply(lambda x: Path(x).name).values
        elif 'image' in df.columns:
            self.image_filenames = df['image'].apply(lambda x: Path(x).name).values
        else:
            raise KeyError("DataFrame must contain 'archive_member', 'source_relative_path', or 'image' column")
            
        # Determine label column
        if 'label_index' in df.columns:
            self.labels = df['label_index'].values
        elif 'label' in df.columns:
            if df['label'].dtype == object:
                self.labels = (df['label'].str.strip().str.upper() == 'PNEUMONIA').astype(int).values
            else:
                self.labels = df['label'].values
        else:
            self.labels = np.zeros(len(df)) # Fallback if no labels
        
        self.transform = self._get_transforms()
        
        # CLAHE initialization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def _get_transforms(self):
        res = self.config.input_resolution
        
        if self.split == 'train':
            return A.Compose([
                A.Resize(res, res),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
                A.GaussNoise(p=0.2),
                A.MotionBlur(p=0.2),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, res//10), hole_width_range=(1, res//10), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(res, res),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        
        if img_name not in self.image_map:
            raise FileNotFoundError(f"Image {img_name} not found in the provided dataset folders.")
            
        img_path = self.image_map[img_name]
        
        # Branch A: Raw Image Reading (Grayscale to match X-ray origin, then to RGB once)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"OpenCV could not read {img_path}")
            
        # Branch B: CLAHE Preprocessing
        if self.config.apply_clahe:
            image = self.clahe.apply(image)
            
        # Convert to RGB exactly once
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Load mask if needed
        mask = None
        if self.config.use_mask_loss and self.mask_dir is not None:
            mask_path = self.mask_dir / img_name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply Albumentations
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0) # [1, H, W]
        else:
            augmented = self.transform(image=image)
            image_tensor = augmented['image']
            mask_tensor = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]))
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image_tensor, label, img_name, mask_tensor

def get_class_weights(df):
    """Calculates class weights for imbalanced datasets."""
    # Support both label_index (int) and label (string) columns
    if 'label_index' in df.columns:
        label_series = df['label_index'].astype(int)
    elif 'label' in df.columns:
        if df['label'].dtype == object:
            label_series = (df['label'].str.strip().str.upper() == 'PNEUMONIA').astype(int)
        else:
            label_series = df['label'].astype(int)
    else:
        return torch.tensor([1.0], dtype=torch.float32)
    
    pos = (label_series == 1).sum()
    neg = (label_series == 0).sum()
    total = pos + neg
    if pos == 0 or neg == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    weight_for_1 = (1 / pos) * (total / 2.0)
    return torch.tensor([weight_for_1], dtype=torch.float32)  # pos_weight for BCEWithLogitsLoss
