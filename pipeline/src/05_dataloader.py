import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def create_dataloaders(train_ds, val_ds, test_ds, config, use_sampler=False):
    """
    Creates DataLoaders with optimized settings for performance.
    """
    # Optional: WeightedRandomSampler instead of focal loss
    train_sampler = None
    if use_sampler and hasattr(train_ds, 'labels'):
        labels = train_ds.labels
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[int(t)] for t in labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        shuffle = False
    else:
        shuffle = True
        
    import os
    use_pin_memory = torch.cuda.is_available()
    num_workers = 0 if os.name == 'nt' else config.num_workers
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader
