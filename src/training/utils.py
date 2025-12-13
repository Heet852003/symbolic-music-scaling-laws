"""
Training utilities: data loading, optimization, etc.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import json


class MusicDataset(Dataset):
    """Dataset for tokenized music data."""
    
    def __init__(self, data_dir, block_size=1024):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing tokenized files
            block_size: Context window size
        """
        self.block_size = block_size
        self.data_dir = Path(data_dir)
        
        # Load all tokenized files
        token_files = list(self.data_dir.rglob('*.tokens'))
        self.data = []
        
        print(f"Loading {len(token_files)} tokenized files...")
        for token_file in token_files:
            with open(token_file, 'r') as f:
                tokens = [int(t) for t in f.read().split()]
                if len(tokens) >= block_size:
                    self.data.extend(tokens)
        
        print(f"Loaded {len(self.data):,} tokens")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y


def get_dataloader(data_dir, block_size=1024, batch_size=64, shuffle=True, num_workers=0):
    """Create a DataLoader for music data."""
    dataset = MusicDataset(data_dir, block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, len(dataset)


def get_optimizer(model, config):
    """Create optimizer from configuration."""
    optimizer_type = config.get('optimizer', 'adamw')
    lr = config.get('learning_rate', 3e-4)
    weight_decay = config.get('weight_decay', 0.1)
    
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def get_lr_scheduler(optimizer, config, num_steps):
    """Create learning rate scheduler."""
    scheduler_type = config.get('scheduler', 'cosine')
    warmup_steps = config.get('warmup_steps', 1000)
    
    if scheduler_type == 'cosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'constant':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


@torch.no_grad()
def estimate_loss(model, dataloader, device, eval_iters=100):
    """Estimate validation/test loss."""
    model.eval()
    losses = []
    
    for i, (x, y) in enumerate(dataloader):
        if i >= eval_iters:
            break
        
        x, y = x.to(device), y.to(device)
        if hasattr(model, 'forward'):
            _, loss = model(x, y)
        else:
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, epoch, step, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

