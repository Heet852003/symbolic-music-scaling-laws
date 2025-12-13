"""
Training script for transformer models.
"""

import torch
import torch.nn as nn
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import time
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transformer import create_transformer_model
from src.training.utils import (
    get_dataloader, get_optimizer, get_lr_scheduler,
    estimate_loss, save_checkpoint
)
from src.models.base import count_parameters


def train_epoch(model, train_loader, optimizer, scheduler, device, config, save_step_losses=False):
    """Train for one epoch."""
    model.train()
    losses = []
    step_losses = []  # Per-step losses for detailed tracking
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        loss_val = loss.item()
        losses.append(loss_val)
        if save_step_losses:
            step_losses.append(loss_val)
        
        pbar.set_postfix({'loss': loss_val})
    
    avg_loss = sum(losses) / len(losses)
    return avg_loss, step_losses if save_step_losses else []


def main():
    parser = argparse.ArgumentParser(description='Train transformer model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/tokenized/splits',
                       help='Directory containing tokenized data')
    parser.add_argument('--output_dir', type=str, default='outputs/models/transformer',
                       help='Directory to save models')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary JSON file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load vocabulary
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model_config = config['model']
    model = create_transformer_model(vocab_size, model_config)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {num_params / 1e6:.2f}M")
    
    # Data loaders
    train_loader, train_size = get_dataloader(
        Path(args.data_dir) / 'train',
        block_size=config['training']['block_size'],
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader, val_size = get_dataloader(
        Path(args.data_dir) / 'val',
        block_size=config['training']['block_size'],
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    print(f"Training samples: {train_size:,}")
    print(f"Validation samples: {val_size:,}")
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, config['training'])
    
    num_steps = len(train_loader) * config['training'].get('num_epochs', 1)
    scheduler = get_lr_scheduler(optimizer, config['training'], num_steps)
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = config['training'].get('num_epochs', 1)
    eval_interval = config['training'].get('eval_interval', 1000)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    all_step_losses = []  # Per-step losses across all epochs
    peak_memory_gb = 0
    
    start_time = time.time()
    
    # Track initial memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, step_losses = train_epoch(
            model, train_loader, optimizer, scheduler, device, config['training'],
            save_step_losses=(epoch == 0)  # Save step losses for first epoch
        )
        train_losses.append(train_loss)
        if step_losses:
            all_step_losses.extend(step_losses)
        
        # Track peak memory
        if device.type == 'cuda':
            peak_memory_bytes = torch.cuda.max_memory_allocated(device)
            peak_memory_gb = peak_memory_bytes / (1024 ** 3)
        
        # Evaluate
        val_loss = estimate_loss(model, val_loader, device, eval_iters=100)
        val_losses.append(val_loss)
        
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        if device.type == 'cuda':
            print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch, 0, val_loss, checkpoint_path)
            print(f"Saved best model (val_loss={val_loss:.4f})")
        
        # Save epoch checkpoint
        epoch_path = output_dir / f'epoch_{epoch+1}.pt'
        save_checkpoint(model, optimizer, epoch, 0, val_loss, epoch_path)
    
    elapsed_time = time.time() - start_time
    
    # Save training statistics
    stats = {
        'num_params': num_params,
        'num_epochs': num_epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'training_time': elapsed_time,
        'peak_memory_gb': peak_memory_gb,
        'step_losses': all_step_losses[:1000] if all_step_losses else [],  # Save first 1000 steps
        'config': config
    }
    
    with open(output_dir / 'training_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training time: {elapsed_time / 60:.2f} minutes")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()

