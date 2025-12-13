"""
Plot training curves for all models.
Creates loss curves over time for transformer and RNN models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_training_stats(model_dir):
    """Load training statistics from a model directory."""
    stats_file = Path(model_dir) / 'training_stats.json'
    if not stats_file.exists():
        return None
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    return stats


def plot_training_curves(transformer_dir, rnn_dir, output_dir):
    """
    Create training curves for all models.
    
    Args:
        transformer_dir: Directory containing transformer model stats
        rnn_dir: Directory containing RNN model stats
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load transformer results
    transformer_curves = []
    transformer_path = Path(transformer_dir)
    
    if transformer_path.exists():
        for model_dir in transformer_path.iterdir():
            if model_dir.is_dir():
                stats = load_training_stats(model_dir)
                if stats and 'train_losses' in stats and 'val_losses' in stats:
                    transformer_curves.append({
                        'name': model_dir.name,
                        'num_params': stats.get('num_params', 0),
                        'train_losses': stats['train_losses'],
                        'val_losses': stats['val_losses'],
                        'architecture': 'Transformer'
                    })
    
    # Load RNN results
    rnn_curves = []
    rnn_path = Path(rnn_dir)
    
    if rnn_path.exists():
        for model_dir in rnn_path.iterdir():
            if model_dir.is_dir():
                stats = load_training_stats(model_dir)
                if stats and 'train_losses' in stats and 'val_losses' in stats:
                    rnn_curves.append({
                        'name': model_dir.name,
                        'num_params': stats.get('num_params', 0),
                        'train_losses': stats['train_losses'],
                        'val_losses': stats['val_losses'],
                        'architecture': 'RNN'
                    })
    
    if len(transformer_curves) == 0 and len(rnn_curves) == 0:
        print("No training statistics found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Transformer training curves
    ax1 = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(transformer_curves)))
    
    for i, curve in enumerate(sorted(transformer_curves, key=lambda x: x['num_params'])):
        epochs = range(1, len(curve['train_losses']) + 1)
        label = f"{curve['name']} ({curve['num_params']/1e6:.1f}M)"
        ax1.plot(epochs, curve['train_losses'], 'o-', color=colors[i], 
                label=label, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Transformer Training Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transformer validation curves
    ax2 = axes[0, 1]
    
    for i, curve in enumerate(sorted(transformer_curves, key=lambda x: x['num_params'])):
        epochs = range(1, len(curve['val_losses']) + 1)
        label = f"{curve['name']} ({curve['num_params']/1e6:.1f}M)"
        ax2.plot(epochs, curve['val_losses'], 's-', color=colors[i], 
                label=label, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Transformer Validation Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RNN training curves
    ax3 = axes[1, 0]
    colors_rnn = plt.cm.Reds(np.linspace(0.4, 0.9, len(rnn_curves)))
    
    for i, curve in enumerate(sorted(rnn_curves, key=lambda x: x['num_params'])):
        epochs = range(1, len(curve['train_losses']) + 1)
        label = f"{curve['name']} ({curve['num_params']/1e6:.1f}M)"
        ax3.plot(epochs, curve['train_losses'], 'o-', color=colors_rnn[i], 
                label=label, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.set_title('RNN Training Curves', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RNN validation curves
    ax4 = axes[1, 1]
    
    for i, curve in enumerate(sorted(rnn_curves, key=lambda x: x['num_params'])):
        epochs = range(1, len(curve['val_losses']) + 1)
        label = f"{curve['name']} ({curve['num_params']/1e6:.1f}M)"
        ax4.plot(epochs, curve['val_losses'], 's-', color=colors_rnn[i], 
                label=label, linewidth=2, markersize=6)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Loss', fontsize=12)
    ax4.set_title('RNN Validation Curves', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined comparison plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot all validation curves together
    for curve in sorted(transformer_curves, key=lambda x: x['num_params']):
        epochs = range(1, len(curve['val_losses']) + 1)
        label = f"Transformer {curve['name']} ({curve['num_params']/1e6:.1f}M)"
        ax.plot(epochs, curve['val_losses'], 'o-', label=label, linewidth=2, alpha=0.8)
    
    for curve in sorted(rnn_curves, key=lambda x: x['num_params']):
        epochs = range(1, len(curve['val_losses']) + 1)
        label = f"RNN {curve['name']} ({curve['num_params']/1e6:.1f}M)"
        ax.plot(epochs, curve['val_losses'], 's--', label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('All Models: Validation Loss Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'all_models_validation_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining curves saved to: {output_path}")
    print(f"  - training_curves.png (individual plots)")
    print(f"  - all_models_validation_curves.png (combined comparison)")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--transformer_dir', type=str, required=True,
                       help='Directory containing transformer model stats')
    parser.add_argument('--rnn_dir', type=str, required=True,
                       help='Directory containing RNN model stats')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_training_curves(args.transformer_dir, args.rnn_dir, args.output_dir)


if __name__ == '__main__':
    main()

