"""
Plot computational efficiency metrics.
Creates plots for training time and memory usage vs model size.
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


def load_model_stats(model_dir):
    """Load statistics from a model directory."""
    stats_file = Path(model_dir) / 'training_stats.json'
    if not stats_file.exists():
        return None
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    return stats


def plot_computational_efficiency(transformer_dir, rnn_dir, output_dir):
    """
    Create computational efficiency plots.
    
    Args:
        transformer_dir: Directory containing transformer model stats
        rnn_dir: Directory containing RNN model stats
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load transformer results
    transformer_stats = []
    transformer_path = Path(transformer_dir)
    
    if transformer_path.exists():
        for model_dir in transformer_path.iterdir():
            if model_dir.is_dir():
                stats = load_model_stats(model_dir)
                if stats:
                    transformer_stats.append({
                        'num_params': stats.get('num_params', 0),
                        'training_time': stats.get('training_time', 0),  # in seconds
                        'memory_usage': stats.get('peak_memory_gb', 0),  # in GB
                        'architecture': 'Transformer',
                        'model_name': model_dir.name
                    })
    
    # Load RNN results
    rnn_stats = []
    rnn_path = Path(rnn_dir)
    
    if rnn_path.exists():
        for model_dir in rnn_path.iterdir():
            if model_dir.is_dir():
                stats = load_model_stats(model_dir)
                if stats:
                    rnn_stats.append({
                        'num_params': stats.get('num_params', 0),
                        'training_time': stats.get('training_time', 0),  # in seconds
                        'memory_usage': stats.get('peak_memory_gb', 0),  # in GB
                        'architecture': 'RNN',
                        'model_name': model_dir.name
                    })
    
    if len(transformer_stats) == 0 and len(rnn_stats) == 0:
        print("No training statistics found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to DataFrames
    transformer_df = pd.DataFrame(transformer_stats) if transformer_stats else pd.DataFrame()
    rnn_df = pd.DataFrame(rnn_stats) if rnn_stats else pd.DataFrame()
    
    # Plot 1: Training time vs parameters
    ax1 = axes[0, 0]
    
    if len(transformer_df) > 0:
        transformer_df = transformer_df.sort_values('num_params')
        ax1.scatter(transformer_df['num_params'], transformer_df['training_time'] / 3600,
                   s=150, alpha=0.7, color='blue', label='Transformer', marker='o')
        ax1.plot(transformer_df['num_params'], transformer_df['training_time'] / 3600,
                '--', color='blue', alpha=0.5)
    
    if len(rnn_df) > 0:
        rnn_df = rnn_df.sort_values('num_params')
        ax1.scatter(rnn_df['num_params'], rnn_df['training_time'] / 3600,
                   s=150, alpha=0.7, color='red', label='RNN', marker='s')
        ax1.plot(rnn_df['num_params'], rnn_df['training_time'] / 3600,
                '--', color='red', alpha=0.5)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Parameters', fontsize=12)
    ax1.set_ylabel('Training Time (hours)', fontsize=12)
    ax1.set_title('Training Time vs Model Size', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory usage vs parameters
    ax2 = axes[0, 1]
    
    if len(transformer_df) > 0 and 'memory_usage' in transformer_df.columns:
        ax2.scatter(transformer_df['num_params'], transformer_df['memory_usage'],
                   s=150, alpha=0.7, color='blue', label='Transformer', marker='o')
        ax2.plot(transformer_df['num_params'], transformer_df['memory_usage'],
                '--', color='blue', alpha=0.5)
    
    if len(rnn_df) > 0 and 'memory_usage' in rnn_df.columns:
        ax2.scatter(rnn_df['num_params'], rnn_df['memory_usage'],
                   s=150, alpha=0.7, color='red', label='RNN', marker='s')
        ax2.plot(rnn_df['num_params'], rnn_df['memory_usage'],
                '--', color='red', alpha=0.5)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Parameters', fontsize=12)
    ax2.set_ylabel('Peak GPU Memory (GB)', fontsize=12)
    ax2.set_title('Memory Usage vs Model Size', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training time per parameter
    ax3 = axes[1, 0]
    
    if len(transformer_df) > 0:
        transformer_df['time_per_param'] = transformer_df['training_time'] / transformer_df['num_params']
        ax3.scatter(transformer_df['num_params'], transformer_df['time_per_param'] * 1e6,
                   s=150, alpha=0.7, color='blue', label='Transformer', marker='o')
        ax3.plot(transformer_df['num_params'], transformer_df['time_per_param'] * 1e6,
                '--', color='blue', alpha=0.5)
    
    if len(rnn_df) > 0:
        rnn_df['time_per_param'] = rnn_df['training_time'] / rnn_df['num_params']
        ax3.scatter(rnn_df['num_params'], rnn_df['time_per_param'] * 1e6,
                   s=150, alpha=0.7, color='red', label='RNN', marker='s')
        ax3.plot(rnn_df['num_params'], rnn_df['time_per_param'] * 1e6,
                '--', color='red', alpha=0.5)
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Number of Parameters', fontsize=12)
    ax3.set_ylabel('Training Time per Parameter (μs)', fontsize=12)
    ax3.set_title('Training Efficiency: Time per Parameter', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Memory per parameter
    ax4 = axes[1, 1]
    
    if len(transformer_df) > 0 and 'memory_usage' in transformer_df.columns:
        transformer_df['memory_per_param'] = transformer_df['memory_usage'] / transformer_df['num_params']
        ax4.scatter(transformer_df['num_params'], transformer_df['memory_per_param'] * 1e9,
                   s=150, alpha=0.7, color='blue', label='Transformer', marker='o')
        ax4.plot(transformer_df['num_params'], transformer_df['memory_per_param'] * 1e9,
                '--', color='blue', alpha=0.5)
    
    if len(rnn_df) > 0 and 'memory_usage' in rnn_df.columns:
        rnn_df['memory_per_param'] = rnn_df['memory_usage'] / rnn_df['num_params']
        ax4.scatter(rnn_df['num_params'], rnn_df['memory_per_param'] * 1e9,
                   s=150, alpha=0.7, color='red', label='RNN', marker='s')
        ax4.plot(rnn_df['num_params'], rnn_df['memory_per_param'] * 1e9,
                '--', color='red', alpha=0.5)
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Number of Parameters', fontsize=12)
    ax4.set_ylabel('Memory per Parameter (bytes)', fontsize=12)
    ax4.set_title('Memory Efficiency: Memory per Parameter', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'computational_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    all_stats = transformer_stats + rnn_stats
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_df['training_time_hours'] = summary_df['training_time'] / 3600
        summary_df['num_params_millions'] = summary_df['num_params'] / 1e6
        
        summary_table = summary_df[['architecture', 'model_name', 'num_params_millions', 
                                   'training_time_hours', 'memory_usage']].copy()
        summary_table.columns = ['Architecture', 'Model', 'Params (M)', 
                                'Time (hours)', 'Memory (GB)']
        summary_table = summary_table.round(2)
        
        summary_table.to_csv(output_path / 'computational_efficiency_summary.csv', index=False)
        
        print(f"\nComputational efficiency analysis complete!")
        print(f"Plots saved to: {output_path}")
        print(f"\nSummary:")
        print(summary_table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Plot computational efficiency')
    parser.add_argument('--transformer_dir', type=str, required=True,
                       help='Directory containing transformer model stats')
    parser.add_argument('--rnn_dir', type=str, required=True,
                       help='Directory containing RNN model stats')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_computational_efficiency(args.transformer_dir, args.rnn_dir, args.output_dir)


if __name__ == '__main__':
    main()

