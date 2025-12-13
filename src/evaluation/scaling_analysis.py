"""
Analyze scaling laws for transformer and RNN models.
Creates scaling plots and fits power laws.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy.optimize import curve_fit
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def power_law(N, a, alpha, c):
    """Power law function: L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def fit_scaling_law(param_counts, losses):
    """
    Fit power law to scaling data.
    
    Args:
        param_counts: List of parameter counts
        losses: List of validation losses
    
    Returns:
        Fitted parameters (a, alpha, c) and R-squared
    """
    # Fit power law
    try:
        popt, pcov = curve_fit(
            power_law,
            param_counts,
            losses,
            p0=[1.0, 0.1, 2.0],
            maxfev=10000
        )
        a, alpha, c = popt
        
        # Calculate R-squared
        y_pred = power_law(param_counts, a, alpha, c)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return a, alpha, c, r_squared
    except:
        return None, None, None, None


def plot_scaling_laws(transformer_dir, rnn_dir, output_dir):
    """
    Create scaling plots for transformer and RNN models.
    
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
                stats_file = model_dir / 'training_stats.json'
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        transformer_stats.append({
                            'num_params': stats['num_params'],
                            'val_loss': stats['best_val_loss'],
                            'architecture': 'Transformer'
                        })
    
    # Load RNN results
    rnn_stats = []
    rnn_path = Path(rnn_dir)
    
    if rnn_path.exists():
        for model_dir in rnn_path.iterdir():
            if model_dir.is_dir():
                stats_file = model_dir / 'training_stats.json'
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        rnn_stats.append({
                            'num_params': stats['num_params'],
                            'val_loss': stats['best_val_loss'],
                            'architecture': 'RNN'
                        })
    
    # Combine data
    all_stats = transformer_stats + rnn_stats
    
    if len(all_stats) == 0:
        print("No training statistics found!")
        return
    
    df = pd.DataFrame(all_stats)
    
    # Separate by architecture
    transformer_df = df[df['architecture'] == 'Transformer'].sort_values('num_params') if len(transformer_stats) > 0 else pd.DataFrame()
    rnn_df = df[df['architecture'] == 'RNN'].sort_values('num_params') if len(rnn_stats) > 0 else pd.DataFrame()
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Individual scaling plots
    ax1 = axes[0]
    
    if len(transformer_df) > 0:
        ax1.scatter(transformer_df['num_params'], transformer_df['val_loss'],
                   label='Transformer', s=100, alpha=0.7, color='blue')
        
        # Fit power law for transformer
        if len(transformer_df) >= 3:
            a, alpha, c, r2 = fit_scaling_law(
                transformer_df['num_params'].values,
                transformer_df['val_loss'].values
            )
            if alpha is not None:
                N_fit = np.logspace(
                    np.log10(transformer_df['num_params'].min()),
                    np.log10(transformer_df['num_params'].max()),
                    100
                )
                L_fit = power_law(N_fit, a, alpha, c)
                ax1.plot(N_fit, L_fit, '--', color='blue', alpha=0.7,
                        label=f'Transformer fit (α={alpha:.3f}, R²={r2:.3f})')
    
    if len(rnn_df) > 0:
        ax1.scatter(rnn_df['num_params'], rnn_df['val_loss'],
                   label='RNN', s=100, alpha=0.7, color='red')
        
        # Fit power law for RNN
        if len(rnn_df) >= 3:
            a, alpha, c, r2 = fit_scaling_law(
                rnn_df['num_params'].values,
                rnn_df['val_loss'].values
            )
            if alpha is not None:
                N_fit = np.logspace(
                    np.log10(rnn_df['num_params'].min()),
                    np.log10(rnn_df['num_params'].max()),
                    100
                )
                L_fit = power_law(N_fit, a, alpha, c)
                ax1.plot(N_fit, L_fit, '--', color='red', alpha=0.7,
                        label=f'RNN fit (α={alpha:.3f}, R²={r2:.3f})')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Parameters', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Scaling Laws: Validation Loss vs. Model Size', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparison plot with power law fits
    ax2 = axes[1]
    
    if len(transformer_df) > 0:
        ax2.scatter(transformer_df['num_params'], transformer_df['val_loss'],
                   label='Transformer', s=100, alpha=0.7, color='blue', marker='o')
        
        # Fit power law for transformer
        if len(transformer_df) >= 3:
            a, alpha, c, r2 = fit_scaling_law(
                transformer_df['num_params'].values,
                transformer_df['val_loss'].values
            )
            if alpha is not None:
                N_fit = np.logspace(
                    np.log10(transformer_df['num_params'].min()),
                    np.log10(transformer_df['num_params'].max()),
                    100
                )
                L_fit = power_law(N_fit, a, alpha, c)
                ax2.plot(N_fit, L_fit, '--', color='blue', alpha=0.7, linewidth=2,
                        label=f'Transformer fit (α={alpha:.3f})')
    
    if len(rnn_df) > 0:
        ax2.scatter(rnn_df['num_params'], rnn_df['val_loss'],
                   label='RNN', s=100, alpha=0.7, color='red', marker='s')
        
        # Fit power law for RNN
        if len(rnn_df) >= 3:
            a, alpha, c, r2 = fit_scaling_law(
                rnn_df['num_params'].values,
                rnn_df['val_loss'].values
            )
            if alpha is not None:
                N_fit = np.logspace(
                    np.log10(rnn_df['num_params'].min()),
                    np.log10(rnn_df['num_params'].max()),
                    100
                )
                L_fit = power_law(N_fit, a, alpha, c)
                ax2.plot(N_fit, L_fit, '--', color='red', alpha=0.7, linewidth=2,
                        label=f'RNN fit (α={alpha:.3f})')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Parameters', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Architecture Comparison with Power Law Fits', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scaling_laws.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary = []
    for _, row in df.iterrows():
        summary.append({
            'Architecture': row['architecture'],
            'Parameters': f"{row['num_params']:,}",
            'Val Loss': f"{row['val_loss']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path / 'scaling_summary.csv', index=False)
    
    print(f"\nScaling analysis complete!")
    print(f"Plots saved to: {output_path}")
    print(f"\nSummary:")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Analyze scaling laws')
    parser.add_argument('--transformer_dir', type=str, required=True,
                       help='Directory containing transformer model stats')
    parser.add_argument('--rnn_dir', type=str, required=True,
                       help='Directory containing RNN model stats')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_scaling_laws(args.transformer_dir, args.rnn_dir, args.output_dir)


if __name__ == '__main__':
    main()

