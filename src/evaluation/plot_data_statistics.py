"""
Plot data statistics and visualizations.
Creates plots for sequence length distribution, vocabulary statistics, etc.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import Counter
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def analyze_sequence_lengths(data_dir):
    """Analyze sequence length distribution in tokenized data."""
    data_path = Path(data_dir)
    sequence_lengths = []
    total_tokens = 0
    total_files = 0
    
    # Check for tokenized files
    token_files = list(data_path.rglob('*.tokens'))
    
    if len(token_files) == 0:
        # Try reading from split directories
        for split in ['train', 'val', 'test']:
            split_path = data_path / split
            if split_path.exists():
                token_files.extend(list(split_path.rglob('*.tokens')))
    
    print(f"Found {len(token_files)} tokenized files")
    
    for token_file in token_files:
        try:
            with open(token_file, 'r') as f:
                tokens = f.read().strip().split()
                length = len(tokens)
                sequence_lengths.append(length)
                total_tokens += length
                total_files += 1
        except Exception as e:
            continue
    
    return {
        'lengths': sequence_lengths,
        'total_tokens': total_tokens,
        'total_files': total_files,
        'mean': np.mean(sequence_lengths) if sequence_lengths else 0,
        'median': np.median(sequence_lengths) if sequence_lengths else 0,
        'std': np.std(sequence_lengths) if sequence_lengths else 0,
        'min': np.min(sequence_lengths) if sequence_lengths else 0,
        'max': np.max(sequence_lengths) if sequence_lengths else 0
    }


def analyze_vocabulary(vocab_path):
    """Analyze vocabulary statistics."""
    vocab_path_obj = Path(vocab_path)
    if not vocab_path_obj.exists():
        print(f"Warning: Vocabulary file not found: {vocab_path}")
        print("  Continuing with default vocabulary statistics...")
        return {
            'vocab_size': 0,
            'vocab': {},
            'token_counts': {}
        }
    
    try:
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        vocab = vocab_data.get('vocab', {})
        vocab_size = vocab_data.get('vocab_size', len(vocab))
        
        # Get token frequencies if available
        token_counts = {}
        if 'token_counts' in vocab_data:
            token_counts = vocab_data['token_counts']
        
        return {
            'vocab_size': vocab_size,
            'vocab': vocab,
            'token_counts': token_counts
        }
    except Exception as e:
        print(f"Error reading vocabulary file: {e}")
        print("  Continuing with default vocabulary statistics...")
        return {
            'vocab_size': 0,
            'vocab': {},
            'token_counts': {}
        }


def plot_data_statistics(data_dir, vocab_path, output_dir):
    """
    Create data statistics visualizations.
    
    Args:
        data_dir: Directory containing tokenized data
        vocab_path: Path to vocabulary JSON file
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Analyze sequence lengths
    print("Analyzing sequence lengths...")
    seq_stats = analyze_sequence_lengths(data_dir)
    
    # Analyze vocabulary
    print("Analyzing vocabulary...")
    vocab_stats = analyze_vocabulary(vocab_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Sequence length distribution (histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    if seq_stats['lengths']:
        lengths = seq_stats['lengths']
        # Filter outliers for better visualization
        q99 = np.percentile(lengths, 99)
        filtered_lengths = [l for l in lengths if l <= q99]
        
        ax1.hist(filtered_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(seq_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {seq_stats["mean"]:.1f}')
        ax1.axvline(seq_stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {seq_stats["median"]:.1f}')
        ax1.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No sequence data available.\nPlease run preprocessing pipeline first.', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
    
    # Plot 2: Sequence length statistics (box plot by split)
    ax2 = fig.add_subplot(gs[0, 1])
    if seq_stats['lengths']:
        # Create box plot
        bp = ax2.boxplot([seq_stats['lengths']], labels=['All Sequences'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('Sequence Length (tokens)', fontsize=12)
        ax2.set_title('Sequence Length Statistics', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"Mean: {seq_stats['mean']:.1f}\n"
        stats_text += f"Median: {seq_stats['median']:.1f}\n"
        stats_text += f"Std: {seq_stats['std']:.1f}\n"
        stats_text += f"Min: {seq_stats['min']}\n"
        stats_text += f"Max: {seq_stats['max']}"
        ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No sequence data available.', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title('Sequence Length Statistics', fontsize=14, fontweight='bold')
    
    # Plot 3: Vocabulary size and token distribution
    ax3 = fig.add_subplot(gs[1, 0])
    vocab_size = vocab_stats.get('vocab_size', 0)
    if vocab_size > 0:
        ax3.barh([0], [vocab_size], color='coral', height=0.5)
        ax3.set_xlabel('Vocabulary Size', fontsize=12)
        ax3.set_title(f'Vocabulary Size: {vocab_size:,} tokens', fontsize=14, fontweight='bold')
        ax3.set_yticks([])
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add text annotation
        ax3.text(vocab_size/2, 0, f'{vocab_size:,}', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
    else:
        ax3.text(0.5, 0.5, 'Vocabulary file not found.\nPlease run preprocessing pipeline first.', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.set_title('Vocabulary Size: Not Available', fontsize=14, fontweight='bold')
        ax3.set_yticks([])
        ax3.set_xticks([])
    
    # Plot 4: Top tokens (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    token_counts = vocab_stats.get('token_counts', {})
    if token_counts:
        # Get top 20 tokens
        sorted_tokens = sorted(token_counts.items(), 
                             key=lambda x: x[1], reverse=True)[:20]
        tokens, counts = zip(*sorted_tokens)
        
        ax4.barh(range(len(tokens)), counts, color='teal', alpha=0.7)
        ax4.set_yticks(range(len(tokens)))
        ax4.set_yticklabels(tokens, fontsize=9)
        ax4.set_xlabel('Frequency', fontsize=12)
        ax4.set_title('Top 20 Most Frequent Tokens', fontsize=14, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'Token frequency data not available.\nVocabulary file may be missing.', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title('Top Tokens', fontsize=14, fontweight='bold')
    
    # Plot 5: Dataset statistics summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    vocab_size = vocab_stats.get('vocab_size', 0)
    summary_data = [
        ['Metric', 'Value'],
        ['Total Files', f"{seq_stats['total_files']:,}" if seq_stats['total_files'] > 0 else 'N/A'],
        ['Total Tokens', f"{seq_stats['total_tokens']:,}" if seq_stats['total_tokens'] > 0 else 'N/A'],
        ['Mean Sequence Length', f"{seq_stats['mean']:.1f}" if seq_stats['total_files'] > 0 else 'N/A'],
        ['Median Sequence Length', f"{seq_stats['median']:.1f}" if seq_stats['total_files'] > 0 else 'N/A'],
        ['Std Sequence Length', f"{seq_stats['std']:.1f}" if seq_stats['total_files'] > 0 else 'N/A'],
        ['Min Sequence Length', f"{seq_stats['min']}" if seq_stats['total_files'] > 0 else 'N/A'],
        ['Max Sequence Length', f"{seq_stats['max']}" if seq_stats['total_files'] > 0 else 'N/A'],
        ['Vocabulary Size', f"{vocab_size:,}" if vocab_size > 0 else 'N/A'],
        ['Avg Tokens per File', f"{seq_stats['total_tokens']/max(seq_stats['total_files'], 1):.1f}" if seq_stats['total_files'] > 0 else 'N/A']
    ]
    
    table = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='left', loc='center', colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Dataset Statistics Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle('Data Statistics and Visualizations', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(output_path / 'data_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics to JSON
    stats_output = {
        'sequence_statistics': {
            'total_files': seq_stats['total_files'],
            'total_tokens': seq_stats['total_tokens'],
            'mean': float(seq_stats['mean']) if seq_stats['total_files'] > 0 else None,
            'median': float(seq_stats['median']) if seq_stats['total_files'] > 0 else None,
            'std': float(seq_stats['std']) if seq_stats['total_files'] > 0 else None,
            'min': int(seq_stats['min']) if seq_stats['total_files'] > 0 else None,
            'max': int(seq_stats['max']) if seq_stats['total_files'] > 0 else None
        },
        'vocabulary_statistics': {
            'vocab_size': vocab_size if vocab_size > 0 else None
        },
        'status': {
            'data_available': seq_stats['total_files'] > 0,
            'vocab_available': vocab_size > 0
        }
    }
    
    with open(output_path / 'data_statistics.json', 'w') as f:
        json.dump(stats_output, f, indent=2)
    
    print(f"\nData statistics saved to: {output_path}")
    print(f"  - data_statistics.png")
    print(f"  - data_statistics.json")


def main():
    parser = argparse.ArgumentParser(description='Plot data statistics')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing tokenized data')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_data_statistics(args.data_dir, args.vocab_path, args.output_dir)


if __name__ == '__main__':
    main()

