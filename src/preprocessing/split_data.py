"""
Split tokenized data into train/validation/test sets.
"""

import os
from pathlib import Path
import argparse
import random
from tqdm import tqdm


def split_data(input_dir, output_dir, train_ratio=0.98, val_ratio=0.01, test_ratio=0.01, min_tokens=100000000):
    """
    Split tokenized data into train/val/test sets.
    
    Args:
        input_dir: Directory containing tokenized files
        output_dir: Directory to save split data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        min_tokens: Minimum tokens required in training set
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Find all tokenized files
    token_files = list(input_path.rglob('*.tokens'))
    print(f"Found {len(token_files)} tokenized files")
    
    # Shuffle files
    random.seed(42)
    random.shuffle(token_files)
    
    # Count tokens in each file
    file_token_counts = []
    for token_file in tqdm(token_files, desc="Counting tokens"):
        with open(token_file, 'r') as f:
            tokens = f.read().split()
            file_token_counts.append((token_file, len(tokens)))
    
    # Sort by token count (largest first) for better distribution
    file_token_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Split files
    total_tokens = sum(count for _, count in file_token_counts)
    train_tokens = 0
    val_tokens = 0
    test_tokens = 0
    
    train_files = []
    val_files = []
    test_files = []
    
    for token_file, count in file_token_counts:
        current_train_ratio = train_tokens / total_tokens if total_tokens > 0 else 0
        current_val_ratio = val_tokens / total_tokens if total_tokens > 0 else 0
        
        if current_train_ratio < train_ratio:
            train_files.append(token_file)
            train_tokens += count
        elif current_val_ratio < val_ratio:
            val_files.append(token_file)
            val_tokens += count
        else:
            test_files.append(token_file)
            test_tokens += count
    
    # Verify minimum training tokens
    if train_tokens < min_tokens:
        print(f"Warning: Training set has {train_tokens:,} tokens, but {min_tokens:,} required")
        print("Consider using more data or adjusting split ratios")
    
    # Create output directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective directories
    def copy_files(files, dest_dir):
        for src_file in tqdm(files, desc=f"Copying to {dest_dir.name}"):
            dest_file = dest_dir / src_file.name
            with open(src_file, 'r') as src, open(dest_file, 'w') as dest:
                dest.write(src.read())
    
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    # Print statistics
    print(f"\nData split complete:")
    print(f"  Training set: {len(train_files)} files, {train_tokens:,} tokens ({train_tokens/total_tokens*100:.2f}%)")
    print(f"  Validation set: {len(val_files)} files, {val_tokens:,} tokens ({val_tokens/total_tokens*100:.2f}%)")
    print(f"  Test set: {len(test_files)} files, {test_tokens:,} tokens ({test_tokens/total_tokens*100:.2f}%)")
    print(f"  Total: {len(token_files)} files, {total_tokens:,} tokens")
    
    # Save split info
    split_info = {
        'train_files': len(train_files),
        'train_tokens': train_tokens,
        'val_files': len(val_files),
        'val_tokens': val_tokens,
        'test_files': len(test_files),
        'test_tokens': test_tokens,
        'total_files': len(token_files),
        'total_tokens': total_tokens
    }
    
    with open(output_path / 'split_info.json', 'w') as f:
        import json
        json.dump(split_info, f, indent=2)
    
    return split_info


def main():
    parser = argparse.ArgumentParser(description='Split tokenized data into train/val/test')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing tokenized files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save split data')
    parser.add_argument('--train_ratio', type=float, default=0.98,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.01,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.01,
                       help='Test set ratio')
    parser.add_argument('--min_tokens', type=int, default=100000000,
                       help='Minimum tokens in training set')
    
    args = parser.parse_args()
    
    split_data(
        args.input_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.min_tokens
    )


if __name__ == '__main__':
    main()

