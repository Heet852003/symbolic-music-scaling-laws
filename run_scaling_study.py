#!/usr/bin/env python
"""
Run complete scaling study for transformers and RNNs.
This is a cross-platform replacement for the bash script run_scaling_study.sh
"""

import subprocess
import sys
from pathlib import Path

def run_training(description, cmd):
    """Run a training command and handle errors."""
    print("="*60)
    print(description)
    print("="*60)
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed!")
        print(f"  Return code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {description} was interrupted by user")
        return False
    except FileNotFoundError:
        print(f"\n[ERROR] Python not found. Make sure Python is in your PATH.")
        return False

def main():
    """Main function to run scaling study."""
    print("="*60)
    print("CS-GY 6923: Scaling Study")
    print("="*60)
    print()
    
    VOCAB_PATH = "data/tokenized/vocab.json"
    DATA_DIR = "data/tokenized/splits"
    
    # Check prerequisites
    if not Path(VOCAB_PATH).exists():
        print(f"[ERROR] Vocabulary file not found: {VOCAB_PATH}")
        print("Please run preprocessing first.")
        return 1
    
    if not Path(DATA_DIR).exists():
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        print("Please run preprocessing first.")
        return 1
    
    success_count = 0
    total_models = 10  # 5 transformers + 4 RNNs + 1 analysis
    
    # Train transformer models
    print("Training Transformer Models...")
    print()
    
    transformer_models = [
        ("1. Training Tiny Transformer (~1M params)...", "transformer_tiny.yaml", "tiny"),
        ("2. Training Small Transformer (~5M params)...", "transformer_small.yaml", "small"),
        ("3. Training Medium Transformer (~20M params)...", "transformer_medium.yaml", "medium"),
        ("4. Training Large Transformer (~50M params)...", "transformer_large.yaml", "large"),
        ("5. Training XL Transformer (~100M+ params)...", "transformer_xl.yaml", "xl")
    ]
    
    for desc, config, size in transformer_models:
        cmd = [
            sys.executable,
            "src/training/train_transformer.py",
            "--config", f"src/configs/{config}",
            "--data_dir", DATA_DIR,
            "--vocab_path", VOCAB_PATH,
            "--output_dir", f"outputs/models/transformer/{size}"
        ]
        if run_training(desc, cmd):
            success_count += 1
        print()
    
    # Train RNN models
    print("Training RNN Models...")
    print()
    
    rnn_sizes = [
        ("1. Training Tiny RNN...", "tiny"),
        ("2. Training Small RNN...", "small"),
        ("3. Training Medium RNN...", "medium"),
        ("4. Training Large RNN...", "large")
    ]
    
    for desc, size in rnn_sizes:
        cmd = [
            sys.executable,
            "src/training/train_rnn.py",
            "--config", "src/configs/rnn_configs.yaml",
            "--data_dir", DATA_DIR,
            "--vocab_path", VOCAB_PATH,
            "--output_dir", "outputs/models/rnn",
            "--model_size", size
        ]
        if run_training(desc, cmd):
            success_count += 1
        print()
    
    # Analyze scaling laws
    print("Analyzing Scaling Laws...")
    cmd_analysis = [
        sys.executable,
        "src/evaluation/scaling_analysis.py",
        "--transformer_dir", "outputs/models/transformer",
        "--rnn_dir", "outputs/models/rnn",
        "--output_dir", "outputs/plots"
    ]
    if run_training("Analyzing Scaling Laws...", cmd_analysis):
        success_count += 1
    
    # Summary
    print()
    print("="*60)
    print("Scaling Study Summary")
    print("="*60)
    print(f"Successfully completed: {success_count}/{total_models}")
    
    if success_count == total_models:
        print()
        print("="*60)
        print("Scaling Study Complete!")
        print("="*60)
        print("Results saved to: outputs/plots/")
        return 0
    else:
        print(f"\n[WARNING] Some training/analysis failed ({total_models - success_count} failed)")
        print("Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
