#!/usr/bin/env python
"""
Run complete data preprocessing pipeline.
This is a cross-platform replacement for the bash script run_preprocessing.sh
"""

import subprocess
import sys
from pathlib import Path

def run_step(description, cmd):
    """Run a preprocessing step and handle errors."""
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
    except FileNotFoundError:
        print(f"\n[ERROR] Python not found. Make sure Python is in your PATH.")
        return False

def main():
    """Main function to run preprocessing pipeline."""
    print("="*60)
    print("CS-GY 6923: Data Preprocessing Pipeline")
    print("="*60)
    print()
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Download Lakh MIDI dataset
    cmd1 = [
        sys.executable,
        "src/preprocessing/download_data.py",
        "--output_dir", "data/raw"
    ]
    if run_step("Step 1: Downloading Lakh MIDI dataset...", cmd1):
        success_count += 1
    print()
    
    # Step 2: Convert MIDI to ABC notation
    cmd2 = [
        sys.executable,
        "src/preprocessing/midi_to_abc.py",
        "--input_dir", "data/raw/lmd_full",
        "--output_dir", "data/processed",
        "--min_length", "50",
        "--max_length", "10000"
    ]
    if run_step("Step 2: Converting MIDI files to ABC notation...", cmd2):
        success_count += 1
    print()
    
    # Step 3: Tokenize ABC files
    cmd3 = [
        sys.executable,
        "src/preprocessing/tokenize.py",
        "--input_dir", "data/processed",
        "--output_dir", "data/tokenized",
        "--tokenization_type", "character",
        "--vocab_path", "data/tokenized/vocab.json"
    ]
    if run_step("Step 3: Tokenizing ABC files...", cmd3):
        success_count += 1
    print()
    
    # Step 4: Split data into train/val/test
    cmd4 = [
        sys.executable,
        "src/preprocessing/split_data.py",
        "--input_dir", "data/tokenized",
        "--output_dir", "data/tokenized/splits",
        "--train_ratio", "0.98",
        "--val_ratio", "0.01",
        "--test_ratio", "0.01",
        "--min_tokens", "100000000"
    ]
    if run_step("Step 4: Splitting data into train/val/test sets...", cmd4):
        success_count += 1
    
    # Summary
    print()
    print("="*60)
    print("Preprocessing Pipeline Summary")
    print("="*60)
    print(f"Successfully completed: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print()
        print("="*60)
        print("Preprocessing Complete!")
        print("="*60)
        print("Data ready for training at: data/tokenized/splits")
        print("Vocabulary saved at: data/tokenized/vocab.json")
        return 0
    else:
        print(f"\n[WARNING] Some steps failed ({total_steps - success_count} failed)")
        print("Please check the errors above and retry.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
