#!/usr/bin/env python
"""
Generate all visualizations for the project report.
This is a cross-platform replacement for the bash script generate_all_visualizations.sh
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(description)
    print('='*60)
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n[OK] Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error in {description}:")
        print(f"  Return code: {e.returncode}")
        print(f"  This may be due to missing data files or models.")
        print(f"  Please ensure the required data/models are available.")
        return False
    except FileNotFoundError as e:
        print(f"\n[ERROR] Command not found - {cmd[0]}")
        print("  Make sure Python is in your PATH")
        return False

def main():
    """Main function to generate all visualizations."""
    print("="*60)
    print("Generating All Visualizations")
    print("="*60)
    
    # Configuration
    DATA_DIR = "data/tokenized/splits"
    VOCAB_PATH = "data/tokenized/vocab.json"
    TRANSFORMER_DIR = "outputs/models/transformer"
    RNN_DIR = "outputs/models/rnn"
    OUTPUT_DIR = "outputs/plots"
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Check for required directories/files and warn if missing
    print("\nChecking for required files/directories...")
    required_paths = {
        "Data directory": DATA_DIR,
        "Vocabulary file": VOCAB_PATH,
        "Transformer models": TRANSFORMER_DIR,
        "RNN models": RNN_DIR
    }
    
    missing_paths = []
    for name, path in required_paths.items():
        if not Path(path).exists():
            print(f"  [WARNING] {name} not found: {path}")
            missing_paths.append(name)
        else:
            print(f"  [OK] {name} found: {path}")
    
    if missing_paths:
        print(f"\n  Note: Some paths are missing. The script will continue,")
        print(f"        but plots requiring these may fail.")
        print(f"  Missing: {', '.join(missing_paths)}")
    
    success_count = 0
    total_count = 4
    
    # 1. Data Statistics
    cmd1 = [
        sys.executable,
        "src/evaluation/plot_data_statistics.py",
        "--data_dir", DATA_DIR,
        "--vocab_path", VOCAB_PATH,
        "--output_dir", OUTPUT_DIR
    ]
    if run_command(cmd1, "1. Generating data statistics visualizations..."):
        success_count += 1
    
    # 2. Scaling Laws
    cmd2 = [
        sys.executable,
        "src/evaluation/scaling_analysis.py",
        "--transformer_dir", TRANSFORMER_DIR,
        "--rnn_dir", RNN_DIR,
        "--output_dir", OUTPUT_DIR
    ]
    if run_command(cmd2, "2. Generating scaling laws plots..."):
        success_count += 1
    
    # 3. Training Curves
    cmd3 = [
        sys.executable,
        "src/evaluation/plot_training_curves.py",
        "--transformer_dir", TRANSFORMER_DIR,
        "--rnn_dir", RNN_DIR,
        "--output_dir", OUTPUT_DIR
    ]
    if run_command(cmd3, "3. Generating training curves..."):
        success_count += 1
    
    # 4. Computational Efficiency
    cmd4 = [
        sys.executable,
        "src/evaluation/plot_computational_efficiency.py",
        "--transformer_dir", TRANSFORMER_DIR,
        "--rnn_dir", RNN_DIR,
        "--output_dir", OUTPUT_DIR
    ]
    if run_command(cmd4, "4. Generating computational efficiency plots..."):
        success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("Visualization Generation Summary")
    print("="*60)
    print(f"Successfully completed: {success_count}/{total_count}")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # List generated files
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        png_files = list(output_path.glob("*.png"))
        csv_files = list(output_path.glob("*.csv"))
        json_files = list(output_path.glob("*.json"))
        
        if png_files:
            print(f"\nGenerated PNG files ({len(png_files)}):")
            for f in png_files:
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")
        
        if csv_files:
            print(f"\nGenerated CSV files ({len(csv_files)}):")
            for f in csv_files:
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")
        
        if json_files:
            print(f"\nGenerated JSON files ({len(json_files)}):")
            for f in json_files:
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")
        
        if not (png_files or csv_files or json_files):
            print("\nNo output files found in the output directory.")
    
    if success_count == total_count:
        print("\n[SUCCESS] All visualizations generated successfully!")
        return 0
    else:
        print(f"\n[WARNING] Some visualizations failed ({total_count - success_count} failed)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
