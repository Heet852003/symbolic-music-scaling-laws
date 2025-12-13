#!/usr/bin/env python
"""
Generate final music samples from best model.
This is a cross-platform replacement for the bash script generate_final_samples.sh
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Main function to generate samples."""
    print("="*60)
    print("CS-GY 6923: Sample Generation")
    print("="*60)
    print()
    
    # Configuration
    VOCAB_PATH = "data/tokenized/vocab.json"
    MODEL_PATH = "outputs/models/transformer/xl/best_model.pt"
    OUTPUT_DIR = "outputs/samples"
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"[WARNING] Model file not found: {MODEL_PATH}")
        print("  The script will continue, but generation may fail.")
    
    # Check if vocab exists
    if not Path(VOCAB_PATH).exists():
        print(f"[WARNING] Vocabulary file not found: {VOCAB_PATH}")
        print("  The script will continue, but generation may fail.")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating samples from best model (Transformer XL)...")
    print()
    
    # Build command
    cmd = [
        sys.executable,
        "src/evaluation/generate_samples.py",
        "--model_path", MODEL_PATH,
        "--vocab_path", VOCAB_PATH,
        "--model_type", "transformer",
        "--output_dir", OUTPUT_DIR,
        "--num_samples", "10",
        "--max_new_tokens", "500",
        "--temperature", "1.0",
        "--top_k", "50",
        "--top_p", "0.9"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("Sample Generation Complete!")
        print("="*60)
        print(f"Samples saved to: {OUTPUT_DIR}/")
        
        # List generated files
        if output_path.exists():
            sample_files = list(output_path.glob("*"))
            if sample_files:
                print(f"\nGenerated {len(sample_files)} sample files:")
                for f in sample_files[:10]:  # Show first 10
                    if f.is_file():
                        print(f"  - {f.name}")
        
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print("[ERROR] Sample generation failed!")
        print(f"  Return code: {e.returncode}")
        return 1
    except FileNotFoundError:
        print()
        print("[ERROR] Python not found. Make sure Python is in your PATH.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
