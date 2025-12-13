"""
Convert MIDI files to ABC notation using music21.
"""

import os
from pathlib import Path
from tqdm import tqdm
import argparse
from music21 import converter, stream, note, chord, meter, key
import warnings
warnings.filterwarnings('ignore')


def midi_to_abc(midi_path, output_path=None):
    """
    Convert a MIDI file to ABC notation.
    
    Args:
        midi_path: Path to MIDI file
        output_path: Path to save ABC file (optional)
    
    Returns:
        ABC notation string or None if conversion fails
    """
    try:
        # Load MIDI file
        score = converter.parse(str(midi_path))
        
        # Convert to ABC notation
        abc_str = score.write('abc')
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(abc_str)
        
        return abc_str
    
    except Exception as e:
        print(f"Error converting {midi_path}: {e}")
        return None


def process_directory(input_dir, output_dir, min_length=50, max_length=10000):
    """
    Process all MIDI files in a directory.
    
    Args:
        input_dir: Directory containing MIDI files
        output_dir: Directory to save ABC files
        min_length: Minimum ABC string length (characters)
        max_length: Maximum ABC string length (characters)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all MIDI files
    midi_files = list(input_path.rglob('*.mid')) + list(input_path.rglob('*.midi'))
    
    print(f"Found {len(midi_files)} MIDI files")
    
    successful = 0
    failed = 0
    filtered = 0
    
    for midi_file in tqdm(midi_files, desc="Converting MIDI to ABC"):
        # Create output path maintaining directory structure
        rel_path = midi_file.relative_to(input_path)
        abc_file = output_path / rel_path.with_suffix('.abc')
        
        # Convert
        abc_str = midi_to_abc(midi_file, abc_file)
        
        if abc_str:
            # Filter by length
            if min_length <= len(abc_str) <= max_length:
                successful += 1
            else:
                filtered += 1
                if abc_file.exists():
                    abc_file.unlink()  # Remove filtered file
        else:
            failed += 1
    
    print(f"\nConversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Filtered (length): {filtered}")
    print(f"  Total processed: {len(midi_files)}")
    
    return successful, failed, filtered


def main():
    parser = argparse.ArgumentParser(description='Convert MIDI files to ABC notation')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing MIDI files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save ABC files')
    parser.add_argument('--min_length', type=int, default=50,
                       help='Minimum ABC string length')
    parser.add_argument('--max_length', type=int, default=10000,
                       help='Maximum ABC string length')
    
    args = parser.parse_args()
    
    process_directory(
        args.input_dir,
        args.output_dir,
        args.min_length,
        args.max_length
    )


if __name__ == '__main__':
    main()

