"""
Download Lakh MIDI Dataset
Downloads and extracts the Lakh MIDI dataset for music language modeling.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse


def download_file(url, output_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description='Download Lakh MIDI Dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Directory to save downloaded files')
    parser.add_argument('--download_url', type=str,
                       default='https://colinraffel.com/projects/lmd/lmd_full.tar.gz',
                       help='URL to download dataset')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    archive_name = Path(args.download_url).name
    archive_path = output_dir / archive_name
    
    if archive_path.exists():
        print(f"Archive already exists: {archive_path}")
    else:
        print(f"Downloading from {args.download_url}...")
        download_file(args.download_url, archive_path)
    
    # Extract if not already extracted
    extracted_dir = output_dir / 'lmd_full'
    if not extracted_dir.exists():
        extract_archive(archive_path, output_dir)
    else:
        print(f"Dataset already extracted to {extracted_dir}")
    
    # Count MIDI files
    midi_files = list(extracted_dir.rglob('*.mid')) + list(extracted_dir.rglob('*.midi'))
    print(f"\nFound {len(midi_files)} MIDI files")
    print(f"Dataset ready at: {extracted_dir}")


if __name__ == '__main__':
    main()

