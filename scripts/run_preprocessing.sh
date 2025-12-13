#!/bin/bash
# Run complete data preprocessing pipeline

echo "=== CS-GY 6923: Data Preprocessing Pipeline ==="
echo ""

# Step 1: Download Lakh MIDI dataset
echo "Step 1: Downloading Lakh MIDI dataset..."
python src/preprocessing/download_data.py --output_dir data/raw

# Step 2: Convert MIDI to ABC notation
echo ""
echo "Step 2: Converting MIDI files to ABC notation..."
python src/preprocessing/midi_to_abc.py \
    --input_dir data/raw/lmd_full \
    --output_dir data/processed \
    --min_length 50 \
    --max_length 10000

# Step 3: Tokenize ABC files
echo ""
echo "Step 3: Tokenizing ABC files..."
python src/preprocessing/tokenize.py \
    --input_dir data/processed \
    --output_dir data/tokenized \
    --tokenization_type character \
    --vocab_path data/tokenized/vocab.json

# Step 4: Split data into train/val/test
echo ""
echo "Step 4: Splitting data into train/val/test sets..."
python src/preprocessing/split_data.py \
    --input_dir data/tokenized \
    --output_dir data/tokenized/splits \
    --train_ratio 0.98 \
    --val_ratio 0.01 \
    --test_ratio 0.01 \
    --min_tokens 100000000

echo ""
echo "=== Preprocessing Complete! ==="
echo "Data ready for training at: data/tokenized/splits"
echo "Vocabulary saved at: data/tokenized/vocab.json"

