#!/bin/bash
# Generate all visualizations for the project report

set -e

# Configuration
DATA_DIR="data/tokenized/splits"
VOCAB_PATH="data/tokenized/vocab.json"
TRANSFORMER_DIR="outputs/models/transformer"
RNN_DIR="outputs/models/rnn"
OUTPUT_DIR="outputs/plots"

echo "=== Generating All Visualizations ==="
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# 1. Data Statistics
echo "1. Generating data statistics visualizations..."
python src/evaluation/plot_data_statistics.py \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir $OUTPUT_DIR

echo ""

# 2. Scaling Laws
echo "2. Generating scaling laws plots..."
python src/evaluation/scaling_analysis.py \
    --transformer_dir $TRANSFORMER_DIR \
    --rnn_dir $RNN_DIR \
    --output_dir $OUTPUT_DIR

echo ""

# 3. Training Curves
echo "3. Generating training curves..."
python src/evaluation/plot_training_curves.py \
    --transformer_dir $TRANSFORMER_DIR \
    --rnn_dir $RNN_DIR \
    --output_dir $OUTPUT_DIR

echo ""

# 4. Computational Efficiency
echo "4. Generating computational efficiency plots..."
python src/evaluation/plot_computational_efficiency.py \
    --transformer_dir $TRANSFORMER_DIR \
    --rnn_dir $RNN_DIR \
    --output_dir $OUTPUT_DIR

echo ""
echo "=== All Visualizations Generated! ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh $OUTPUT_DIR/*.png 2>/dev/null || echo "No PNG files found"
ls -lh $OUTPUT_DIR/*.csv 2>/dev/null || echo "No CSV files found"
ls -lh $OUTPUT_DIR/*.json 2>/dev/null || echo "No JSON files found"

