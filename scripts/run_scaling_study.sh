#!/bin/bash
# Run complete scaling study for transformers and RNNs

echo "=== CS-GY 6923: Scaling Study ==="
echo ""

VOCAB_PATH="data/tokenized/vocab.json"
DATA_DIR="data/tokenized/splits"

# Train transformer models
echo "Training Transformer Models..."
echo ""

echo "1. Training Tiny Transformer (~1M params)..."
python src/training/train_transformer.py \
    --config src/configs/transformer_tiny.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/transformer/tiny

echo ""
echo "2. Training Small Transformer (~5M params)..."
python src/training/train_transformer.py \
    --config src/configs/transformer_small.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/transformer/small

echo ""
echo "3. Training Medium Transformer (~20M params)..."
python src/training/train_transformer.py \
    --config src/configs/transformer_medium.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/transformer/medium

echo ""
echo "4. Training Large Transformer (~50M params)..."
python src/training/train_transformer.py \
    --config src/configs/transformer_large.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/transformer/large

echo ""
echo "5. Training XL Transformer (~100M+ params)..."
python src/training/train_transformer.py \
    --config src/configs/transformer_xl.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/transformer/xl

# Train RNN models
echo ""
echo "Training RNN Models..."
echo ""

echo "1. Training Tiny RNN..."
python src/training/train_rnn.py \
    --config src/configs/rnn_configs.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/rnn \
    --model_size tiny

echo ""
echo "2. Training Small RNN..."
python src/training/train_rnn.py \
    --config src/configs/rnn_configs.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/rnn \
    --model_size small

echo ""
echo "3. Training Medium RNN..."
python src/training/train_rnn.py \
    --config src/configs/rnn_configs.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/rnn \
    --model_size medium

echo ""
echo "4. Training Large RNN..."
python src/training/train_rnn.py \
    --config src/configs/rnn_configs.yaml \
    --data_dir $DATA_DIR \
    --vocab_path $VOCAB_PATH \
    --output_dir outputs/models/rnn \
    --model_size large

# Analyze scaling laws
echo ""
echo "Analyzing Scaling Laws..."
python src/evaluation/scaling_analysis.py \
    --transformer_dir outputs/models/transformer \
    --rnn_dir outputs/models/rnn \
    --output_dir outputs/plots

echo ""
echo "=== Scaling Study Complete! ==="
echo "Results saved to: outputs/plots/"

