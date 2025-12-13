#!/bin/bash
# Generate final music samples from best model

echo "=== CS-GY 6923: Sample Generation ==="
echo ""

VOCAB_PATH="data/tokenized/vocab.json"
MODEL_PATH="outputs/models/transformer/xl/best_model.pt"

echo "Generating samples from best model (Transformer XL)..."
python src/evaluation/generate_samples.py \
    --model_path $MODEL_PATH \
    --vocab_path $VOCAB_PATH \
    --model_type transformer \
    --output_dir outputs/samples \
    --num_samples 10 \
    --max_new_tokens 500 \
    --temperature 1.0 \
    --top_k 50 \
    --top_p 0.9

echo ""
echo "=== Sample Generation Complete! ==="
echo "Samples saved to: outputs/samples/"

