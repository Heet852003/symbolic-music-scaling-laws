# CS-GY 6923: Scaling Laws for Symbolic Music Language Models

This project explores scaling laws for transformer and RNN-based language models trained on symbolic music data (ABC notation).

## Project Structure

```
cs_gy_6923_project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Raw MIDI files
│   ├── processed/        # Converted ABC files
│   └── tokenized/        # Tokenized data
├── src/
│   ├── preprocessing/
│   │   ├── download_data.py
│   │   ├── midi_to_abc.py
│   │   ├── tokenize.py
│   │   └── split_data.py
│   ├── models/
│   │   ├── transformer.py
│   │   ├── rnn.py
│   │   └── base.py
│   ├── training/
│   │   ├── train_transformer.py
│   │   ├── train_rnn.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── generate_samples.py
│   │   └── scaling_analysis.py
│   └── configs/
│       ├── transformer_tiny.yaml
│       ├── transformer_small.yaml
│       ├── transformer_medium.yaml
│       ├── transformer_large.yaml
│       ├── transformer_xl.yaml
│       └── rnn_configs.yaml
├── scripts/
│   ├── run_preprocessing.sh
│   ├── run_scaling_study.sh
│   └── generate_final_samples.sh
├── outputs/
│   ├── models/           # Trained model checkpoints
│   ├── samples/          # Generated music samples
│   └── plots/            # Scaling plots and training curves
└── report/
    └── report.md         # Project report (convert to PDF)

```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Preprocess Data

**Option A: Using the Python script (Recommended - Cross-platform):**
```bash
python run_preprocessing.py
```

**Option B: Using bash script (Linux/Mac/Git Bash):**
```bash
bash scripts/run_preprocessing.sh
```

**Option C: Manual steps:**
```bash
# Download Lakh MIDI dataset
python src/preprocessing/download_data.py

# Convert MIDI to ABC notation
python src/preprocessing/midi_to_abc.py --input_dir data/raw --output_dir data/processed

# Tokenize and split data
python src/preprocessing/tokenize.py --input_dir data/processed --output_dir data/tokenized
python src/preprocessing/split_data.py --input_dir data/tokenized
```

### 3. Run Scaling Studies

**Option A: Using the Python script (Recommended - Cross-platform):**
```bash
python run_scaling_study.py
```

**Option B: Using bash script (Linux/Mac/Git Bash):**
```bash
bash scripts/run_scaling_study.sh
```

**Option C: Manual training:**
```bash
# Train transformer models
python src/training/train_transformer.py --config src/configs/transformer_tiny.yaml
python src/training/train_transformer.py --config src/configs/transformer_small.yaml
python src/training/train_transformer.py --config src/configs/transformer_medium.yaml
python src/training/train_transformer.py --config src/configs/transformer_large.yaml
python src/training/train_transformer.py --config src/configs/transformer_xl.yaml

# Train RNN models
python src/training/train_rnn.py --config src/configs/rnn_configs.yaml
```

**Note:** Training all models will take many hours. You can also train individual models.

### 4. Generate Visualizations and Analyze Scaling Laws

**Option A: Generate all visualizations (Recommended - Cross-platform):**
```bash
python generate_all_visualizations.py
```

**Option B: Using bash script (Linux/Mac/Git Bash):**
```bash
bash scripts/generate_all_visualizations.sh
```

**Option C: Individual scaling analysis:**
```bash
python src/evaluation/scaling_analysis.py --transformer_dir outputs/models/transformer --rnn_dir outputs/models/rnn
```

### 5. Generate Samples

**Option A: Using the Python script (Recommended - Cross-platform):**
```bash
python generate_final_samples.py
```

**Option B: Using bash script (Linux/Mac/Git Bash):**
```bash
bash scripts/generate_final_samples.sh
```

**Option C: Manual generation:**
```bash
python src/evaluation/generate_samples.py --model_path outputs/models/transformer/xl/best_model.pt --vocab_path data/tokenized/vocab.json --num_samples 10
```

## Key Features

- **Data Pipeline**: Automated MIDI to ABC conversion with music21
- **Tokenization**: Flexible tokenization schemes (character-level, note-level)
- **Model Architectures**: Transformer and LSTM implementations
- **Scaling Analysis**: Power law fitting and comparative analysis
- **Sample Generation**: Conditional and unconditional music generation

## Windows Users

**Windows-compatible scripts are available!** All bash scripts have Python equivalents that work on Windows:

- `generate_all_visualizations.py` - Generate all plots
- `run_preprocessing.py` - Run data preprocessing
- `run_scaling_study.py` - Train all models
- `generate_final_samples.py` - Generate music samples

See `WINDOWS_SETUP.md` for detailed Windows setup instructions.

## Results

See `report/report.md` for detailed experimental results, scaling plots, and analysis.

## Citation

If you use this code, please cite:
- Lakh MIDI Dataset: Raffel, C. (2016). "Learning-Based Musical Similarity"
- nanoGPT: Karpathy, A. (2023). "nanoGPT"

