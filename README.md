<h1 align="center">Symbolic Music Scaling Laws</h1>
<p align="center"><b>Do neural scaling laws hold for music? Training Transformers and RNNs of five sizes on symbolic music to find out.</b></p>

<p align="center">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.0+-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="Weights & Biases" src="https://img.shields.io/badge/tracking-wandb-FFBE00?logo=weightsandbiases&logoColor=black">
</p>

Neural scaling laws (loss decreasing as a predictable power law in model
size and data) are well studied for text. This project runs the same
experiment on symbolic music: Transformer and LSTM language models at five
sizes (tiny through XL) are trained on ABC-notation music converted from
the Lakh MIDI Dataset, then their validation loss is fit against parameter
count to see whether music follows the same power-law trend as language.

## Pipeline

```
MIDI (Lakh dataset) → ABC notation (music21) → tokenize → train N model sizes → fit scaling curve
```

1. **Preprocessing** (`src/preprocessing/`): downloads the Lakh MIDI
   Dataset, converts MIDI to ABC notation with `music21`, tokenizes it,
   and splits train/val/test.
2. **Models** (`src/models/`): a from-scratch Transformer and an
   LSTM-based RNN, both implemented directly (not wrapping a pretrained
   model), sharing a common `base.py` interface.
3. **Training** (`src/training/`): trains each architecture at five
   configured sizes (`src/configs/transformer_{tiny,small,medium,large,xl}.yaml`,
   `rnn_configs.yaml`), logging to TensorBoard/W&B.
4. **Evaluation** (`src/evaluation/`): fits a power law to loss vs.
   parameter count, plots training curves and compute efficiency, and
   generates music samples from trained checkpoints.

## Running it

```bash
pip install -r requirements.txt

# 1. Data: download Lakh MIDI, convert to ABC, tokenize, split
python run_preprocessing.py

# 2. Train every configured model size and fit the scaling curve
python run_scaling_study.py

# 3. Plots: training curves, compute efficiency, scaling law fit
python generate_all_visualizations.py

# 4. Sample generation from a trained checkpoint
python generate_final_samples.py
```

Every script above has a `scripts/*.sh` equivalent if you'd rather run
steps individually; see the `--help` on each `src/**/*.py` entry point for
manual, single-model runs. Training every configured size takes several
hours on a single GPU.

## Repository layout

```
src/preprocessing/   MIDI download, MIDI->ABC conversion, tokenization, splitting
src/models/           Transformer and RNN implementations
src/training/         training loops per architecture
src/evaluation/       scaling-law fitting, plots, sample generation
src/configs/           model size configs (tiny -> xl)
scripts/               shell entry points mirroring the run_*.py scripts
```

## Citation

- Lakh MIDI Dataset: Raffel, C. (2016). "Learning-Based Musical Similarity."
- Architecture reference: Karpathy, A. (2023). "nanoGPT."
