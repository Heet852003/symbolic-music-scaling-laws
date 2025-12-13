"""
Generate music samples from trained models.
"""

import torch
import argparse
import json
from pathlib import Path
from music21 import converter, stream
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transformer import create_transformer_model
from src.models.rnn import create_rnn_model


def load_model(model_path, model_type='transformer', vocab_path=None, device='cuda'):
    """Load a trained model."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    
    # Create model
    if model_type == 'transformer':
        config = checkpoint.get('config', {}).get('model', {})
        model = create_transformer_model(vocab_size, config)
    elif model_type == 'rnn':
        config = checkpoint.get('config', {}).get('model', {})
        model = create_rnn_model(vocab_size, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, vocab_data


def generate_sample(model, vocab_data, model_type='transformer', 
                   prompt=None, max_new_tokens=500, temperature=1.0,
                   top_k=50, top_p=0.9, device='cuda'):
    """
    Generate a music sample.
    
    Args:
        model: Trained model
        vocab_data: Vocabulary data
        model_type: 'transformer' or 'rnn'
        prompt: Starting sequence (optional)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to use
    
    Returns:
        Generated token sequence
    """
    # Create tokenizer
    vocab = vocab_data['vocab']
    idx_to_token = {idx: token for token, idx in vocab.items()}
    
    # Create prompt
    if prompt is None:
        # Start with BOS token
        start_token = vocab.get('<BOS>', 0)
        idx = torch.tensor([[start_token]], dtype=torch.long, device=device)
    else:
        # Encode prompt
        idx = torch.tensor([[vocab.get(c, vocab.get('<UNK>', 1)) for c in prompt]],
                          dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(
        idx, max_new_tokens, temperature, top_k, top_p
    )
    
    # Decode
    tokens = generated[0].cpu().tolist()
    text = ''.join([idx_to_token.get(t, '<UNK>') for t in tokens])
    
    return text, tokens


def tokens_to_abc(tokens, vocab_data):
    """Convert token IDs back to ABC notation."""
    idx_to_token = {idx: token for token, idx in vocab_data['vocab'].items()}
    text = ''.join([idx_to_token.get(t, '') for t in tokens])
    return text


def evaluate_sample(abc_text):
    """
    Evaluate a generated sample.
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'length': len(abc_text),
        'valid_abc': False,
        'convertible_to_midi': False
    }
    
    # Check if valid ABC
    try:
        score = converter.parse(abc_text, format='abc')
        metrics['valid_abc'] = True
        
        # Check if convertible to MIDI
        try:
            midi_data = score.write('midi')
            metrics['convertible_to_midi'] = True
        except:
            pass
    except:
        pass
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Generate music samples')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary JSON file')
    parser.add_argument('--model_type', type=str, default='transformer',
                       choices=['transformer', 'rnn'],
                       help='Type of model')
    parser.add_argument('--output_dir', type=str, default='outputs/samples',
                       help='Directory to save samples')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, vocab_data = load_model(
        args.model_path, args.model_type, args.vocab_path, device
    )
    print("Model loaded!")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    all_metrics = []
    
    for i in range(args.num_samples):
        print(f"\nGenerating sample {i+1}/{args.num_samples}...")
        
        # Generate
        abc_text, tokens = generate_sample(
            model, vocab_data, args.model_type,
            prompt=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        
        # Evaluate
        metrics = evaluate_sample(abc_text)
        metrics['sample_id'] = i
        all_metrics.append(metrics)
        
        # Save sample
        sample_file = output_dir / f'sample_{i+1}.abc'
        with open(sample_file, 'w') as f:
            f.write(abc_text)
        
        # Try to convert to MIDI
        try:
            score = converter.parse(abc_text, format='abc')
            midi_file = output_dir / f'sample_{i+1}.mid'
            score.write('midi', midi_file)
            print(f"  Saved: {sample_file} and {midi_file}")
        except Exception as e:
            print(f"  Saved: {sample_file} (MIDI conversion failed: {e})")
    
    # Save evaluation summary
    summary = {
        'total_samples': args.num_samples,
        'valid_abc': sum(m['valid_abc'] for m in all_metrics),
        'convertible_to_midi': sum(m['convertible_to_midi'] for m in all_metrics),
        'metrics': all_metrics
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Valid ABC: {summary['valid_abc']}/{args.num_samples}")
    print(f"Convertible to MIDI: {summary['convertible_to_midi']}/{args.num_samples}")
    print(f"Samples saved to: {output_dir}")


if __name__ == '__main__':
    main()

