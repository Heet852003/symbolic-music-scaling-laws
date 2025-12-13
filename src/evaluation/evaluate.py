"""
Evaluate model on test set.
"""

import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transformer import create_transformer_model
from src.models.rnn import create_rnn_model
from src.training.utils import get_dataloader, estimate_loss


def evaluate_model(model_path, model_type, vocab_path, data_dir, device='cuda'):
    """Evaluate a model on test set."""
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model
    if model_type == 'transformer':
        model = create_transformer_model(vocab_size, config.get('model', {}))
    elif model_type == 'rnn':
        model = create_rnn_model(vocab_size, config.get('model', {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_loader, test_size = get_dataloader(
        Path(data_dir) / 'test',
        block_size=config.get('training', {}).get('block_size', 1024),
        batch_size=config.get('training', {}).get('batch_size', 64),
        shuffle=False
    )
    
    # Evaluate
    test_loss = estimate_loss(model, test_loader, device, eval_iters=1000)
    perplexity = torch.exp(torch.tensor(test_loss))
    
    return {
        'test_loss': test_loss,
        'perplexity': perplexity.item(),
        'test_size': test_size
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['transformer', 'rnn'],
                       help='Type of model')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary JSON file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    results = evaluate_model(
        args.model_path, args.model_type, args.vocab_path,
        args.data_dir, device
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Test Loss: {results['test_loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Test Size: {results['test_size']:,} samples")


if __name__ == '__main__':
    main()

