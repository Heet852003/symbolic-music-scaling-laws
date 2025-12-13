"""
Tokenize ABC notation files for language modeling.
Supports character-level and note-level tokenization.
"""

import os
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import re
from collections import Counter


class ABCTokenizer:
    """Tokenizer for ABC notation."""
    
    def __init__(self, tokenization_type='character'):
        """
        Initialize tokenizer.
        
        Args:
            tokenization_type: 'character' or 'note'
        """
        self.tokenization_type = tokenization_type
        self.vocab = {}
        self.vocab_size = 0
        
    def build_vocab(self, text_files):
        """Build vocabulary from text files."""
        print("Building vocabulary...")
        
        all_tokens = []
        
        for text_file in tqdm(text_files, desc="Reading files"):
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
            if self.tokenization_type == 'character':
                tokens = list(text)
            elif self.tokenization_type == 'note':
                tokens = self._extract_notes(text)
            else:
                raise ValueError(f"Unknown tokenization type: {self.tokenization_type}")
            
            all_tokens.extend(tokens)
        
        # Build vocabulary
        token_counts = Counter(all_tokens)
        
        # Special tokens
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
        # Add tokens sorted by frequency
        for token, count in token_counts.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        
        # Create reverse mapping
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common tokens: {token_counts.most_common(10)}")
        
        return self.vocab
    
    def _extract_notes(self, abc_text):
        """Extract note-level tokens from ABC text."""
        # Simple note extraction: find note patterns like A, B, C, etc.
        # This is a simplified version - can be enhanced
        notes = re.findall(r'[A-Ga-g][#b]?[0-9]*', abc_text)
        return notes
    
    def encode(self, text):
        """Encode text to token IDs."""
        if self.tokenization_type == 'character':
            tokens = list(text)
        else:
            tokens = self._extract_notes(text)
        
        token_ids = []
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab['<UNK>']))
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in token_ids]
        return ''.join(tokens)
    
    def save(self, path):
        """Save tokenizer to file."""
        with open(path, 'w') as f:
            json.dump({
                'tokenization_type': self.tokenization_type,
                'vocab': self.vocab,
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load(self, path):
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.tokenization_type = data['tokenization_type']
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}


def tokenize_files(input_dir, output_dir, tokenizer_type='character', vocab_path=None):
    """
    Tokenize all ABC files in a directory.
    
    Args:
        input_dir: Directory containing ABC files
        output_dir: Directory to save tokenized files
        tokenizer_type: Type of tokenization ('character' or 'note')
        vocab_path: Path to save/load vocabulary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all ABC files
    abc_files = list(input_path.rglob('*.abc'))
    print(f"Found {len(abc_files)} ABC files")
    
    # Initialize tokenizer
    tokenizer = ABCTokenizer(tokenizer_type)
    
    # Build vocabulary
    if vocab_path and Path(vocab_path).exists():
        print(f"Loading vocabulary from {vocab_path}")
        tokenizer.load(vocab_path)
    else:
        tokenizer.build_vocab(abc_files)
        if vocab_path:
            tokenizer.save(vocab_path)
            print(f"Saved vocabulary to {vocab_path}")
    
    # Tokenize files
    total_tokens = 0
    
    for abc_file in tqdm(abc_files, desc="Tokenizing files"):
        with open(abc_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        token_ids = tokenizer.encode(text)
        total_tokens += len(token_ids)
        
        # Save tokenized file
        rel_path = abc_file.relative_to(input_path)
        token_file = output_path / rel_path.with_suffix('.tokens')
        token_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(token_file, 'w') as f:
            f.write(' '.join(map(str, token_ids)))
    
    print(f"\nTokenization complete:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Average tokens per file: {total_tokens / len(abc_files):.1f}")
    
    return tokenizer, total_tokens


def main():
    parser = argparse.ArgumentParser(description='Tokenize ABC notation files')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing ABC files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save tokenized files')
    parser.add_argument('--tokenization_type', type=str, default='character',
                       choices=['character', 'note'],
                       help='Type of tokenization')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to save/load vocabulary')
    
    args = parser.parse_args()
    
    tokenize_files(
        args.input_dir,
        args.output_dir,
        args.tokenization_type,
        args.vocab_path
    )


if __name__ == '__main__':
    main()

