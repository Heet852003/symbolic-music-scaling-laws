"""
RNN/LSTM language model for music generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import count_parameters


class RNNLM(nn.Module):
    """
    RNN-based language model for music generation.
    
    Architecture:
    - Token embeddings
    - LSTM layers
    - Output projection to vocabulary
    """
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512,
                 n_layers=2, dropout=0.1, tie_weights=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Optionally tie weights
        if tie_weights:
            self.head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, idx, targets=None, hidden=None):
        """
        Forward pass.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target token indices [batch_size, seq_len] (optional)
            hidden: Hidden state tuple (h, c) (optional)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar (if targets provided)
            hidden: Updated hidden state
        """
        batch_size, seq_len = idx.size()
        
        # Token embeddings
        x = self.token_embedding(idx)
        
        # LSTM forward
        if hidden is None:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=idx.device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=idx.device)
            hidden = (h0, c0)
        
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        
        # Output projection
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss, hidden
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate new tokens.
        
        Args:
            idx: Starting sequence [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            Generated sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        hidden = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _, hidden = self(idx, hidden=hidden)
                
                # Get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


def create_rnn_model(vocab_size, config):
    """
    Create an RNN model from configuration.
    
    Args:
        vocab_size: Vocabulary size
        config: Dictionary with model configuration
    
    Returns:
        RNNLM model
    """
    model = RNNLM(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        n_layers=config.get('n_layers', 2),
        dropout=config.get('dropout', 0.1),
        tie_weights=config.get('tie_weights', False)
    )
    
    return model

