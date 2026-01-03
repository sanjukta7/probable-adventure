"""
Inference utilities for MergeDNA models.

Provides high-level functions for running predictions with trained models.
"""

import sys
import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.backbone import MergeDNAModel, MergeDNAClassifier
from dataloader import DNATokenizer


def predict_promoter(
    model: MergeDNAClassifier,
    sequence: str,
    tokenizer: DNATokenizer,
    device: torch.device,
    seq_length: int = 256
) -> Dict:
    """
    Predict whether a DNA sequence is a promoter.
    
    Args:
        model: Trained MergeDNAClassifier
        sequence: DNA sequence string
        tokenizer: DNATokenizer instance
        device: Device to run inference on
        seq_length: Expected sequence length
        
    Returns:
        Dictionary with prediction, confidence, and probabilities
    """
    model.eval()
    
    # Tokenize
    tokens = tokenizer.encode(sequence, max_length=seq_length).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(tokens)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
    
    return {
        'sequence_preview': sequence[:50] + '...' if len(sequence) > 50 else sequence,
        'prediction': 'Promoter' if pred == 1 else 'Non-Promoter',
        'predicted_class': pred,
        'confidence': probs[0, pred].item(),
        'probabilities': {
            'non_promoter': probs[0, 0].item(),
            'promoter': probs[0, 1].item()
        }
    }


def batch_predict(
    model: MergeDNAClassifier,
    sequences: List[str],
    tokenizer: DNATokenizer,
    device: torch.device,
    seq_length: int = 256,
    batch_size: int = 32
) -> List[Dict]:
    """
    Batch prediction for multiple sequences.
    
    Args:
        model: Trained MergeDNAClassifier
        sequences: List of DNA sequences
        tokenizer: DNATokenizer instance
        device: Device to run inference on
        seq_length: Expected sequence length
        batch_size: Batch size for inference
        
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    results = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        # Tokenize batch
        tokens = tokenizer.batch_encode(batch_seqs, max_length=seq_length).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(tokens)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
        
        # Collect results
        for j, seq in enumerate(batch_seqs):
            pred = preds[j].item()
            results.append({
                'sequence_preview': seq[:50] + '...' if len(seq) > 50 else seq,
                'prediction': 'Promoter' if pred == 1 else 'Non-Promoter',
                'predicted_class': pred,
                'confidence': probs[j, pred].item(),
                'probabilities': {
                    'non_promoter': probs[j, 0].item(),
                    'promoter': probs[j, 1].item()
                }
            })
    
    return results


def get_embeddings(
    model: Union[MergeDNAModel, MergeDNAClassifier],
    sequences: List[str],
    tokenizer: DNATokenizer,
    device: torch.device,
    seq_length: int = 256,
    pooled: bool = True
) -> torch.Tensor:
    """
    Extract embeddings from sequences.
    
    Args:
        model: MergeDNA model (either autoencoder or classifier)
        sequences: List of DNA sequences
        tokenizer: DNATokenizer instance
        device: Device to run inference on
        seq_length: Expected sequence length
        pooled: If True, return mean-pooled embeddings
        
    Returns:
        Embeddings tensor [batch, dim] if pooled, [batch, seq/4, dim] otherwise
    """
    model.eval()
    
    # Tokenize
    tokens = tokenizer.batch_encode(sequences, max_length=seq_length).to(device)
    
    # Get embeddings
    with torch.no_grad():
        if hasattr(model, 'get_embeddings'):
            embeddings = model.get_embeddings(tokens)
        else:
            embeddings = model.encode(tokens)
        
        if pooled:
            embeddings = embeddings.mean(dim=1)
    
    return embeddings


def load_classifier(
    checkpoint_path: str,
    vocab_size: int = 12,
    dim: int = 64,
    local_layers: int = 2,
    latent_layers: int = 2,
    heads: int = 4,
    mlp_dim: int = 256,
    num_classes: int = 2,
    device: Optional[torch.device] = None
) -> MergeDNAClassifier:
    """
    Load a trained classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        vocab_size: Vocabulary size
        dim: Model dimension
        local_layers: Number of local encoder layers
        latent_layers: Number of latent transformer layers
        heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        num_classes: Number of classes
        device: Device to load model on
        
    Returns:
        Loaded MergeDNAClassifier
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MergeDNAClassifier(
        vocab_size=vocab_size,
        dim=dim,
        local_layers=local_layers,
        latent_layers=latent_layers,
        heads=heads,
        mlp_dim=mlp_dim,
        num_classes=num_classes
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


if __name__ == "__main__":
    # Demo usage
    print("MergeDNA Inference Utilities")
    print("=" * 40)
    
    # Initialize tokenizer
    tokenizer = DNATokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create a dummy model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MergeDNAClassifier(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        num_classes=2
    ).to(device)
    
    # Test prediction
    from data.dataloader import generate_promoter_sequence, generate_non_promoter_sequence
    
    test_seq = generate_promoter_sequence(256)
    result = predict_promoter(model, test_seq, tokenizer, device)
    
    print(f"\nTest prediction:")
    print(f"  Sequence: {result['sequence_preview']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")



