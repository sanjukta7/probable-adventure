"""
Data utilities for MergeDNA.

Includes:
- DNATokenizer: Tokenize DNA sequences to integer indices
- PromoterDataset: Dataset for promoter prediction task
- ToyDNADataset: Simple dataset for pretraining
"""

import torch
from torch.utils.data import Dataset
import random
from typing import Optional, List, Tuple


class DNATokenizer:
    """
    DNA sequence tokenizer.
    
    Maps nucleotides and special tokens to integer indices.
    Supports IUPAC ambiguity codes.
    """
    
    def __init__(self):
        # Token vocabulary
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<CLS>': 2,
            '<SEP>': 3,
            'A': 4,
            'T': 5,
            'C': 6,
            'G': 7,
            # IUPAC ambiguity codes
            'N': 8,   # Any nucleotide
            'R': 9,   # A or G (purine)
            'Y': 10,  # C or T (pyrimidine)
            'S': 11,  # G or C
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Nucleotides for random sequence generation
        self.nucleotides = ['A', 'T', 'C', 'G']
    
    def encode(self, sequence: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode a DNA sequence to token indices.
        
        Args:
            sequence: DNA sequence string
            max_length: Optional max length (pads/truncates if provided)
            
        Returns:
            Tensor of token indices
        """
        tokens = [self.vocab.get(c.upper(), self.vocab['<UNK>']) for c in sequence]
        
        if max_length:
            if len(tokens) < max_length:
                tokens += [self.vocab['<PAD>']] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode token indices back to DNA sequence.
        
        Args:
            tokens: Tensor of token indices
            
        Returns:
            DNA sequence string
        """
        return ''.join(self.inv_vocab.get(t.item(), '?') for t in tokens)
    
    def batch_encode(self, sequences: List[str], max_length: int) -> torch.Tensor:
        """
        Encode a batch of sequences.
        
        Args:
            sequences: List of DNA sequences
            max_length: Maximum sequence length
            
        Returns:
            Tensor of shape [batch_size, max_length]
        """
        return torch.stack([self.encode(seq, max_length) for seq in sequences])


def generate_random_dna(length: int) -> str:
    """Generate a random DNA sequence."""
    return ''.join(random.choices(['A', 'T', 'C', 'G'], k=length))


def generate_promoter_sequence(length: int = 256) -> str:
    """
    Generate a synthetic promoter-like sequence with characteristic motifs.
    
    Promoter characteristics (simplified):
    - TATA box motif (~25-35 bp upstream of TSS)
    - GC-rich regions (CpG islands)
    - Initiator element (Inr)
    
    Args:
        length: Sequence length
        
    Returns:
        Synthetic promoter sequence
    """
    seq = list(generate_random_dna(length))
    
    # Insert TATA box around position 50-60
    tata_variants = ['TATAAA', 'TATATA', 'TATAAG', 'TATAAT']
    tata_pos = random.randint(45, 55)
    tata_seq = random.choice(tata_variants)
    for i, nucleotide in enumerate(tata_seq):
        if tata_pos + i < length:
            seq[tata_pos + i] = nucleotide
    
    # Insert GC-rich region (CpG island-like) around position 80-120
    gc_start = random.randint(75, 85)
    gc_length = random.randint(30, 40)
    for i in range(gc_length):
        if gc_start + i < length:
            # ~75% GC content
            seq[gc_start + i] = random.choice(['G', 'C', 'G', 'C', 'G', 'C', 'A', 'T'])
    
    # Insert initiator element around position 130-140
    inr_pos = random.randint(125, 135)
    inr_patterns = ['TCAGTT', 'CCAATT', 'TCAATT', 'CCAGTT']
    inr_seq = random.choice(inr_patterns)
    for i, nucleotide in enumerate(inr_seq):
        if inr_pos + i < length:
            seq[inr_pos + i] = nucleotide
    
    return ''.join(seq)


def generate_non_promoter_sequence(length: int = 256) -> str:
    """
    Generate a non-promoter sequence (random, avoiding promoter patterns).
    
    Args:
        length: Sequence length
        
    Returns:
        Non-promoter sequence
    """
    seq = generate_random_dna(length)
    # Remove any accidental TATA boxes
    seq = seq.replace('TATAAA', 'GGCCGG')
    seq = seq.replace('TATATA', 'GGCCGG')
    return seq


class PromoterDataset(Dataset):
    """
    Dataset for promoter prediction task.
    
    Generates synthetic promoter and non-promoter sequences
    for binary classification.
    """
    
    def __init__(
        self, 
        num_samples: int = 1000, 
        seq_length: int = 256, 
        tokenizer: Optional[DNATokenizer] = None
    ):
        """
        Args:
            num_samples: Total number of samples (balanced 50/50)
            seq_length: Length of each sequence
            tokenizer: DNATokenizer instance (creates one if not provided)
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.tokenizer = tokenizer or DNATokenizer()
        
        # Generate balanced dataset
        self.sequences: List[str] = []
        self.labels: List[int] = []
        
        for i in range(num_samples):
            if i < num_samples // 2:
                # Promoter (positive)
                seq = generate_promoter_sequence(seq_length)
                label = 1
            else:
                # Non-promoter (negative)
                seq = generate_non_promoter_sequence(seq_length)
                label = 0
            
            self.sequences.append(seq)
            self.labels.append(label)
        
        # Shuffle
        combined = list(zip(self.sequences, self.labels))
        random.shuffle(combined)
        self.sequences, self.labels = zip(*combined)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(seq, max_length=self.seq_length)
        
        return {
            'input_ids': tokens,
            'label': torch.tensor(label, dtype=torch.long)
        }


class ToyDNADataset(Dataset):
    """
    Simple dataset for pretraining (random sequences).
    
    Used for autoencoder-style pretraining where the model
    learns to reconstruct the input sequence.
    """
    
    def __init__(self, size: int = 1000, seq_len: int = 128):
        """
        Args:
            size: Number of samples
            seq_len: Sequence length
        """
        self.size = size
        self.seq_len = seq_len
        self.vocab = [0, 1, 2, 3]  # A, C, G, T (simplified)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Generate random sequence
        seq = torch.randint(0, 4, (self.seq_len,))
        return seq


