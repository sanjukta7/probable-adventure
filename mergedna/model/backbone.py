"""
MergeDNA Model Backbones.

This module contains the core model architectures:
- MergeDNAModel: Autoencoder for pretraining
- MergeDNAClassifier: Classifier for downstream tasks
"""

import torch
import torch.nn as nn
from model.encoder import LocalEncoder, LatentEncoder
from model.decoder import LatentDecoder, LocalDecoder
from mdna.utils import SinusoidalPositionalEmbedding


class MergeDNAModel(nn.Module):
    """
    MergeDNA Autoencoder Model.
    
    Uses dynamic token merging in the encoder to compress sequences,
    then reconstructs them in the decoder. Useful for pretraining.
    """
    
    def __init__(
        self, 
        vocab_size: int = 12, 
        dim: int = 64, 
        local_layers: int = 2, 
        latent_layers: int = 2, 
        heads: int = 4, 
        mlp_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            dim: Model dimension
            local_layers: Number of local encoder/decoder layers (each halves/doubles seq length)
            latent_layers: Number of latent transformer layers
            heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.local_layers = local_layers
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = SinusoidalPositionalEmbedding(dim)
        
        # Encoder
        self.local_encoder = LocalEncoder(local_layers, dim, heads, mlp_dim, dropout)
        self.latent_encoder = LatentEncoder(latent_layers, dim, heads, mlp_dim, dropout)
        
        # Decoder
        self.latent_decoder = LatentDecoder(latent_layers, dim, heads, mlp_dim, dropout)
        self.local_decoder = LocalDecoder(local_layers, dim, heads, mlp_dim, dropout)
        
        # Head
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for autoencoder.
        
        Args:
            x: Input token indices [Batch, Seq_Len]
            
        Returns:
            Logits [Batch, Seq_Len, Vocab_Size]
        """
        # Embed
        x_emb = self.embedding(x)
        x_emb = self.pos_emb(x_emb)
        
        # Encode
        # Local Encoder reduces sequence length by 2^local_layers
        x_local = self.local_encoder(x_emb)
        x_latent = self.latent_encoder(x_local)
        
        # Decode
        x_dec_latent = self.latent_decoder(x_latent)
        # Local Decoder increases sequence length by 2^local_layers
        x_out = self.local_decoder(x_dec_latent)
        
        # Prediction
        logits = self.head(x_out)
        
        return logits
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input token indices [Batch, Seq_Len]
            
        Returns:
            Latent embeddings [Batch, Seq_Len / 2^local_layers, Dim]
        """
        x_emb = self.embedding(x)
        x_emb = self.pos_emb(x_emb)
        x_local = self.local_encoder(x_emb)
        x_latent = self.latent_encoder(x_local)
        return x_latent


class MergeDNAClassifier(nn.Module):
    """
    MergeDNA Classifier for sequence classification tasks.
    
    Uses the encoder from MergeDNA to extract features, then applies
    global pooling and a classification head.
    """
    
    def __init__(
        self,
        vocab_size: int = 12,
        dim: int = 64,
        local_layers: int = 2,
        latent_layers: int = 2,
        heads: int = 4,
        mlp_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        pooling: str = 'mean'
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            dim: Model dimension
            local_layers: Number of local encoder layers
            latent_layers: Number of latent transformer layers
            heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Pooling strategy ('mean', 'max', or 'cls')
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.local_layers = local_layers
        self.num_classes = num_classes
        self.pooling = pooling
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = SinusoidalPositionalEmbedding(dim)
        
        # Encoder (from MergeDNA)
        self.local_encoder = LocalEncoder(local_layers, dim, heads, mlp_dim, dropout)
        self.latent_encoder = LatentEncoder(latent_layers, dim, heads, mlp_dim, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: Input token indices [Batch, Seq_Len]
            
        Returns:
            Class logits [Batch, Num_Classes]
        """
        # Get embeddings
        embeddings = self.get_embeddings(x)
        
        # Global pooling
        if self.pooling == 'mean':
            x_pooled = embeddings.mean(dim=1)
        elif self.pooling == 'max':
            x_pooled = embeddings.max(dim=1).values
        else:  # cls token (first position)
            x_pooled = embeddings[:, 0]
        
        # Classification
        logits = self.classifier(x_pooled)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent embeddings for input sequences.
        
        Useful for visualization or downstream tasks.
        
        Args:
            x: Input token indices [Batch, Seq_Len]
            
        Returns:
            Latent embeddings [Batch, Seq_Len / 2^local_layers, Dim]
        """
        x_emb = self.embedding(x)
        x_emb = self.pos_emb(x_emb)
        x_local = self.local_encoder(x_emb)
        x_latent = self.latent_encoder(x_local)
        return x_latent
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model: MergeDNAModel, 
        num_classes: int = 2,
        freeze_encoder: bool = False
    ) -> 'MergeDNAClassifier':
        """
        Create a classifier from a pretrained MergeDNA model.
        
        Args:
            pretrained_model: Pretrained MergeDNAModel
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze encoder weights
            
        Returns:
            MergeDNAClassifier with pretrained encoder weights
        """
        # Create classifier with same architecture
        classifier = cls(
            vocab_size=pretrained_model.vocab_size,
            dim=pretrained_model.dim,
            local_layers=pretrained_model.local_layers,
            num_classes=num_classes
        )
        
        # Copy encoder weights
        classifier.embedding.load_state_dict(pretrained_model.embedding.state_dict())
        classifier.pos_emb.load_state_dict(pretrained_model.pos_emb.state_dict())
        classifier.local_encoder.load_state_dict(pretrained_model.local_encoder.state_dict())
        classifier.latent_encoder.load_state_dict(pretrained_model.latent_encoder.state_dict())
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in classifier.embedding.parameters():
                param.requires_grad = False
            for param in classifier.pos_emb.parameters():
                param.requires_grad = False
            for param in classifier.local_encoder.parameters():
                param.requires_grad = False
            for param in classifier.latent_encoder.parameters():
                param.requires_grad = False
        
        return classifier

