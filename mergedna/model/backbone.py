import torch
import torch.nn as nn
from model.encoder import LocalEncoder, LatentEncoder
from model.decoder import LatentDecoder, LocalDecoder
from mdna.utils import SinusoidalPositionalEmbedding

class MergeDNAModel(nn.Module):
    def __init__(
        self, 
        vocab_size=12, 
        dim=64, 
        local_layers=2, 
        latent_layers=2, 
        heads=4, 
        mlp_dim=256
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = SinusoidalPositionalEmbedding(dim)
        
        # Encoder
        self.local_encoder = LocalEncoder(local_layers, dim, heads, mlp_dim)
        self.latent_encoder = LatentEncoder(latent_layers, dim, heads, mlp_dim)
        
        # Decoder
        self.latent_decoder = LatentDecoder(latent_layers, dim, heads, mlp_dim)
        self.local_decoder = LocalDecoder(local_layers, dim, heads, mlp_dim)
        
        # Head
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        # x: [Batch, Seq_Len] (indices)
        
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

