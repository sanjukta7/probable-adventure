import torch
import torch.nn as nn
from mdna.merging import DynamicTokenUnmerge
from mdna.utils import TransformerBlock, SinusoidalPositionalEmbedding

class LatentDecoder(nn.Module):
    def __init__(self, layers, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.pos_emb = SinusoidalPositionalEmbedding(dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(layers)
        ])
        
    def forward(self, x):
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        return x

class LocalDecoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.unmerger = DynamicTokenUnmerge(dim)
        self.transformer = TransformerBlock(dim, heads, mlp_dim, dropout)
        
    def forward(self, x):
        # Reverse of Encoder: Unmerge then Transform
        x = self.unmerger(x)
        x = self.transformer(x)
        return x

class LocalDecoder(nn.Module):
    def __init__(self, layers, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            LocalDecoderBlock(dim, heads, mlp_dim, dropout)
            for _ in range(layers)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

