import torch
import torch.nn as nn
from mdna.merging import DynamicTokenMerge
from mdna.utils import TransformerBlock, SinusoidalPositionalEmbedding

class LocalEncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.transformer = TransformerBlock(dim, heads, mlp_dim, dropout)
        self.merger = DynamicTokenMerge(dim)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.merger(x)
        return x

class LocalEncoder(nn.Module):
    def __init__(self, layers, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            LocalEncoderBlock(dim, heads, mlp_dim, dropout)
            for _ in range(layers)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class LatentEncoder(nn.Module):
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
