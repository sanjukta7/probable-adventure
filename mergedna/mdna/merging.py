import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicTokenMerge(nn.Module):
    """
    Implements the dynamic token merging with local-window constraints.
    It reduces the sequence length by half (2->1 merge) by computing
    attention-like weights between adjacent tokens.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # A small scorer to decide importance/weight of each token in the pair
        self.scorer = nn.Linear(dim, 1)
        
    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        """
        B, N, C = x.shape
        if N % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            N += 1
            
        # [B, N/2, 2, C]
        pairs = x.view(B, N // 2, 2, C)
        
        # Compute scores: [B, N/2, 2, 1]
        scores = self.scorer(pairs)
        
        # Softmax over the pair dimension to get merge weights
        weights = F.softmax(scores, dim=2) # [B, N/2, 2, 1]
        
        # Weighted sum: [B, N/2, C]
        # x_merged = w1*x1 + w2*x2
        x_merged = (pairs * weights).sum(dim=2)
        
        return x_merged

class DynamicTokenUnmerge(nn.Module):
    """
    Symmetric operation to DynamicTokenMerge.
    Expands 1 token back into 2 tokens using a learned up-projection.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Projects 1 token to 2 tokens (concatenated features)
        self.expand = nn.Linear(dim, dim * 2)
        
    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        Returns: [Batch, Seq_Len*2, Dim]
        """
        B, N, C = x.shape
        
        # [B, N, 2*C]
        expanded = self.expand(x)
        
        # [B, N, 2, C]
        expanded = expanded.view(B, N, 2, C)
        
        # [B, N*2, C]
        output = expanded.view(B, N * 2, C)
        
        return output
