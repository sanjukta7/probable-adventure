import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLocalMerge(nn.Module):
    """
    A simple baseline merging strategy that uses average pooling
    to reduce sequence length by half.
    """
    def __init__(self, dim=None):
        super().__init__()
        # dim argument included for API compatibility with more complex mergers
        
    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        Returns: [Batch, Seq_Len/2, Dim]
        """
        B, N, C = x.shape
        if N % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            N += 1
            
        # Reshape to [B, N/2, 2, C] and average over the pair dimension
        x = x.view(B, N // 2, 2, C)
        return x.mean(dim=2)

class SimpleLocalUnmerge(nn.Module):
    """
    Simple upsampling by repeating elements.
    """
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        Returns: [Batch, Seq_Len*2, Dim]
        """
        # [B, N, C] -> [B, N, 2, C] -> [B, N*2, C]
        B, N, C = x.shape
        x = x.unsqueeze(2).expand(-1, -1, 2, -1)
        return x.reshape(B, N * 2, C)

