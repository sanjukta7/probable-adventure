from torch import nn
from mdna.merging import LocalTokenMerge

class MergeDNAEncoder(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(...),
            TransformerBlock(...),
            TransformerBlock(...)
        ])
        
        # The merging layer from your folder
        self.merger = LocalTokenMerge(dim=768)

    def forward(self, x):
        # Layer 1: Process raw bases
        x = self.blocks[0](x)
        
        # MERGE STEP: Reduce sequence length (Context Awareness)
        # "Zoom out" from 1000bp -> 500bp
        x = self.merger(x) 
        
        # Layer 2: Process the merged "words"
        x = self.blocks[1](x)
        
        return x