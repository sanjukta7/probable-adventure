import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.backbone import MergeDNAModel

# Toy Dataset
class ToyDNADataset(Dataset):
    def __init__(self, size=1000, seq_len=128):
        self.size = size
        self.seq_len = seq_len
        self.vocab = [0, 1, 2, 3] # A, C, G, T

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random sequence
        seq = torch.randint(0, 4, (self.seq_len,))
        return seq

def train():
    # Hyperparams
    VOCAB_SIZE = 4
    DIM = 64
    SEQ_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 2
    LR = 1e-3
    
    # Checkpoints
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Data
    dataset = ToyDNADataset(size=1000, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = MergeDNAModel(
        vocab_size=VOCAB_SIZE, 
        dim=DIM,
        local_layers=2,  # Reduces 128 -> 64 -> 32
        latent_layers=2,
        heads=4,
        mlp_dim=128
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    print("Starting training...")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            # batch: [B, Seq_Len]
            optimizer.zero_grad()
            
            # Forward (Autoencoder task: Predict input)
            logits = model(batch)
            
            # Reshape for loss
            # logits: [B, Seq_Len, Vocab]
            # target: [B, Seq_Len]
            loss = criterion(logits.view(-1, VOCAB_SIZE), batch.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
                
        print(f"Epoch {epoch} Average Loss: {total_loss / len(loader):.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'mergedna_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(loader),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train()

