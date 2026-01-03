

def train_mergedna(seq_dictionary, save_path="mergedna_ckpt.pt", epochs=5, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Data
    dataset = DNADataset(seq_dictionary)
    # Collate function to pad sequences if they differ in length
    def collate_fn(batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 2. Model Init
    DIM = 64 # Small for demo
    local_enc = MockLocalEncoder(DIM)
    local_dec = MockLocalDecoder(DIM)
    
    model = MergeDNAModel(local_enc, local_dec, dim=DIM, latent_enc_depth=2, latent_dec_depth=1)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device) # [B, N] (Indices)
            
            optimizer.zero_grad()
            
            # Forward pass (computes all losses internally)
            loss, logs = model.forward_train(batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f} | "
                      f"MTR: {logs['loss_mtr']:.3f} Latent: {logs['loss_latent']:.3f} AMTM: {logs['loss_amtm']:.3f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch} Completed. Avg Loss: {avg_loss:.4f} ===")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Checkpoint saved to {save_path}")