import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Attention Mechanism ---

class FlashAttention(nn.Module):
    """
    Standard Self-Attention (Flash Attention compatible).
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- 2. Global Merging for Latent Selection ---

class GlobalTokenSelector(nn.Module):
    """
    Selects K salient tokens for the Latent MTR task using bipartite matching.
    Unlike Local ToMe, this operates on the whole sequence to find redundancy globally.
    """
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, dim // 4)

    def forward(self, x, k_target):
        """
        x: [B, L, D]
        k_target: int, number of tokens to keep
        Returns:
            x_selected: [B, K, D]
            source_map: [B, L] -> maps each original token to a group ID
            group_sizes: [B, K] -> how many tokens were merged into each selected token
        """
        B, L, D = x.shape
        r = L - k_target # Number of tokens to remove
        
        if r <= 0: return x, torch.arange(L).expand(B, L), torch.ones(B, L)

        # Compute similarity metric
        metric = self.scorer(x) # [B, L, D']
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # Bipartite matching (simplified for one-shot reduction)
        # Partition into A (keepers) and B (candidates for merging)
        # Note: A robust implementation would loop. Here we assume we can remove 'r' in one pass
        # or we just select top-k salient tokens based on magnitude/norm for simplicity 
        # given the complexity of global bipartite matching in one block.
        
        # Paper approach: "ToMe-style Attention... merges tokens".
        # We will use a magnitude-based selection for stability in this example,
        # or we can assume strictly bipartite. Let's do a strict Top-K selection 
        # based on "saliency" (norm of metric projection) as a proxy for 'unmergeable'.
        
        saliency = torch.norm(metric, dim=-1) # [B, L]
        
        # Keep top K tokens
        _, indices = torch.topk(saliency, k_target, dim=1) # [B, K]
        indices = indices.sort(dim=1)[0] # Keep order
        
        # Gather selected tokens
        x_selected = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        # For the "source matrix", we need to assign the dropped tokens to the nearest kept token.
        # This is essentially K-Means with fixed centroids (the kept tokens).
        # We compute distance between all L tokens and the K kept tokens.
        
        with torch.no_grad():
            # [B, L, D] vs [B, K, D] -> [B, L, K] distance
            dists = torch.cdist(x, x_selected) 
            # assign each of L tokens to nearest K token
            group_ids = torch.argmin(dists, dim=2) # [B, L]
            
            # Calculate group sizes for AMTM masking probability
            group_sizes = torch.zeros(B, k_target, device=x.device)
            group_ids_flat = group_ids.view(-1)
            # Standard bincount per batch element is tricky; loop for clarity or use scatter
            for b in range(B):
                group_sizes[b] = torch.bincount(group_ids[b], minlength=k_target).float()
                
        return x_selected, group_ids, group_sizes

# --- 3. Latent Modules ---

class LatentEncoder(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.selector = GlobalTokenSelector(dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_selection(self, x, k_target):
        # Run encoder first to get context
        x_ctx = self.forward(x)
        # Select K salient tokens
        x_k, group_ids, group_sizes = self.selector(x_ctx, k_target)
        return x_k, group_ids, group_sizes

class LatentDecoder(nn.Module):
    def __init__(self, dim, depth=4, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.proj_out = nn.Linear(dim, dim) 

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(x)

# --- 4. Main Backbone with Training Logic ---

class MergeDNAModel(nn.Module):
    def __init__(self, local_encoder, local_decoder, dim=512, 
                 latent_enc_depth=6, latent_dec_depth=2, vocab_size=4):
        super().__init__()
        self.dim = dim
        self.local_encoder = local_encoder
        self.local_decoder = local_decoder
        
        self.latent_encoder = LatentEncoder(dim, depth=latent_enc_depth)
        self.latent_decoder = LatentDecoder(dim, depth=latent_dec_depth)
        
        # Prediction head for final reconstruction (logits over A, C, G, T)
        self.head = nn.Linear(dim, vocab_size)

    def forward_train(self, x_input, lambda_latent=0.25):
        """
        Computes the composite loss: L_MTR + lambda*L_Latent_MTR + L_AMTM
        x_input: [B, N, 4] (One-hot) or [B, N] (indices)
        """
        # Convert indices to one-hot if necessary for inputs
        if x_input.dim() == 2:
            x_onehot = F.one_hot(x_input, num_classes=4).float()
            x_indices = x_input
        else:
            x_onehot = x_input.float()
            x_indices = x_input.argmax(dim=-1)

        B, N, _ = x_onehot.shape
        
        # ==========================================
        # 1. Main Path (MTR) - Merged Token Reconstruction
        # ==========================================
        # Local Encode (get tokens Z_L and source map S)
        z_l, s_local = self.local_encoder(x_onehot) # [B, L, D]
        
        # Latent Encode
        z_prime_l = self.latent_encoder(z_l) # [B, L, D]
        
        # Latent Decode
        z_hat_l = self.latent_decoder(z_prime_l) # [B, L, D]
        
        # Local Decode (Unmerge + Refine)
        x_hat_l = self.local_decoder(z_hat_l, s_local) # [B, N, D]
        logits_mtr = self.head(x_hat_l) # [B, N, 4]
        
        loss_mtr = F.cross_entropy(logits_mtr.view(-1, 4), x_indices.view(-1))

        # ==========================================
        # 2. Latent MTR Path (Adaptive Selection)
        # ==========================================
        # Select K salient tokens (e.g., K = L/2)
        k_target = z_l.shape[1] // 2
        
        # Detach local encoder for this path (phi not updated)
        z_l_detached = z_l.detach()
        
        # Get selected tokens and group info
        z_k, group_ids, group_sizes = self.latent_encoder.forward_selection(z_l_detached, k_target)
        
        # Unmerge back to L for decoding (Latent Unmerge Step)
        # We broadcast the K tokens back to L based on group_ids
        # z_k: [B, K, D], group_ids: [B, L]
        group_ids_expanded = group_ids.unsqueeze(-1).expand(-1, -1, self.dim) # [B, L, D]
        z_prime_l_restored = torch.gather(z_k, 1, group_ids_expanded) # [B, L, D]
        
        # Decode
        z_hat_l_latent = self.latent_decoder(z_prime_l_restored)
        x_hat_latent = self.local_decoder(z_hat_l_latent, s_local)
        logits_latent = self.head(x_hat_latent)
        
        loss_latent_mtr = F.cross_entropy(logits_latent.view(-1, 4), x_indices.view(-1))

        # ==========================================
        # 3. Adaptive Masked Token Modeling (AMTM)
        # ==========================================
        # Calculate masking probabilities: P(j) propto 1 / group_size
        # Map group sizes back to tokens
        token_weights = 1.0 / torch.gather(group_sizes, 1, group_ids) # [B, L]
        token_probs = token_weights / token_weights.sum(dim=1, keepdim=True)
        
        # Create Mask M_L based on probs (Mask K tokens)
        # We sample indices to mask
        mask_indices = torch.multinomial(token_probs, k_target, replacement=False)
        mask_l = torch.zeros(B, z_l.shape[1], device=x_input.device)
        mask_l.scatter_(1, mask_indices, 1.0) # 1 = Masked
        
        # We must mask the INPUT X based on these token masks.
        # Use Local Decoder logic (Unmerge) to project mask_l up to mask_n
        # s_local is the ownership map [B, N] pointing to L indices
        mask_n = torch.gather(mask_l, 1, s_local) # [B, N]
        
        # Apply mask to input (use a specific token, e.g., zero or learned mask token)
        # Here we just zero out for simplicity, in production use a learnable mask embedding
        x_masked = x_onehot * (1 - mask_n.unsqueeze(-1))
        
        # Forward masked input
        z_l_masked, s_masked = self.local_encoder(x_masked)
        z_prime_masked = self.latent_encoder(z_l_masked)
        
        # We predict ONLY the masked tokens. 
        # Typically AMTM predicts z_l tokens, but paper implies reconstructing X.
        # Let's use the full decoder path for consistency.
        z_hat_masked = self.latent_decoder(z_prime_masked)
        x_hat_amtm = self.local_decoder(z_hat_masked, s_masked)
        logits_amtm = self.head(x_hat_amtm)
        
        # Compute loss only on masked positions
        loss_amtm = F.cross_entropy(logits_amtm.permute(0, 2, 1), x_indices, reduction='none')
        loss_amtm = (loss_amtm * mask_n).sum() / (mask_n.sum() + 1e-6)

        # Total Loss
        total_loss = loss_mtr + (lambda_latent * loss_latent_mtr) + loss_amtm
        
        return total_loss, {
            "loss_mtr": loss_mtr.item(),
            "loss_latent": loss_latent_mtr.item(),
            "loss_amtm": loss_amtm.item()
        }