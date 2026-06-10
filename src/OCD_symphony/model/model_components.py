import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from torch import eig
from scipy.linalg import eigh
from joblib import Parallel, delayed
from brainspace.gradient import GradientMaps



# Build gradients using diffusion maps and normalized angle
gm = GradientMaps(n_components=30, approach='dm', kernel='normalized_angle')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################
# 1. Positional Encoding #
##########################
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Initialized small
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        """
        x: Shape (batch, time_steps, dim) -> Ensure the embedding dimension is correct
        """
        seq_len, embed_dim = x.shape[1], x.shape[2]
        # return self.ln(x + self.alpha * self.pe[:, :seq_len, :embed_dim].to(x.device))
        return x + self.pe[:, :seq_len, :embed_dim].to(x.device), self.pe[:, :seq_len, :embed_dim].to(x.device)



class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=200):
        super().__init__()
        self.rel_pos = nn.Parameter(torch.randn(2 * max_len - 1, dim))
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, T, D = x.shape
        indices = torch.arange(T, device=x.device)
        rel_indices = indices[None, :] - indices[:, None] + (T - 1)  # (T, T)
        rel_pe = self.rel_pos[rel_indices]  # (T, T, D)
        
        # print("temp att, ",x.mean(), x.std(), rel_pe.mean(), rel_pe.std())
        return self.ln(x + self.alpha * rel_pe.mean(dim=1))  # (B, T, D)



##########################
# 2. Learnable Positional Encoding #
#########################

class LearnablePosEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, gm, x, fc_matrices=None):
        B, N, _ = x.shape
        return self.pe[:, :N, :].expand(B, N, -1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, gm, x, fc_matrices=None):
        B, N, _ = x.shape
        # print("spat att, ",x.mean(), x.std(), self.pe[:, :N, :].mean(), self.pe[:, :N, :].std())
        # return x + self.pe[:, :N, :].expand(B, N, -1)
        pe = self.pe[:, :x.size(1), :].expand(x.size(0), -1, -1)   
        # return self.ln(x + self.alpha * pe)
        return x + pe, pe
        


##############################
# 3 Functional Connectivity #
##############################
def compute_functional_connectivity(fmri_data):
    """
    fmri_data: Tensor of shape (subjects, ROIs, time_steps)
    Returns: Tensor of shape (subjects, ROIs, ROIs)
    """
    subject_fcs = []
    for subj_data in fmri_data:
        subj_data = subj_data.cpu().numpy()
        fc = np.corrcoef(subj_data)
        subject_fcs.append(fc)
    return torch.tensor(subject_fcs, dtype=torch.float32)
    return torch.from_numpy(np.array(subject_fcs)).float()






class BrainGradientPosEncoding(nn.Module):
    def __init__(self, dim, num_gradients=10, delta=0.5):
        super().__init__()
        self.num_gradients = num_gradients
        self.delta = delta
        self.proj = nn.Linear(self.num_gradients, dim)
        # self.gm = GradientMaps(n_components=self.num_gradients, approach='dm', kernel='normalized_angle')

    def forward(self, gm, x, fc_matrices):
        # G_proj = self.proj(gm.to(device))               # Shape: (B, N, dim)
        G_proj = self.proj(gm.to(self.proj.weight.device))  # ✅ always matches model's device
        return G_proj

class BrainGradientPositionalEncoding(nn.Module):
    def __init__(self, dim, num_gradients=10, delta=0.5):
        super().__init__()
        self.num_gradients = num_gradients
        self.delta = delta
        self.proj = nn.Linear(self.num_gradients, dim)
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # self.gm = GradientMaps(n_components=self.num_gradients, approach='dm', kernel='normalized_angle')

    def forward(self, gm, x, fc_matrices):
        # G_proj = self.proj(gm.to(device))               # Shape: (B, N, dim)
        G_proj = self.proj(gm.to(self.proj.weight.device))  # ✅ always matches model's device
        return self.ln(x + self.alpha * G_proj), self.alpha * G_proj



class MixedPositionalEncoding(nn.Module):
    def __init__(self, full_dim, max_len=5000, num_gradients=10, delta=0.5):
        super().__init__()
        half_dim = full_dim // 2
        self.learnable_pe = LearnablePositionalEncoding(half_dim, max_len)
        self.gradient_pe = BrainGradientPositionalEncoding(half_dim, num_gradients, delta)

    def forward(self, gm, x, fc_matrices):
        B, N, D = x.shape
        pe_learnable = self.learnable_pe(gm, x, fc_matrices)
        pe_gradient = self.gradient_pe(gm, x, fc_matrices)
        pe = torch.cat([pe_learnable, pe_gradient], dim=-1)
        return x + pe



class FusedMixedPositionalEncoding(nn.Module):
    def __init__(self, full_dim, max_len=5000, num_gradients=10, delta=0.5):
        super().__init__()
        self.learnable_pe = LearnablePosEncoding(full_dim, max_len)
        self.gradient_pe = BrainGradientPosEncoding(full_dim, num_gradients, delta)
        self.ln = nn.LayerNorm(full_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * full_dim, 2 * full_dim),
            nn.GELU(),
            nn.Linear(2 * full_dim, full_dim)
        )

    def forward(self, gm, x, fc_matrices):
        pe_learnable = self.learnable_pe(gm, x, fc_matrices)
        pe_gradient = self.gradient_pe(gm, x, fc_matrices)
        fused = torch.cat([pe_learnable, pe_gradient], dim=-1)
        # return self.ln(x + self.alpha * self.fusion_proj(fused)), self.alpha * self.fusion_proj(fused)
        return x + self.fusion_proj(fused), self.fusion_proj(fused)
        



class GraphPositionalEncoding(nn.Module):
    def __init__(self, fc_matrix, method='eigen'):
        super().__init__()
        self.method = method
        
        G = nx.from_numpy_array(fc_matrix.cpu().numpy())
        if method == 'eigen':
            eigenvectors = torch.tensor(nx.laplacian_spectrum(G)).float().to(device)
            self.pe = eigenvectors.unsqueeze(0)
        elif method == 'random_walk':
            rw_matrix = torch.tensor(nx.normalized_laplacian_matrix(G).toarray()).float().to(device)
            self.pe = rw_matrix.unsqueeze(0)
        else:
            raise ValueError("Unsupported graph-based positional encoding")
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

############################
# 4. Transformer Components #
############################
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) should be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # Use exponentiation for scale

        # NO .to(device) here. Module creation stays on CPU initially.
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape # Batch, Sequence Length, Hidden Dimension

        qkv = self.qkv(x)  # (B, T, 3 * C)
        
        # Reshape and permute for multi-head operation
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, T, head_dim)

        # Scaled Dot-Product Attention
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale # Use self.scale

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values
        # (B, num_heads, T, T) @ (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        out = attn_weights @ v

        # Concatenate heads and apply final linear layer
        # (B, T, num_heads, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.fc_out(out)

        return out, attn_weights # Return attention output and weights




class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.2):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim) # NO .to(device) here
        self.norm2 = nn.LayerNorm(dim) # NO .to(device) here
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        ) # NO .to(device) here
        self.dropout = nn.Dropout(dropout) # NO .to(device) here

    def forward(self, x):
        # Multi-Head Self-Attention Block (Pre-Norm)
        # Apply LayerNorm before attention
        normed_x_attn = self.norm1(x)
        attn_output, attn_weights = self.attn(normed_x_attn)
        # Apply dropout to attention output, then add residual connection
        x = x + self.dropout(attn_output)

        # MLP Block (Pre-Norm)
        # Apply LayerNorm before MLP
        normed_x_mlp = self.norm2(x)
        mlp_output = self.mlp(normed_x_mlp)
        # Apply dropout to MLP output, then add residual connection
        x = x + self.dropout(mlp_output)
        
        return x, attn_weights





############################
# 5. Transformer Models #
############################
class TemporalTransformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, mlp_dim, time_steps, max_len, num_rois):
        super().__init__()
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # New CLS token
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.pos_enc = SinusoidalPositionalEncoding(dim, max_len)
        # self.pos_enc = RelativePositionalEncoding(dim, max_len)
        self.embedding_layer = nn.Linear(num_rois, dim)  # Ensure correct embedding dim
    
    def forward(self, x):
        """
        x: Shape (batch, ROIs, time_steps)
        - We transpose to (batch, time_steps, ROIs)
        - Apply embedding projection (ROIs → dim)
        - Apply positional encoding
        """
        x = x.transpose(1, 2)  # Shape: (batch, time_steps, ROIs)
        x = self.embedding_layer(x)  # Shape: (batch, time_steps, dim)
        B = x.size(0)
        x, raw_pe = self.pos_enc(x)  # Apply positional encoding

        
        
        # for layer in self.layers:
        #     x = layer(x)
        # return x  # Output: (batch, time_steps, dim)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all.append(attn_weights)
        return x, attn_weights_all, raw_pe



class SpatialTransformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, mlp_dim, time_steps, max_len, num_gradients, delta, pos_enc_type="learnable"):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.embedding_layer = nn.Linear(time_steps, dim)
        

        if pos_enc_type == "learnable":
            self.pos_enc = LearnablePositionalEncoding(dim, max_len)
        elif pos_enc_type == "brain_gradient":
            self.pos_enc = BrainGradientPositionalEncoding(dim, num_gradients, delta)
        elif pos_enc_type == "mixed":
            assert dim % 2 == 0, "Dimension must be divisible by 2 for mixed encoding."
            self.pos_enc = FusedMixedPositionalEncoding(dim, max_len, num_gradients, delta)
        else:
            raise ValueError("Unknown positional encoding type")

        self.pos_enc_type = pos_enc_type
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, gm, x, fc_matrices=None):
        x = self.embedding_layer(x)
        x, raw_pe = self.pos_enc(gm, x, fc_matrices)
        
        # for layer in self.layers:
        #     x = layer(x)
        # return x

        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all.append(attn_weights)
        return x, attn_weights_all, raw_pe




class PerceiverIODecoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, num_latents=32, signal_dim=8, num_queries=450*200, num_heads=2, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        self.conv_signal = nn.Conv1d(in_channels=1, out_channels=signal_dim, kernel_size=5, padding=2)
        self.signal_dim = signal_dim

        self.gate_layer = nn.Linear(2 * input_dim, input_dim)  # input_dim = D


        # Learnable latent array (L, D)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Project ROI-time embeddings to latent dimension
        # self.input_proj = nn.Linear(2*input_dim, latent_dim)
        # self.input_proj = nn.Linear(2*input_dim + signal_dim, latent_dim)
        self.input_proj = nn.Linear(3*128, latent_dim)
        
        self.input_ln = nn.LayerNorm(latent_dim)


        # Cross-attention: latents query the input (fused ROI-Time embeddings)
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Latent processing block (can be stacked)
        self.latent_transformer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, batch_first=True, dropout=dropout
        )

        # Output projection: output queries (R*T positions) attend to latents
        self.output_proj = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Final head: map from latent to scalar value
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )

        self.roi_proj = nn.Linear(256, 128)
        self.time_proj = nn.Linear(256, 128)
        self.signal_proj = nn.Linear(16, 128)

        self.decoding_queries = nn.Parameter(torch.randn(1, num_queries, latent_dim))  # e.g., num_queries = R*T


    def forward(self, roi_embeds, time_embeds, x):
        """
        roi_embeds: (B, R, D)
        time_embeds: (B, T, D)
        returns: reconstructed time series: (B, R, T)
        """
        B, R, D = roi_embeds.shape
        _, T, _ = time_embeds.shape

        # === Step 0: Compute signal context from input time series ===
                     
        signal_flat = x.contiguous().view(B * R, 1, T)        # (B*R, 1, T)
        signal_context = self.conv_signal(signal_flat)         # (B*R, d_signal, T)
        signal_context = signal_context.view(B, R, self.signal_dim, T)  # (B, R, d_signal, T)
        signal_context = signal_context.permute(0, 1, 3, 2)     # (B, R, T, d_signal)


        # === Step 1: Fuse ROI and Time ===
        roi_exp = roi_embeds.unsqueeze(2).expand(-1, -1, T, -1)   # (B, R, T, D)
        time_exp = time_embeds.unsqueeze(1).expand(-1, R, -1, -1) # (B, R, T, D)

        # gate = torch.sigmoid(self.gate_layer(torch.cat([roi_exp, time_exp], dim=-1)))
        # fused = gate * roi_exp + (1 - gate) * time_exp

        # fused = roi_exp + time_exp                                # (B, R, T, D)
        
        # fused = torch.cat([roi_exp, time_exp], dim=-1)  # (B, R, T, 2D)


        # fused = torch.cat([roi_exp, time_exp, signal_context], dim=-1)  # (B, R, T, 2D + d_signal)


        roi_proj = self.roi_proj(roi_exp)           # (B, R, T, 128)
        time_proj = self.time_proj(time_exp)        # (B, R, T, 128)
        signal_proj = self.signal_proj(signal_context)  # (B, R, T, 128)
        
        fused = torch.cat([roi_proj, time_proj, signal_proj], dim=-1)  # (B, R, T, 384)


        # fused = signal_context


        
        fused = fused.contiguous().view(B, R*T, -1)                             # (B, R*T, total_dim)

        # === Step 2: Project input to latent space ===
        # fused_proj = self.input_proj(fused)  # (B, R*T, latent_dim)
        fused_proj = self.input_ln(self.input_proj(fused))

        # === Step 3: Cross-attention (latents query input) ===
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        latents, _ = self.cross_attn(query=latents, key=fused_proj, value=fused_proj)  # (B, L, D)

        # === Step 4: Process latents ===
        latents = self.latent_transformer(latents)  # (B, L, D)

        # subject_embedding = latents.mean(dim=1)  # (B, D)
        # subject_embedding = nn.functional.normalize(subject_embedding, dim=-1)


        # === Step 5: Output queries (R*T) attend to latents ===
        # decoding_queries = fused_proj  # (B, R*T, D)
        # decoding_queries = signal_proj.contiguous().view(B, R*T, -1)   # (B, R*T, D)
        decoding_queries = self.decoding_queries.expand(B, R*T, -1)

        recon_tokens, _ = self.output_proj(query=decoding_queries, key=latents, value=latents)  # (B, R*T, D)

        

        # === Step 6: Final scalar regression per token ===
        recon_flat = self.head(recon_tokens)  # (B, R*T, 1)
        recon = recon_flat.view(B, R, T)      # (B, R, T)

        return recon, latents




######################
# 7. Full Model #
######################



class ROIWiseTransformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, mlp_dim, time_steps, max_len_temp, max_len_spatial, num_gradients, delta, spatial_pos_enc_type="learnable"):
        super().__init__()
        self.temporal_transformer = TemporalTransformer(num_layers, dim, num_heads, mlp_dim, time_steps, max_len_temp, num_rois=450)
        
        # Pass fc_tensor to spatial transformer for brain gradient positional encoding
        self.spatial_transformer = SpatialTransformer(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            time_steps=time_steps,
            max_len=max_len_spatial,
            num_gradients = num_gradients,
            delta = delta,
            pos_enc_type = spatial_pos_enc_type
        )

        # SN comment: those haven't been declared, missing from the original code
        #self.cross_attn_r2t = CrossAttentionFusion(dim, num_heads)
        #self.cross_attn_t2r = CrossAttentionFusion(dim, num_heads)


        self.decoder = PerceiverIODecoder(input_dim=dim, latent_dim=128, num_latents=64, signal_dim=16,num_queries=450*200, num_heads=2)
        

        
    def forward(self, gm, x, fc_batch):
        """
        x: (batch, ROIs, time_steps)
        - Temporal Transformer: (batch, time_steps, ROIs) → (batch, time_steps, dim)
        - Spatial Transformer: (batch, ROIs, time_steps) → (batch, ROIs, dim)
        - Bilinear Decoder: Maps (batch, ROIs, dim) and (batch, time_steps, dim) → (batch, ROIs, time_steps)
        """
  

        time_embeds, temporal_attn_weights,_ = self.temporal_transformer(x)
        roi_embeds, spatial_attn_weights,_ = self.spatial_transformer(gm, x, fc_batch)

        reconstruction, subject_embedding = self.decoder(roi_embeds, time_embeds, x)



        return reconstruction, roi_embeds, time_embeds, spatial_attn_weights, temporal_attn_weights, subject_embedding




class SpatiotemporalMasker(nn.Module):
    def __init__(self, block_size=10, num_blocks=20):
        super().__init__()
        self.block_size = block_size
        self.num_blocks = num_blocks

    def forward(self, x):
        B, R, T = x.shape
        mask = torch.ones((B, R, T), device=x.device)

        # Generate random start indices for each block in each batch
        r_starts = torch.randint(0, R - self.block_size + 1, (B, self.num_blocks), device=x.device)
        t_starts = torch.randint(0, T - self.block_size + 1, (B, self.num_blocks), device=x.device)

        # Generate relative block indices
        r_offsets = torch.arange(self.block_size, device=x.device).view(1, 1, -1)  # (1, 1, block_size)
        t_offsets = torch.arange(self.block_size, device=x.device).view(1, 1, -1)  # (1, 1, block_size)

        # Compute full mask indices
        r_idx = r_starts.unsqueeze(-1) + r_offsets  # (B, num_blocks, block_size)
        t_idx = t_starts.unsqueeze(-1) + t_offsets  # (B, num_blocks, block_size)

        # Broadcast to get all (r, t) coordinate pairs per block
        r_idx = r_idx.unsqueeze(-1).expand(-1, -1, -1, self.block_size)  # (B, num_blocks, block_size, block_size)
        t_idx = t_idx.unsqueeze(-2).expand(-1, -1, self.block_size, -1)  # (B, num_blocks, block_size, block_size)

        # Flatten for advanced indexing
        r_idx = r_idx.reshape(B, -1)  # (B, num_blocks * block_size²)
        t_idx = t_idx.reshape(B, -1)  # (B, num_blocks * block_size²)
        b_idx = torch.arange(B, device=x.device).view(-1, 1).expand(-1, r_idx.size(1))  # (B, num_blocks * block_size²)

        # Set mask to zero at all block positions
        mask[b_idx, r_idx, t_idx] = 0
        return mask


####################
###    MASKERS   ###
####################

class MaskedROIWiseTransformer(nn.Module):
    def __init__(self, fmri_encoder, masker, loss_fn=nn.MSELoss()):
        super().__init__()
        self.fmri_encoder = fmri_encoder
        self.masker = masker
        self.loss_fn = loss_fn

        # Learnable scalar token to replace masked values
        self.mask_token = nn.Parameter(torch.randn(1))

    def forward(self, gm, x, fc_batch):
        B, R, T = x.shape
        mask = self.masker(x)  # (B, R, T), 1 = visible, 0 = masked

        # Apply pairwise masking to the input values
        masked_x = x.clone()
        masked_x[mask == 0] = self.mask_token

       
        recon_x, roi_embeds, time_embeds,  spatial_attn_weights, temporal_attn_weights, subject_embedding = self.fmri_encoder(
            gm, masked_x, fc_batch
        )

        

      
        
        masked_loss = ((recon_x - x) * (1 - mask)).pow(2).sum() / (1 - mask).sum().clamp(min=1)
        visible_loss = ((recon_x - x) * mask).pow(2).sum() / mask.sum().clamp(min=1)
        
        loss =  visible_loss

        return loss, recon_x, mask, roi_embeds, time_embeds, spatial_attn_weights, temporal_attn_weights, subject_embedding



class TokenMasker1D(nn.Module):
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, R, T = x.shape
        mask = torch.ones((B, T), device=x.device)

        num_mask = int(self.mask_ratio * T)
        for b in range(B):
            idx = torch.randperm(T)[:num_mask]
            mask[b, idx] = 0
        return mask  # shape: (B, T)





class MaskedTemporalTransformer(nn.Module):
    def __init__(self, temporal_transformer, masker, roi_dim, hidden_dim=128, loss_fn=nn.MSELoss()):
        super().__init__()
        self.temporal_transformer = temporal_transformer
        self.masker = masker
        self.loss_fn = loss_fn
        self.mask_token = nn.Parameter(torch.randn(1, roi_dim))  # (1, R)

        latent_dim = temporal_transformer.layers[0].attn.qkv.in_features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, roi_dim)
        )

    def forward(self, x):
        B, R, T = x.shape
        mask = self.masker(x)            # (B, T)
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)

        x_transposed = x.transpose(1, 2)     # (B, T, R)
        masked_x = x_transposed * mask_expanded + (1 - mask_expanded) * self.mask_token  # (B, T, R)

        embedded, attn,temporal_pe = self.temporal_transformer(masked_x.transpose(1, 2))  # (B, T, D)

        recon = self.decoder(embedded)   # (B, T, R)
        recon = recon.transpose(1, 2)    # (B, R, T)

        mask_for_loss = mask[:, None, :]  # (B, 1, T)
        loss = ((recon - x) * (1 - mask_for_loss)).pow(2).sum() / ((R*(1 - mask_for_loss)).sum().clamp(min=1))

        return loss, recon, mask, attn, embedded, temporal_pe





class TokenMasker2D(nn.Module):
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        x: (B, R, T)
        Returns: mask of shape (B, R), where 0 = masked, 1 = visible
        """
        B, R, T = x.shape
        mask = torch.ones((B, R), device=x.device)

        num_mask = int(self.mask_ratio * R)
        for b in range(B):
            idx = torch.randperm(R)[:num_mask]
            mask[b, idx] = 0

        return mask  # shape: (B, R)



class MaskedSpatialTransformer(nn.Module):
    def __init__(self, spatial_transformer, masker, time_steps, hidden_dim=128, loss_fn=nn.MSELoss()):
        super().__init__()
        self.spatial_transformer = spatial_transformer
        self.masker = masker
        self.loss_fn = loss_fn
        self.mask_token = nn.Parameter(torch.randn(1, time_steps))  # ✅ each masked ROI is replaced with (1, T)

        latent_dim = spatial_transformer.layers[0].attn.qkv.in_features

        # Decoder: latent D → original time series (T) per ROI
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, time_steps)
        )

    def forward(self, gm, x, fc_batch=None):
        """
        x: (B, R, T)
        """
        B, R, T = x.shape
        mask = self.masker(x)                # (B, R)
        mask_expanded = mask.unsqueeze(-1)   # (B, R, 1)

        mask_token = self.mask_token.to(x.device)  # move to the same device as x
        # Apply mask: masked ROIs get replaced by a learnable T-dim token

        masked_x = x * mask_expanded + (1 - mask_expanded) * mask_token



        # Forward through spatial transformer
        embedded, attn,spatial_pe = self.spatial_transformer(gm, masked_x, fc_batch)  # (B, R, D)

        # Decode to (B, R, T)
        recon = self.decoder(embedded)  # (B, R, T)

        # # Masked loss: only penalize masked ROIs
        loss = ((recon - x) * (1 - mask_expanded)).pow(2).sum() / (((1 - mask_expanded)*T).sum().clamp(min=1))


        return loss, recon, mask, attn, embedded, spatial_pe




class TimePointMasker(nn.Module):
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, R, T = x.shape
        mask = torch.ones((B, R, T), device=x.device)
        num_mask = int(self.mask_ratio * T)
        for b in range(B):
            for r in range(R):
                idx = torch.randperm(T)[:num_mask]
                mask[b, r, idx] = 0
        return mask



class MaskedConv1DTemporalModel(nn.Module):
    def __init__(self, input_channels=1, signal_dim=128, decoder_hidden_dims=[128, 64]):
        super().__init__()

        # Encoder: Conv1D over time for each ROI
        self.conv1d = nn.Conv1d(in_channels=input_channels,
                                out_channels=signal_dim,
                                kernel_size=5,
                                padding=2)  # to preserve time length

        # Learnable token to replace masked time points
        self.mask_token = nn.Parameter(torch.randn(1))

        # Decoder: MLP from signal_dim → scalar prediction for each time point
        decoder_layers = []
        prev_dim = signal_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.GELU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, 1))  # Output: scalar per time point

        self.decoder = nn.Sequential(*decoder_layers)

    def _mlp_decode_in_chunks(self, feats_2d: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
        """
        feats_2d: (N=B*R*T, D)
        Returns: (N, 1)
        """
        outs = []
        # Force float32 for stability, and disable AMP to avoid cublasLt issues
        with torch.autocast("cuda", enabled=False):
            f32 = feats_2d.float()
            for i in range(0, f32.shape[0], chunk):
                outs.append(self.decoder(f32[i:i+chunk]))
        return torch.cat(outs, dim=0)


    def forward(self, x, mask=None):
        B, R, T = x.shape
        device, dtype = x.device, x.dtype
    
        # mask
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool, device=device)
        else:
            if mask.dtype != torch.bool:
                mask = mask > 0
            mask = mask.to(device)
    
        token = self.mask_token.to(device=device, dtype=dtype)
        x_masked = torch.where(mask, x, token.expand_as(x))
    
        # conv
        x_flat = x_masked.contiguous().view(B * R, 1, T)
        features = self.conv1d(x_flat)                        # (B*R, D, T)
        features = features.transpose(1, 2).contiguous()      # (B*R, T, D)
    
        # decode PER (ROI,time) in chunks
        BR, TT, D = features.shape
        feats_2d = features.reshape(BR * TT, D)               # (B*R*T, D)
        decoded  = self._mlp_decode_in_chunks(feats_2d)       # (B*R*T, 1)
        preds    = decoded.view(B, R, T)                      # (B, R, T)
    
        # return
        features = features.view(B, R, T, D)
        return preds, features








class RandomTokenMasker:
    def __init__(self, mask_ratio=0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        """
        x: (B, R, T, D) — just for shape reference
        Returns:
            mask: (B, R, T) with 1 = keep, 0 = mask
        """
        B, R, T, _ = x.shape
        mask = torch.rand(B, R, T, device=x.device) > self.mask_ratio
        return mask.float()





class PerceiverIOModelROIWise(nn.Module):
    def __init__(self,
                 input_dim,          # input feature dim (e.g. 3×128 = 384)
                 latent_dim=64,
                 num_rois=450,
                 num_heads=8,      # Increased num_heads for demonstration
                 time_steps=200,
                 dropout=0.1,      # Using 0.1 for dropout
                 num_transformer_blocks=1): # Added option to stack transformer blocks
        super().__init__()
        self.latent_dim = latent_dim
        self.num_rois = num_rois
        self.time_steps = time_steps

        # One latent per ROI
        self.latents = nn.Parameter(torch.randn(num_rois, latent_dim))

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_ln = nn.LayerNorm(latent_dim)

        # Cross-attention: latents attend to full input (flattened R×T)
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim,
                                                 num_heads=num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(latent_dim) # Added LayerNorm after cross-attn

        # Latent processing block(s) using our custom TransformerBlock
        # Common practice is to set mlp_dim = 4 * dim
        mlp_dim_for_transformer_block = latent_dim * 4
        
        self.latent_transformers = nn.ModuleList([
            TransformerBlock(dim=latent_dim, num_heads=num_heads, 
                             mlp_dim=mlp_dim_for_transformer_block, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Final head: map each latent to time series of length T
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.GELU(),
            nn.Linear(latent_dim*2, time_steps) # Output is (B, R, T)
        )

        # Optional: projections before fusion (same as your original)
        # Assuming input_dim = 3 * 128 = 384 based on your comment
        # So input_dim // 3 = 128
        self.roi_proj = nn.Linear(128, input_dim // 3) # This might be wrong if input_dim is not 3*128
        self.time_proj = nn.Linear(128, input_dim // 3)
        self.signal_proj = nn.Linear(64, input_dim // 3) # This might be wrong if input_dim is not 3*128

     # ---- NEW: safe linear apply in chunks (float32, out of AMP) ----
    @staticmethod
    def _apply_linear_in_chunks(x2d: torch.Tensor, layer: nn.Linear, chunk: int = 200_000) -> torch.Tensor:
        """
        x2d: (N, Din) on CUDA
        returns: (N, Dout)
        """
        outs = []
        with torch.autocast(device_type="cuda", enabled=False):
            x = x2d.float()  # force fp32
            w = layer.weight.float()
            b = layer.bias.float() if layer.bias is not None else None
            for i in range(0, x.shape[0], chunk):
                outs.append(F.linear(x[i:i+chunk], w, b))
        y = torch.cat(outs, dim=0)
        return y.to(x2d.dtype)  # restore original dtype (e.g., bf16/fp16) if needed


    def forward(self, roi_embeds, time_embeds, context_features, loaded_roi_pos):
        """
        roi_embeds:       (B, R, D_roi)
        time_embeds:      (B, T, D_time)
        context_features: (B, R, T, D_signal)
        loaded_roi_pos:   (B, R, latent_dim)
        """
        B, R, D_roi  = roi_embeds.shape
        _, T, D_time = time_embeds.shape

        # Expand to (B, R, T, *)
        roi_exp  = roi_embeds.unsqueeze(2).expand(-1, -1, T, -1)
        time_exp = time_embeds.unsqueeze(1).expand(-1, R, -1, -1)

        # (Keep these three proj layers consistent with input_dim//3!)
        roi_proj    = self.roi_proj(roi_exp)
        time_proj   = self.time_proj(time_exp)
        signal_proj = self.signal_proj(context_features)

        fused = torch.cat([roi_proj, time_proj, signal_proj], dim=-1)   # (B, R, T, input_dim)
        fused = fused.contiguous()

        # ---- CRITICAL: flatten to 2D and apply input_proj in chunks ----
        fused_2d   = fused.view(B * R * T, -1)                          # (B*R*T, input_dim)
        proj_2d    = self._apply_linear_in_chunks(fused_2d, self.input_proj)  # (B*R*T, latent_dim)
        fused_proj = proj_2d.view(B, R * T, self.latent_dim)            # (B, R*T, latent_dim)
        fused_proj = self.input_ln(fused_proj)

        # Cross-attention: latents (B, R, D) attend to keys/values (B, R*T, D)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1) + loaded_roi_pos
        latents, _ = self.cross_attn(query=latents, key=fused_proj, value=fused_proj)
        latents = self.cross_attn_norm(latents)
        
        # Latent transformer blocks
        latent_attention_weights = []
        for block in self.latent_transformers:
            latents, attn_w = block(latents)
            latent_attention_weights.append(attn_w)

        # Decode to (B, R, T)
        recon = self.head(latents)

        return recon, latents, latent_attention_weights






class PerceiverClassifier(nn.Module):
    def __init__(self, pretrained_perceiver: PerceiverIOModelROIWise, latent_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.perceiver = pretrained_perceiver

        self.reduction = nn.Sequential(nn.Linear(latent_dim, 30))

       

        self.classifier = nn.Sequential(
            nn.Linear(30*450, 64),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(16, num_classes)  # Binary classification logits
        )
        
      

    def forward(self, roi_embeds, time_embeds, context_features, roi_pos):
        recon, latents,_ = self.perceiver(roi_embeds, time_embeds, context_features, roi_pos)  # (B, R, D)
        
        # global_latent = latents.mean(dim=1)  # mean-pool across ROIs (B, D)
        
        reduced = self.reduction(latents)  
        global_latent = reduced.reshape(reduced.size(0), -1)  # (B, 450*10)
        
        logits = self.classifier(global_latent)  # (B, num_classes)
        return logits, recon





class PerceiverRegressor(nn.Module):
    """
    Tiny head for N=~50 subjects:
      Perceiver -> (B, R, D) latents
      ROI->Network mean-pooling -> (B, Nnet, D)
      Attention over networks -> (B, D)
      Small MLP -> scalar
    """
    def __init__(
        self,
        pretrained_perceiver,
        roi_labels: np.ndarray,          # shape (R,), ints in [0..Nnet-1]
        latent_dim: int = 128,
        freeze_perceiver: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.perceiver = pretrained_perceiver
        R = int(roi_labels.shape[0])
        nets = np.unique(roi_labels)
        self.num_networks = int(len(nets))

        # Build a (Nnet, R) pooling matrix W where each row averages ROIs of that network
        W = np.zeros((self.num_networks, R), dtype=np.float32)
        for n in nets:
            idx = (roi_labels == n)
            cnt = idx.sum()
            if cnt > 0:
                W[int(n), idx] = 1.0 / cnt
        # Register as buffer so it moves with the model and is not trainable
        self.register_buffer("W", torch.from_numpy(W))   # (Nnet, R)

        if freeze_perceiver:
            for p in self.perceiver.parameters():
                p.requires_grad = False

        # Attention over networks (works on per-network mean latents)
        self.net_attn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)   # score per network
        )

        # Final tiny regressor head
        self.regressor = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, roi_embeds, time_embeds, context_features, roi_pos):
        """
        Perceiver forward should return: recon, latents, ...
        latents: (B, R, D)
        """
        recon, latents, _ = self.perceiver(roi_embeds, time_embeds, context_features, roi_pos)  # latents: (B,R,D)

        # Mean-pool ROIs within each network with fixed W: (Nnet,R) x (B,R,D) -> (B,Nnet,D)
        # einsum is clear & fast: pooled[b,n,d] = sum_r W[n,r] * latents[b,r,d]
        pooled = torch.einsum("nr,brd->bnd", self.W, latents)   # (B, Nnet, D)

        # Attention over networks
        attn_scores = self.net_attn(pooled).squeeze(-1)         # (B, Nnet)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, Nnet, 1)

        # Weighted sum → single global latent per subject
        global_latent = (pooled * attn_weights).sum(dim=1)      # (B, D)

        pred = self.regressor(global_latent)                    # (B, 1)
        return pred, recon, attn_weights.squeeze(-1)            # return weights for plotting


#################
###   UTILS   ###
#################

def compute_fc_fast(fmri_tensor):
    B, R, T = fmri_tensor.shape
    x = fmri_tensor - fmri_tensor.mean(dim=-1, keepdim=True)
    x = x / (fmri_tensor.std(dim=-1, keepdim=True) + 1e-8)
    return torch.bmm(x, x.transpose(1, 2)) / T
    # return torch.bmm(fmri_tensor, fmri_tensor.transpose(1, 2)) / T



def functional_connectivity_loss(spatial_embeddings, fc_batch):
    reconstructed_fc = torch.bmm(spatial_embeddings, spatial_embeddings.transpose(1, 2))
    return F.mse_loss(reconstructed_fc, fc_batch)




def roi_diversity_loss(roi_embeddings):
    subj_mean = roi_embeddings.mean(dim=1)  # (B, D)
    return - subj_mean.var(dim=0).mean()  # encourage variance across subjects


def time_diversity_loss(time_embeddings):
    subj_mean = time_embeddings.mean(dim=1)  # (B, D)
    return - subj_mean.var(dim=0).mean()



### 2. Temporal Smoothness Loss

def temporal_smoothness_loss(temporal_embeddings):
    """
    Encourages smoothness in the learned temporal embeddings.
    temporal_embeddings: (batch, T, d)
    """
    return torch.mean((temporal_embeddings[:, 1:, :] - temporal_embeddings[:, :-1, :]) ** 2)
    

### 3. Reconstruction with Time-Series Statistics Preservation

def statistics_preservation_loss(original_data, reconstructed_data):
    """
    Ensures that the mean and variance of the reconstructed data match the original.
    original_data: (batch, ROIs, T)
    reconstructed_data: (batch, ROIs, T)
    """
    return torch.mean((original_data.mean(dim=-1) - reconstructed_data.mean(dim=-1)) ** 2) + \
           torch.mean((original_data.std(dim=-1) - reconstructed_data.std(dim=-1)) ** 2)
    

### 4. ROI-Specific Consistency Loss

def roi_consistency_loss(spatial_embeddings):
    """
    Ensures stability of spatial embeddings across subjects.
    spatial_embeddings: (batch, ROIs, d)
    """
    mean_embeddings = spatial_embeddings.mean(dim=0, keepdim=True)  # (1, ROIs, d)
    return torch.mean((spatial_embeddings - mean_embeddings) ** 2)




def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Args:
        features: (B, D) tensor of embeddings
        labels: (B,) tensor of subject IDs
    """
    device = features.device
    B = features.shape[0]

    # Safe normalization
    features = features / (features.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity
    sim_matrix = torch.matmul(features, features.T) / temperature

    # Mask out self-similarities
    mask = torch.eye(B, dtype=torch.bool).to(device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    # Positive pairs
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T).float().to(device)
    positive_mask = positive_mask.masked_fill(mask, 0)

    # Log-softmax
    log_prob = F.log_softmax(sim_matrix, dim=1)

    # Final loss
    loss = -(positive_mask * log_prob).sum() / (positive_mask.sum().clamp(min=1.0))

    return loss


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)