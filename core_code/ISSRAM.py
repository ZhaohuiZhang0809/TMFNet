import torch
import torch.nn.functional as F
from einops import einsum, repeat
from functorch.einops import rearrange
from thop import profile
from torchinfo import summary

from InfoNCE import InfoNCE
from models.common import MultiCrossAttention
from torch import nn, no_grad
from models.configs.RecMamba import Mamba as RecMamba, ModelArgs
from models.common import Mlp


class FCGAM(nn.Module):
    def __init__(self, in_dim, n_layer=1, d_state=16, expand=2):
        super(FCGAM, self).__init__()
        self.CrossAttention1 = MultiCrossAttention(embedding_dim=in_dim, num_heads=3)
        self.CrossAttention2 = MultiCrossAttention(embedding_dim=in_dim, num_heads=3)
        # self.Mlp1 = Mlp(in_features=in_dim)
        # self.Mlp2 = Mlp(in_features=in_dim)

        self.args = ModelArgs(
            d_model=in_dim,  # Hidden layer dimension
            n_layer=n_layer,  # Number of layers
            d_state=d_state,  # State space dimension
            expand=expand,  # Expansion factor
        )

        self.RecMamba = RecMamba(self.args)

        # dt_proj projects Δ from dt_rank to d_in
        self.x_proj = nn.Linear(in_dim, self.args.dt_rank + self.args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.args.dt_rank, self.args.d_inner, bias=True)

        A = repeat(torch.arange(1, self.args.d_state + 1), 'n -> d n', d=self.args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.args.d_inner))
        self.in_proj = nn.Linear(in_dim, in_dim * 2, bias=self.args.bias)
        self.out_proj = nn.Linear(self.args.d_inner, self.args.d_model, bias=self.args.bias)

    def token_masking(self, x, mode='max', mask_value=0):
        """
        Args:
            x: [B, L, C] Input tensor
            k: Number of top-K elements to keep
            mode: Aggregation method ('max', 'mean', 'norm')
            mask_value: Replacement value (e.g., 0 or [MASK] embedding)
        Returns:
            maske_values: Masked token values [B, k, C]
            unmaske_values: Unmasked token values [B, L-k, C]
            topk_indexs: Indices of top-K elements [B, k]
        """
        B, L, C = x.shape
        k = L // 16

        # 1. Calculate activation values and get topk indices
        if mode == 'max':
            activation = x.max(dim=2).values  # [B, L]
        elif mode == 'mean':
            activation = x.mean(dim=2)       # [B, L]
        elif mode == 'norm':
            activation = x.norm(dim=2)       # [B, L]

        _, topk_indexs = torch.topk(activation, k=k, dim=1)  # [B, k]

        # 2. Generate boolean mask
        bool_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        for b in range(B):
            bool_mask[b, topk_indexs[b]] = True

        # 3. Save original values
        maske_values = torch.zeros(B, k, C, device=x.device)
        unmaske_values = torch.zeros(B, L - k, C, device=x.device)
        for b in range(B):
            maske_values[b] = x[b, bool_mask[b]]
            unmaske_values[b] = x[b, ~bool_mask[b]]

        return maske_values, unmaske_values, topk_indexs

    def compute_loss(self):
        """Calculate contrastive loss between image-to-text and text-to-image features"""
        text2img = self.text2img  # [B, L, C]
        img2text = self.img2text
        loss_fn = InfoNCE(temperature=0.05)

        contrast_loss = loss_fn(
            img2text.mean(dim=1, keepdim=True).squeeze(1),
            text2img.mean(dim=1, keepdim=True).squeeze(1)
        )

        return contrast_loss

    def recons_tokens(self, x, x_vis, indexs):
        """Reconstruct masked tokens using autoregressive generation"""
        steps = x.size(1) - x_vis.size(1)
        # Precompute static parameters
        A = -torch.exp(self.A_log.float())  # (d_in, n)
        D = self.D.float()
        d_in, n = A.shape
        B, L, C = x_vis.shape

        # Get initial state
        logits, hidden_state = self.RecMamba(x_vis)
        next_token = logits[:, -1:, :]
        x_vis = torch.cat([x_vis, next_token], dim=1)

        # Optimized autoregressive loop
        for _ in range(steps - 1):
            # Parallel compute all projections
            x_dbl = self.x_proj(x_vis)
            x_proj = self.in_proj(x_vis)

            # Optimized split operation
            delta, B_mat, C_mat = torch.split(
                x_dbl,
                [self.args.dt_rank, n, n],
                dim=-1
            )
            delta = F.softplus(self.dt_proj(delta))

            # Einsum computations
            deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
            deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B_mat, x_proj[..., :d_in])

            # Update state
            hidden_state = deltaA[:, -1] * hidden_state + deltaB_u[:, -1]

            # Predict next token
            next_token = torch.einsum('bdn,bn->bd', hidden_state, C_mat[:, -1, :])
            next_token = self.out_proj(next_token).unsqueeze(1)
            x_vis = torch.cat([x_vis, next_token], dim=1)

        full_seq = x
        # Reorder according to original indexes
        if indexs is not None:
            for i in range(B):
                full_seq[i][indexs] = x_vis[i][:steps, :]

        return full_seq

    def forward(self, img_x, text_x):
        """Forward pass for cross-modal attention and feature fusion"""
        B, C, H, W = img_x.shape
        # Cross-modal fusion
        img_x = rearrange(img_x, "b c h w -> b (h w) c")  # B, L, C
        
        # Compute text-to-image attention
        text2img = self.CrossAttention2(img_x, text_x)

        # Reshape back to image dimensions
        text2img = rearrange(text2img, " b (h w) c -> b c h w", h=H, w=W)

        return text2img
