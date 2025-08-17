from typing import Any
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

from models.common import *
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from functools import partial
import torch
from einops import rearrange
import math
from thop import profile
from torch import nn
from torchinfo import summary

from models.configs.MambaScanner import mamba_init, MambaScanner
from models.DynamicMasking import DynamicMasking


# class FreqConvModule(nn.Module):
#     "" Frequency domain convolution module """
#     def __init__(self, in_channels, out_channels):
#         super(FreqConvModule, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         # 1x1 convolution layer
#         self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         # Transform the input tensor to the frequency domain
#         x_fft = torch.fft.fft2(x, dim=(2, 3))  # Perform 2D Fourier transform on the input tensor
#         x_fft = self.conv1(x_fft.real)
#         x_fft = self.relu(x_fft.real)
#         x_fft = self.conv2(x_fft.real)
#         x_ifft = torch.fft.ifft2(x_fft, dim=(2, 3)).real
#
#         return x_ifft

# # Define global convolution module
# class GlobalConvModule(nn.Module):
#     def __init__(self, in_dim, out_dim, kernel_size):
#         super(GlobalConvModule, self).__init__()
#         pad0 = (kernel_size[0] - 1) // 2
#         pad1 = (kernel_size[1] - 1) // 2
#         self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
#                                  padding=(pad0, 0))
#         self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
#                                  padding=(0, pad1))
#         self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
#                                  padding=(0, pad1))
#         self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
#                                  padding=(pad0, 0))
#
#     def forward(self, x):
#         x_l = self.conv_l1(x)
#         x_l = self.conv_l2(x_l)
#         x_r = self.conv_r1(x)
#         x_r = self.conv_r2(x_r)
#         x = x_l + x_r
#         return x


# # Sparse channel attention
# class SparseChannelAttention(nn.Module):
#     def __init__(self, channel, reduction_ratio=16, k_ratio=0.5):
#         """
#         Args:
#             channel (int): Number of input channels
#             reduction_ratio (int): Intermediate layer compression ratio (default 16)
#             k_ratio (float): Sparse ratio, the proportion of top-K channels to select (0 < k_ratio <=1)
#         """
#         super(SparseChannelAttention, self).__init__()
#         self.channel = channel
#         self.k = max(1, int(channel * k_ratio))  # At least keep 1 channel
#         self.reduction_ratio = reduction_ratio
#
#         # Global average pooling
#         self.gap = nn.AdaptiveAvgPool2d(1, 1)
#
#         # Intermediate layer (fully connected + ReLU)
#         self.mlp = nn.Sequential(
#             nn.Linear(channel, channel // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction_ratio, channel)  # Output channel importance scores
#         )
#
#         # Global learnable parameters (to enhance unselected channels)
#         self.global_weight = nn.Parameter(torch.ones(1, channel, 1, 1))
#         self.global_bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # 1. Calculate channel importance scores
#         scores = self.gap(x).view(b, c)  # [B, C]
#         scores = self.mlp(scores)        # [B, C]
#
#         # 2. Create sparse mask (Top-K selection)
#         _, topk_indices = torch.topk(scores, self.k, dim=1)  # [B, K]
#         sparse_mask = torch.zeros_like(scores)               # [B, C]
#         sparse_mask.scatter_(1, topk_indices, 1.0)           # Only top-K positions are 1
#
#         # 3. Calculate attention weights (only for top-K channels)
#         attention = torch.sigmoid(scores) * sparse_mask      # [B, C]
#
#         # 4. Apply attention weights + global parameters
#         attention = attention.view(b, c, 1, 1)               # [B, C, 1, 1]
#         out = x * ((attention * self.global_bias + 1) * self.global_weight)
#         return out


# support: v0, v0seq
class SS2Dv0:
    def __init__(
            self,
            # basic dims ============
            d_model=96,
            topk=4,
            mlp_ratio=4.0,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================#
            dropout=0.0,
            # ======================#
            seq=False,
            force_fp32=True,
            windows_size=3,
            depth=1,
            **kwargs,
    ):
        """ V-Mamba-v0 framework
        Arg:
            d_model: Output dimension of the model (default 9).
            d_state: State dimension (default 16).
            ssm_ratio: Ratio of state dimension to model dimension (default 2.0).
            dt_rank: Dimension of dynamic parameters, default "auto", calculated based on d_model
        """

        # Spatial dimension first
        # if "channel_first" in kwargs:
        #     assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        # act_layer = nn.ReLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 7
        k_group = 2     # Scan direction
        num_heads = 8
        dilation = [1, 2, 3]

        self.dim = d_model
        self.windows_size = windows_size
        # self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_dilation = len(dilation)

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # self.selective_scan = selective_scan_fn  # Selective scan (acceleration)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        # self.in_proj = nn.Conv2d(d_model, d_inner * 2, kernel_size=3, padding=1)


        self.act: nn.Module = act_layer
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        init_dt_A_D = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=k_group,
        )

        top_k = max(1, (20 // windows_size) ** 2) * depth
        self.kv = nn.Linear(d_inner, d_inner * 2)

        # mask
        self.DynamicMasking = DynamicMasking(window_size=windows_size, top_k=top_k)

        self.average_pool = nn.AdaptiveAvgPool1d(1, 1)

        # batch, length, force_fp32, seq, k_group, inner, rank
        self.mambaScanner = MambaScanner(seq=seq, force_fp32=force_fp32, init_dt_A_D=init_dt_A_D, x_proj_weight=self.x_proj_weight)

        # # Mutil-conv
        # self.GConv = GlobalConvModule(in_dim=d_model, out_dim=d_model, kernel_size=(7, 7))
        # self.LocalConv = DWConv(d_model, d_model)
        # self.FreqConv = FreqConvModule(in_channels=d_model, out_channels=d_model)
        #
        # # Sparse Channel Attention
        # self.SCA = SparseChannelAttention(d_model)

        # out proj ========================================
        self.out_norm = nn.LayerNorm(d_inner)
        # self.out_norm = nn.BatchNorm2d(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        # self.out_proj = nn.Conv2d(d_inner, d_model, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # partition windows
        self.unfold = nn.ModuleList([
            nn.Unfold(
                kernel_size=windows_size,
                dilation=d,
                padding=(d * (windows_size - 1) + 1) // 2,
                stride=1
            ) for d in dilation
        ])  #  will generate equally divided dilated windows, (B, C, H, W) -> (B, C*window_size*window_size, n)

        # # channel transfer
        # self.gap = nn.AdaptiveAvgPool2d(1, 1)
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(d_model, d_model // int(mlp_ratio), bias=False),
        #     nn.ReLU(),
        #     nn.Linear(d_model // int(mlp_ratio), d_model, bias=False),
        #     nn.Sigmoid()
        # )

    # def batch_cosine_similarity(self, A, B):
    #     """
    #     Args:
    #         A: Tensor of shape (B, C, 1)
    #         B: Tensor of shape (B, C, L)
    #     Returns:
    #         Similarity matrix of shape (B, 1, L)
    #     """
    #     # A_expanded = A.expand(-1, -1, B.size(-1))  # Expand to (B, C, L)
    #     A_normalized = F.normalize(A, p=2, dim=1)
    #     B_normalized = F.normalize(B, p=2, dim=1)
    #     return torch.sum(A_normalized * B_normalized, dim=1, keepdim=True)


    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)

        x, z = x.chunk(2, dim=-1)  # (b, d, h, w)
        # z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # (b, d, h, w)
        # x = self.act(x)

        B, D, H, W = x.shape
        # L = H * W

        """ Partition based on activation values """
        active_feat, passive_feat, only_active_feat, only_passive_feat = self.DynamicMasking(x) # (B, C, H, W), ~; (B, C, nW*window_size*window_size)

        """ Local traversal path """
        '''SSM -> Foreground class feature '''
        # partition
        xs_fore = only_active_feat.unsqueeze(dim=1)
        # concatenate x_s and its flip
        xs_fore = torch.cat([xs_fore, torch.flip(xs_fore, dims=[-1])], dim=1)  # (B, 2, C, nW*window_size*window_size)
        # selective scan
        out = self.mambaScanner(xs_fore)
        # """ Combine two traversal paths (after Mamba) """
        # token position restoration
        inv_1 = torch.flip(out[:,1:2], dims=[-1])
        # combine two states, add projection weights
        y1 = inv_1[:, 0] + out[:, 0]   # (B, C, nW*window_size*window_size)
        # foreground_class = self.average_pool(y1) # (B, C, 1)

        # restore shape for output
        y1 = y1.transpose(dim0=1, dim=2).contiguous()

        '''Foreground multi-scale convolution'''
        # Local_feature = self.LocalConv(active_feat)
        # Freq_feature = self.FreqConv(active_feat)

        # # Multi-scale foreground aggregation attention
        SSM_Q, SSM_K = rearrange(self.kv(y1),"b (n2 ws) (h_n h_dim) -> h_n b h_dim ws2", ws=self.windows_size ** 2).unsqueeze(-1).chunk(
                                    2, dim=2)  # (B, N2*window_size*window_size, C) -> (h_n, B, h_dim, window_size*window_size, N2,1)

        x = x.reshape(B, self.num_dilation, D // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)

        y_list = []
        for i in range(self.num_dilation):
            Win_V = self.unfold[i](x[i])
            Win_V = rearrange(Win_V, 'b (h_dim ws) n1 -> b h_dim n1 ws', h_dim=D // self.num_dilation)  # (B, h_dim, N1, window_size*window_size)

            A_M = SSM_Q[i].transpose(-1, -2) @ SSM_K[i] / math.sqrt(D // self.num_dilation)  # (B, h_dim, window_size*window_size, 1)
            A_M = F.softmax(A_M, dim=-1).squeeze(-1)
            y = Win_V @ A_M     # (B, h_dim, N1)
            y_list.append(y.squeeze(-1))

        y = torch.cat(y_list, dim=1)
        y = y.transpose(-1, -2)

        # y = y * self.fc(foreground_class.squeeze(-1)).view(B, D, 1, 1)
        # fore_sim = self.batch_cosine_similarity(foreground_class, y.view(B, D, H * W)).view(B, -1, H, W)

        # foreground = active_feat * torch.sigmoid(Local_feature + Freq_feature) * fore_sim

        # '''Background SSM -> Background class feature'''
        # # partition
        # xs_back = only_passive_feat.unsqueeze(dim=1)
        # # concatenate x_s and its flip
        # xs_back = torch.cat([xs_back, torch.flip(xs_back, dims=[-1])], dim=1)  # (B, 2, C, nW*window_size*window_size)
        # # selective scan
        # out_2 = self.mambaScanner(xs_back)
        # # """ Combine two traversal paths (after Mamba) """
        # # token position restoration
        # inv_2 = torch.flip(out_2[:, 1:2], dims=[-1])
        # # combine two states, add projection weights
        # y2 = inv_2[:, 0] + out_2[:, 0]      # (B, C, nW*window_size*window_size)
        background_class = self.average_pool(y2) # (B, C, 1)

        # # SSM
        # self.GConv = GlobalConvModule(in_dim=d_model, out_dim=d_model, kernel_size=(7, 7))
        # self.LocalConv = DWConv(d_model, d_model)
        # self.FreqConv = FreqConvModule(in_channels=d_model, out_channels=d_model)
        #
        # # Sparse Channel Attention
        # self.SCA = SparseChannelAttention(d_model)

        # out proj ========================================
        self.out_norm = nn.LayerNorm(d_inner)
        # self.out_norm = nn.BatchNorm2d(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        # self.out_proj = nn.Conv2d(d_inner, d_model, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # partition windows
        self.unfold = nn.ModuleList([
            nn.Unfold(
                kernel_size=windows_size,
                dilation=d,
                padding=(d * (windows_size - 1) + 1) // 2,
                stride=1
            ) for d in dilation
        ])  #  will generate equally divided dilated windows, (B, C, H, W) -> (B, C*window_size*window_size, n)

        # # channel transfer
        # self.gap = nn.AdaptiveAvgPool2d(1, 1)
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(d_model, d_model // int(mlp_ratio), bias=False),
        #     nn.ReLU(),
        #     nn.Linear(d_model // int(mlp_ratio), d_model, bias=False),
        #     nn.Sigmoid()
        # )

    # def forward(self, x: torch.Tensor):
    #     x = self.patch_embed(x)
    #     x = rearrange(x, 'b c h w -> b h w c')
    #     if self.pos_embed is not None:
    #         pos_embed = self.pos_embed.permute(0, 2, 3, 1) if self.channel_first else self.pos_embed
    #         x = x + pos_embed
    #     ''' Stack modules '''
    #     x_list = []
    #     for layer in self.layers:
    #         x_list.append(rearrange(x, 'b h w c -> b c h w'))
    #         x = layer(x)

    #     x = rearrange(x, 'b h w c -> b c h w')
    #     return x, x_list


class SS2D(nn.Module, SS2Dv0):
    def __init__(
            self,
            # basic dims ============
            d_model=96,
            windows_size=2,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            mlp_ratio=4.0,
            # ======================#
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================#
            dropout=0.0,
            bias=False,
            # ======================#
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================#
            forward_type="v2",
            channel_first=False,
            depth=1,
            # ======================#
            **kwargs,
    ):
        """ Initialize SS2D
        Arg:
            d_model, d_state: Model dimension and state dimension, affecting feature representation size
            ssm_ratio: Ratio of state space model, may affect model complexity
            dt_rank: Dimension of dynamic parameters, controls how time sequences are processed
            act_layer: Activation function for SSM, default SiLU (Sigmoid Linear Unit)
            d_conv: Convolution layer dimension, value less than 2 means no convolution
            conv_bias: Whether to use convolution bias
            dropout: Dropout probability to prevent overfitting
            bias: Whether to use bias in the model
            dt_min, dt_max: Minimum and maximum values of time steps
            dt_init: Initialization method for time steps, can be "random", etc.
            dt_scale, dt_init_floor: Affect time step scaling and lower limit
            initialize: Specify initialization method
            forward_type: Decides the implementation of forward propagation, supports multiple types
            channel_first: Indicates whether to use channel-first format
            **kwargs: Allows passing additional parameters for easy extension
        """
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, windows_size=windows_size, mlp_ratio=mlp_ratio, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, depth=depth
        )

        # Call different initialization functions
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        # else:
        #     self.__initv1__(**kwargs)


# =====================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            windows_size: int = 4,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            m_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            _SS2D: type = SS2D,
            depth=1,
            **kwargs,
    ):
        r""" VMamba overall architecture
        Arg:
            Dimension transformation parameters:
                hidden_dim: Input and output feature dimension
                drop_path: Probability for random path dropping to prevent overfitting
                norm_layer: Normalization layer, default LayerNorm
                channel_first: Data format, indicates whether to use channel-first
            SSM related parameters:
                ssm_d_state: State dimension of state space model
                ssm_ratio: Determines whether to use SSM
                ssm_dt_rank: Dimension of dynamic parameters
                ssm_act_layer: Activation function for SSM, default SiLU
                ssm_conv: Convolution layer size
                ssm_conv_bias: Whether to use convolution bias
                ssm_drop_rate: Dropout probability in SSM
                ssm_init: Initialization method for SSM
                forward_type: Decides the implementation of forward propagation
            MLP related parameters:
                mlp_ratio: Hidden layer to input layer dimension ratio
                mlp_act_layer: Activation function for MLP, default GELU
                mlp_drop_rate: Dropout probability in MLP
                gmlp: Whether to use GMLP structure
            Other parameters:
                use_checkpoint: Whether to use gradient checkpointing to save memory
                post_norm: Whether to normalize after adding residual connection
                _SS2D: Type of state space model class
        """

        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        ''' SSM module, initialized to V0 version '''
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS2D(
                d_model=hidden_dim,
                windows_size=windows_size,
                d_state=m_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                mlp_ratio=mlp_ratio,
                # ==================================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==================================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==================================
                forward_type=forward_type,
                channel_first=channel_first,
                depth = depth,
            ))

        self.drop_path = DropPath(drop_path)

        ''' MLP '''
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


# Main function
class SCA_SSM(nn.Module):
    """ Spatial Context-Aware State Space Module """
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            windows_size=2,
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            # =========================
            posembed=False,
            imgsize=224,
            _SS2D=SS2D,
            # =========================
            depth = [1, 1, 1],
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.input_resolution = [(10, 10), (20, 20), (40, 40), (80, 80)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        #
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)  # Choose embedding function based on patchembed version
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_down_sample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                windows_size=windows_size,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1)]),
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,                  # Three versions of downsampling
                channel_first=self.channel_first,
                # =========================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =========================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =========================
                _SS2D=_SS2D,
                depth = depth[i_layer],
            ))

        # self.classifier = nn.Sequential(OrderedDict(
        #     norm=norm_layer(self.num_features),  # B,H,W,C
        #     permute=Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity(),
        #     avgpool=nn.AdaptiveAvgPool2d(1),
        #     flatten=nn.Flatten(1),
        #     head=nn.Linear(self.num_features, num_classes),
        # ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                      channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                      channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_down_sample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_down_sample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            windows_size=2,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            _SS2D=SS2D,
            depth=1,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                windows_size=windows_size,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type='v0' if d % 2 == 0 and forward_type == ['v0', 'v1'] else ('v1' if forward_type == ['v0', 'v1'] else forward_type),
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
                depth = depth,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        # x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if self.channel_first else self.pos_embed
            x = x + pos_embed
        ''' Stack modules '''
        x_list = []
        for layer in self.layers:
            x_list.append(rearrange(x, 'b h w c -> b c h w'))
            x = layer(x)

        x = rearrange(x, 'b h w c -> b c h w')
        return x, x_list


class SCA_SSM(nn.Module):
    "" Spatial Context-Aware State Space Module """
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            windows_size=2,
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            # =========================
            posembed=False,
            imgsize=224,
            _SS2D=SS2D,
            # =========================
            depth = [1, 1, 1],
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.input_resolution = [(10, 10), (20, 20), (40, 40), (80, 80)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        #
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)  # Choose embedding function based on patchembed version
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_down_sample,
            v3=self._make_down_sample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                windows_size=windows_size,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,                  # Three versions of downsampling
                channel_first=self.channel_first,
                # =========================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =========================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =========================
                _SS2D=_SS2D,
                depth = depth[i_layer],
            ))

        # self.classifier = nn.Sequential(OrderedDict(
        #     norm=norm_layer(self.num_features),  # B,H,W,C
        #     permute=Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity(),
        #     avgpool=nn.AdaptiveAvgPool2d(1),
        #     flatten=nn.Flatten(1),
        #     head=nn.Linear(self.num_features, num_classes),
        # ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                        channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                        channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_down_sample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_down_sample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            windows_size=2,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            _SS2D=SS2D,
            depth=1,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                windows_size=windows_size,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type='v0' if d % 2 == 0 and forward_type == ['v0', 'v1'] else ('v1' if forward_type == ['v0', 'v1'] else forward_type),
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
                depth = depth_,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        # x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if self.channel_first else self.pos_embed
            x = x + pos_embed
        ''' Stack modules '''
        x_list = []
        for layer in self.layers:
            x_list.append(rearrange(x, 'b h w c -> b c h w'))
            x = layer(x)

        x = rearrange(x, 'b h w c -> b c h w')
        return x, x_list
