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
#     """ 频域卷积模块 """
#     def __init__(self, in_channels, out_channels):
#         super(FreqConvModule, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         # 1x1卷积层
#         self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         # 将输入张量转换到频域
#         x_fft = torch.fft.fft2(x, dim=(2, 3))  # 对输入张量进行二维傅里叶变换
#         x_fft = self.conv1(x_fft.real)
#         x_fft = self.relu(x_fft.real)
#         x_fft = self.conv2(x_fft.real)
#         x_ifft = torch.fft.ifft2(x_fft, dim=(2, 3)).real
#
#         return x_ifft

# # 定义全局卷积模块
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


# # 稀疏通道注意力
# class SparseChannelAttention(nn.Module):
#     def __init__(self, channel, reduction_ratio=16, k_ratio=0.5):
#         """
#         Args:
#             channel (int): 输入通道数
#             reduction_ratio (int): 中间层压缩比例（默认16）
#             k_ratio (float): 稀疏比例，选择Top-K通道的比例（0 < k_ratio <= 1）
#         """
#         super(SparseChannelAttention, self).__init__()
#         self.channel = channel
#         self.k = max(1, int(channel * k_ratio))  # 至少保留1个通道
#         self.reduction_ratio = reduction_ratio
#
#         # 全局平均池化
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#
#         # 中间层（全连接 + ReLU）
#         self.mlp = nn.Sequential(
#             nn.Linear(channel, channel // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction_ratio, channel)  # 输出通道重要性得分
#         )
#
#         # 全局可学习参数（用于增强未被选中的通道）
#         self.global_weight = nn.Parameter(torch.ones(1, channel, 1, 1))
#         self.global_bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # 1. 计算通道重要性得分
#         scores = self.gap(x).view(b, c)  # [B, C]
#         scores = self.mlp(scores)        # [B, C]
#
#         # 2. 生成稀疏掩码（Top-K选择）
#         _, topk_indices = torch.topk(scores, self.k, dim=1)  # [B, K]
#         sparse_mask = torch.zeros_like(scores)               # [B, C]
#         sparse_mask.scatter_(1, topk_indices, 1.0)           # 仅Top-K位置为1
#
#         # 3. 计算注意力权重（仅对Top-K通道）
#         attention = torch.sigmoid(scores) * sparse_mask      # [B, C]
#
#         # 4. 应用注意力权重 + 全局参数
#         attention = attention.view(b, c, 1, 1)               # [B, C, 1, 1]
#         out = x * ((attention * self.global_bias + 1) * self.global_weight)
#         # out = x * attention
#
#         return out



# support: v0, v0seq
class SS2Dv0:
    def __initv0__(
            self,
            # basic dims ===========
            d_model=96,
            topk=4,
            mlp_ratio=4.0,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            windows_size=3,
            depth=1,
            **kwargs,
    ):
        r""" V-Mamba-v0 框架
        Arg:
            d_model: 模型的输出维度（默认为96）。
            d_state: 状态维度（默认为16）。
            ssm_ratio: 状态维度与模型维度的比率（默认为2.0）。
            dt_rank: 动态时间参数的维度，默认为“auto”，会根据 d_model 计算
        """

        # 空间维度优先
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
        k_group = 2     # 扫描方向
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

        # self.selective_scan = selective_scan_fn  # 选择性扫描（加速）

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        # self.in_proj = nn.Conv2d(d_model, d_inner * 2, kernel_size=3, padding=1)


        self.act: nn.Module = act_layer()
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

        self.average_pool = nn.AdaptiveAvgPool1d(1)

        # batch, length, force_fp32, seq, k_group, inner, rank
        self.mambaScanner = MambaScanner(seq=seq, force_fp32=force_fp32, init_dt_A_D=init_dt_A_D, x_proj_weight=self.x_proj_weight)

        # # Mutil-conv
        # # self.GConv = GlobalConvModule(in_dim=d_model, out_dim=d_model, kernel_size=(7, 7))
        # self.LocalConv = DWConv(d_model, d_model)
        # self.FreqConv = FreqConvModule(in_channels=d_model, out_channels=d_model)
        #
        # # Sparse Channel Attention
        # self.SCA = SparseChannelAttention(d_model)

        # out proj =======================================
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
        ])  # 会为生成均等划分的膨胀窗口, (B, C, H, W) -> (B, C*window_size*window_size, n)

        # # channel transfer
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
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

        """ 根据激活值划分 """
        active_feat, passive_feat, only_active_feat, only_passive_feat = self.DynamicMasking(x) # (B, C, H, W), ~ ; (B, C, nW*window_size*window_size), ~

        """ local遍历路径 """
        '''前景SSM -> 前景类别特征'''
        # partition
        xs_fore = only_active_feat.unsqueeze(dim=1)
        # 拼接 x_s 和 其翻转
        xs_fore = torch.cat([xs_fore, torch.flip(xs_fore, dims=[-1])], dim=1)  # # (B, 2, C, nW*window_size*window_size)
        # 选择性扫描
        out = self.mambaScanner(xs_fore)
        # """ 两种遍历路径叠加 (Mamba之后) """
        # token位置还原
        inv_1 = torch.flip(out[:, 1:2], dims=[-1])
        # 两种状态叠加, 添加 投影权重
        y1 = inv_1[:, 0] + out[:, 0]   # (B, C, nW*window_size*window_size)
        # foreground_class = self.average_pool(y1) # (B, C, 1)

        # 还原形状，方便输出
        y1 = y1.transpose(dim0=1, dim1=2).contiguous()

        '''前景多尺度卷积'''
        # Local_feature = self.LocalConv(active_feat)
        # Freq_feature = self.FreqConv(active_feat)

        # # 多尺度前景聚合注意力
        SSM_Q, SSM_K = rearrange(self.kv(y1),"b (n2 ws) (h_n h_dim) -> h_n b h_dim ws n2", ws=self.windows_size ** 2, h_n=self.num_dilation).unsqueeze(-1).chunk(
                                    2, dim=2)  # (B, N2*window_size*window_size, C) -> (h_n, B, h_dim, window_size*window_size, N2, 1)

        x = x.reshape(B, self.num_dilation, D // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)

        y_list = []
        for i in range(self.num_dilation):
            Win_V = self.unfold[i](x[i])
            Win_V = rearrange(Win_V, 'b (h_dim ws) n1 -> b h_dim n1 ws', h_dim=D // self.num_dilation)  # (B, h_dim, N1, window_size*window_size); N1 = L

            A_M = SSM_Q[i].transpose(-1, -2) @ SSM_K[i] / math.sqrt(D // self.num_dilation)  # (B, h_dim, window_size*window_size, 1)
            A_M = F.softmax(A_M, dim=-1).squeeze(-1)
            y = Win_V @ A_M     # (B, h_dim, N1)
            y_list.append(y.squeeze(-1))

        y = torch.cat(y_list, dim=1)
        y = y.transpose(-1, -2)

        # y = y * self.fc(foreground_class.squeeze(-1)).view(B, D, 1, 1)
        # fore_sim = self.batch_cosine_similarity(foreground_class, y.view(B, D, H * W)).view(B, -1, H, W)

        # foreground = active_feat * torch.sigmoid(Local_feature + Freq_feature) * fore_sim
        # y = y * fore_sim

        # '''背景稀疏通道注意力'''
        # background_att = self.SCA(passive_feat)
        # back_sim = self.batch_cosine_similarity(background_class, background_att.view(B, D, H * W)).view(B, -1, H, W)
        # background = passive_feat * back_sim

        # # y = torch.cat((foreground, background), dim=1)
        # y = foreground + background

        '''背景SSM -> 背景类别特征'''
        # # partition
        # xs_back = only_passive_feat.unsqueeze(dim=1)
        # # 拼接 x_s 和 其翻转
        # xs_back = torch.cat([xs_back, torch.flip(xs_back, dims=[-1])], dim=1)  # # (B, 2, C, nW*window_size*window_size)
        # # 选择性扫描
        # out_2 = self.mambaScanner(xs_back)
        # # """ 两种遍历路径叠加 (Mamba之后) """
        # # token位置还原
        # inv_2 = torch.flip(out_2[:, 1:2], dims=[-1])
        # # 两种状态叠加, 添加 投影权重
        # y2 = inv_2[:, 0] + out_2[:, 0]      # (B, C, nW*window_size*window_size)
        # background_class = self.average_pool(y2) # (B, C, 1)

        # 正则输出
        y = self.out_norm(y).view(B, H, W, -1)
        # z是一个门控（SiLU激活分支）
        # y = torch.cat([y, z], dim=1)
        y = y + z
        out = self.dropout(self.out_proj(y))

        return out


class SS2D(nn.Module, SS2Dv0):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            windows_size=2,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            mlp_ratio=4.0,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            depth=1,
            # ======================
            **kwargs,
    ):
        r""" 初始化 SS2D
        Arg:
            d_model, d_state: 模型的维度和状态维度，影响特征表示的大小
            ssm_ratio: 状态空间模型的比例，可能影响模型的复杂度
            dt_rank: 时间步长的秩，控制时间序列的处理方式
            act_layer: 激活函数，默认为 SiLU（Sigmoid Linear Unit）
            d_conv: 卷积层的维度，值小于 2 时表示不使用卷积
            conv_bias: 是否使用卷积偏置项
            dropout: dropout 概率，用于防止过拟合
            bias: 是否在模型中使用偏置项
            dt_min, dt_max: 时间步长的最小和最大值
            dt_init: 时间步长的初始化方式，可以为 "random" 等
            dt_scale, dt_init_floor: 影响时间步长的缩放和下限
            initialize: 指定初始化方法
            forward_type: 决定前向传播的实现方式，支持多种类型
            channel_first: 指示是否使用通道优先的格式
            **kwargs: 允许传递额外参数，方便扩展
        """
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, windows_size=windows_size, mlp_ratio=mlp_ratio, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, depth=depth
        )

        # 调用不同的初始化函数
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        # else:
        #     self.__initv1__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            windows_size: int = 4,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
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
        r""" VMamba整体架构
        Arg:
            维度变换参数:
                hidden_dim: 输入和输出的特征维度
                drop_path: 用于随机丢弃路径的概率，防止过拟合
                norm_layer: 归一化层，默认为 LayerNorm
                channel_first: 数据格式，指示是否采用通道优先
            SSM相关参数:
                ssm_d_state: 状态空间模型的状态维度
                ssm_ratio: 决定是否使用 SSM 的比例
                ssm_dt_rank: 时间步长的秩
                ssm_act_layer: SSM 的激活函数，默认为 SiLU
                ssm_conv: 卷积层的大小
                ssm_conv_bias: 是否使用卷积偏置
                ssm_drop_rate: SSM 中的 dropout 概率
                ssm_init: SSM 的初始化方式
                forward_type: 决定前向传播的实现方式
            MLP相关参数:
                mlp_ratio: MLP 隐藏层与输入层的维度比率
                mlp_act_layer: MLP 的激活函数，默认为 GELU
                mlp_drop_rate: MLP 中的 dropout 概率
                gmlp: 是否使用 GMLP 结构
            其他参数:
                use_checkpoint: 是否使用梯度检查点以节省内存
                post_norm: 是否在添加残差连接后进行归一化
                _SS2D: 状态空间模型的类类型
        """

        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        ''' SSM模块, 初始化设置为 V0 版本 '''
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS2D(
                d_model=hidden_dim,
                windows_size=windows_size,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                mlp_ratio=mlp_ratio,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                depth = depth
            )

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


# 主函数
class SCA_SSM(nn.Module):
    """Spatial Context-Aware State Space Module """
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

        ##
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)  # 根据patchembed版本选择嵌入化函数
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
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
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,                  # 有三个版本的下采样
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
                depth = depth[i_layer],
            ))

        # self.classifier = nn.Sequential(OrderedDict(
        #     norm=norm_layer(self.num_features),  # B,H,W,C
        #     permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
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
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                          channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
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
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
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
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ===========================
            _SS2D=SS2D,
            depth_ = 1,
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
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        ''' 堆叠模块 '''
        x_list = []
        for layer in self.layers:
            x_list.append(rearrange(x, 'b h w c ->  b c h w'))
            x = layer(x)

        x = rearrange(x, 'b h w c ->  b c h w')
        return x, x_list



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = SCA_SSM(
        depths=[2, 2, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=1, num_classes=1, windows_size=3,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="v0",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v2",
        use_checkpoint=False, posembed=False,
    ) # 窗口大小必须为 20 的因数，因为最底层特征大小为 20*20


    net.cuda().train()

    summary(net, (2, 96, 80, 80))

    inputs = torch.randn(2, 96, 80, 80).cuda()
    flops, params = profile(net, (inputs,))
    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


    import time
    import numpy as np

    def calculate_fps(model, input_size, batch_size=1, num_iterations=100):
        t_all = []
        # 模型设置为评估模式
        model.eval()
        # 模拟输入数据
        input_data = torch.randn(batch_size, *input_size).to(device)  # 如果有GPU的话

        # 运行推理多次
        with torch.no_grad():
            for _ in range(num_iterations):
                # 启动计时器
                start_time = time.time()
                output = model(input_data)
                # 计算总时间
                total_time = time.time() - start_time
                t_all.append(total_time)

        print('average time:', np.mean(t_all) / 1)
        print('average fps:', 1 / np.mean(t_all))

        print('fastest time:', min(t_all) / 1)
        print('fastest fps:', 1 / min(t_all))

        print('slowest time:', max(t_all) / 1)
        print('slowest fps:', 1 / max(t_all))


    calculate_fps(net, input_size=(96, 80, 80), batch_size=2, num_iterations=10)
