import torch
from torch import nn
import torch.nn.functional as F
import torch_dct as dct
from models.common import Dwt2d, Iwt2d, DoubleConv, DWConv, MultiCrossAttention, DSConv


# 自适应卷积核：通过学习输入特征图的动态调整卷积核的权重，使得模型能够更好地适应不同的输入数据。
# 频率响应：通过使用离散余弦变换（DCT）和逆离散余弦变换（IDCT），模块可以在频率域中调整卷积核的响应，从而增强模型对特定频率特征的捕捉能力。
# 增强特征表示：通过自适应调整卷积核，模型可以更有效地提取和表示输入特征图中的重要信息，从而可能提高下游任务的性能。
class OmniAttention(nn.Module):
    """
    For adaptive kernel, AdaKern
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AFRM(nn.Module):
    def __init__(self, in_dim, out_dim, use_dct=True, groups=1, kernel_size=3):
        super(AFRM, self).__init__()

        # 1. 多尺度特征提取分支
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        # 多尺度卷积核
        self.conv_s1 = nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1, groups=groups)
        self.conv_s2 = nn.Conv2d(in_dim, in_dim // 2, kernel_size=5, padding=2, groups=groups)

        # 通道调整
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=1),
            # nn.BatchNorm2d(in_dim // 2),
            # nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=1),
            # nn.BatchNorm2d(in_dim // 2),
            # nn.ReLU(inplace=True),
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_dim, out_dim, kernel_size=1),
        #     nn.BatchNorm2d(out_dim),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv3 = DWConv(in_dim, out_dim)
        self.DSConv = DSConv(in_dim, out_dim, kernel_size=7, extend_scope=1, morph=0,
                 if_offset=True, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        # 2. 频率处理分支
        # 缩放因子 (Scaled f)
        self.scale_factors = [1.0, 2.0, 3.0, 4.0]  # 对应 V1-V4

        self.dwt = Dwt2d()
        self.iwt = Iwt2d()
        # 定义可学习权重 (c_out, c_in, k, k)
        self.weight = nn.Parameter(torch.Tensor(in_dim, in_dim// groups, kernel_size, kernel_size))
        self.ratio = nn.Parameter(torch.tensor(0.5))

        self.OMNI_ATT1 = OmniAttention(in_planes=in_dim, out_planes=in_dim, kernel_size=kernel_size,
                                       groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
        self.OMNI_ATT2 = OmniAttention(in_planes=in_dim, out_planes=in_dim,
                                       kernel_size=kernel_size if use_dct else 1, groups=1,
                                       reduction=0.0625, kernel_num=1, min_channel=16)
        self.in_channels = in_dim
        self.out_channels = in_dim
        self.use_dct = use_dct
        self.groups = groups
        self.kernel_size = kernel_size

        #
        self.proj = Mlp(in_features=out_dim)

    def adaptive_freq_conv(self, x):
        b, c, h, w = x.shape

        c_att1, f_att1, _, _, = self.OMNI_ATT1(x)  # 获取通道和滤波器注意力
        c_att2, f_att2, spatial_att2, _, = self.OMNI_ATT2(x)  # 获取空间注意力

        adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # b, c_out, c_in, k, k
        adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
        adaptive_weight_res = adaptive_weight - adaptive_weight_mean
        _, c_out, c_in, k, k = adaptive_weight.shape

        if self.use_dct:
            dct_coefficients = dct.dct_2d(adaptive_weight_res)
            # print(adaptive_weight_res.shape, dct_coefficients.shape)
            spatial_att2 = spatial_att2.reshape(b, 1, 1, k, k)
            dct_coefficients = dct_coefficients * (spatial_att2 * 2)
            # print(dct_coefficients.shape)
            adaptive_weight_res = dct.idct_2d(dct_coefficients)
            # adaptive_weight_res = adaptive_weight_res.reshape(b, c_out, c_in, k, k)
            # print(adaptive_weight_res.shape, dct_coefficients.shape)

        adaptive_weight = self.ratio * (adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (
                f_att1.unsqueeze(2) * 2)) + (1 - self.ratio) * (adaptive_weight_res * (c_att2.unsqueeze(1) * 2) * (
                                  f_att2.unsqueeze(2) * 2))
        adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)

        return adaptive_weight

    # c -> 2c
    def forward(self, x):
        """ 自适应频率权重的卷积 """
        b, c_in, h, w = x.shape

        # # 多尺度
        # feat_h = self.pool_h(x)
        # feat_w = self.pool_w(x)
        #
        # feat_hw = self.conv1(feat_h + feat_w)
        #
        # feat_s1 = self.conv_s1(x)
        # feat_s2 = self.conv_s2(x)
        # feat_s = self.conv2(feat_s1 + feat_s2)
        #
        # x = torch.cat([feat_s, feat_hw], dim=1)
        # x = feat_s * feat_hw

        # DWT分解
        LL, LH, HL, HH = self.dwt(x)  # 各分量形状[b, c_in * 4, h//2, w//2]
        adaptive_weight = self.adaptive_freq_conv(x)  # [b*out_c, in_c, k, k]

        # 动态卷积核调整
        weight = adaptive_weight.view(b * self.out_channels * self.in_channels, 1,
                                      self.kernel_size, self.kernel_size)

        components = [HH]
        proc = []
        for comp in components:
            feat = F.conv2d(
                input=comp.reshape(1, b * self.in_channels, h // 2, w // 2),
                weight=weight,
                padding=self.kernel_size // 2,
                groups=b * self.in_channels
            ).reshape(b, self.out_channels, self.in_channels, h // 2, w // 2).sum(dim=2)
            # feat = self.relu(self.bn(feat))
            proc.append(feat)

        # LL = self.conv(proc[0])
        # LL_comp = self.conv(torch.cat([LL, LH, HL], dim=1))
        # 逆变换重构
        out = self.iwt(torch.cat([LL, LH, HL, proc[0]], dim=1)) + x

        out_skip = self.DSConv(out)
        # out = out_skip.permute(0, 2, 3, 1)
        # out = self.proj(out).permute(0, 3, 1, 2) + out_skip

        return out_skip


# 测试 AFRM
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 参数设置
    in_dim = 24    # 输入通道数
    out_dim = 48  # 输出通道数
    kernel_size = 3
    batch_size = 2
    height, width = 160, 160

    # 初始化 AFRM 模块
    afrm = AFRM(in_dim=in_dim, out_dim=out_dim, use_dct=True, groups=1, kernel_size=kernel_size).to(device)

    # 生成随机输入数据 (batch_size, in_dim, height, width)
    x = torch.randn(batch_size, in_dim, height, width).to(device)

    # 前向传播
    out = afrm(x)  # 输出动态生成的卷积核权重

    # 打印结果形状
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)