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
    def __init__(self, in_dim, n_layer = 1, d_state = 16, expand = 2):
        super(FCGAM, self).__init__()
        self.CrossAttention1 = MultiCrossAttention(embedding_dim=in_dim, num_heads=3)
        self.CrossAttention2 = MultiCrossAttention(embedding_dim=in_dim, num_heads=3)
        # self.Mlp1 = Mlp(in_features=in_dim)
        # self.Mlp2 = Mlp(in_features=in_dim)

        self.args = ModelArgs(
            d_model=in_dim,  # 隐藏层维度
            n_layer=n_layer,  # 层数
            d_state=d_state,  # 状态空间维度
            expand=expand,  # 扩展因子
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
            x: [B, L, C]
            k: Top-K数量
            mode: 聚合方式 ('max', 'mean', 'norm')
            mask_value: 替换值（如0或[MASK]的嵌入）
        """
        B, L, C = x.shape
        k = L // 16

        # 1. 计算激活值并获取topk索引
        if mode == 'max':
            activation = x.max(dim=2).values  # [B, L]
        elif mode == 'mean':
            activation = x.mean(dim=2)       # [B, L]
        elif mode == 'norm':
            activation = x.norm(dim=2)       # [B, L]

        _, topk_indexs = torch.topk(activation, k=k, dim=1)  # [B, k]

        # 2. 生成布尔掩码
        bool_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        for b in range(B):
            bool_mask[b, topk_indexs[b]] = True

        # 3. 保存原始值
        maske_values = torch.zeros(B, k, C, device=x.device)
        unmaske_values = torch.zeros(B, L - k, C, device=x.device)
        for b in range(B):
            maske_values[b] = x[b, bool_mask[b]]
            unmaske_values[b] = x[b, ~bool_mask[b]]

        # # 4. 应用掩码
        # masked_x = x.clone()
        # masked_x[bool_mask] = mask_value

        return maske_values, unmaske_values, topk_indexs

    def compute_loss(self):
        # spa_rec = self.spa_rec  # [B, L, C]
        text2img = self.text2img  # [B, L, C]
        img2text = self.img2text
        loss_fn = InfoNCE(temperature=0.05)

        # 计算重建损失
        # recon_loss = F.mse_loss(spa_rec.softmax(dim=-1).float(), text2img.softmax(dim=-1).float())
        contrast_loss = loss_fn(img2text.mean(dim=1, keepdim=True).squeeze(1), text2img.mean(dim=1, keepdim=True).squeeze(1))

        # return 0.1 * recon_loss + contrast_loss
        return contrast_loss


    def recons_tokens(self, x, x_vis, indexs):
        steps = x.size(1) - x_vis.size(1)
        # 预计算静态参数
        A = -torch.exp(self.A_log.float())  # (d_in, n)
        D = self.D.float()
        d_in, n = A.shape
        B, L, C = x_vis.shape

        # 初始状态获取
        logits, hidden_state = self.RecMamba(x_vis)
        next_token = logits[:, -1:, :]
        x_vis = torch.cat([x_vis, next_token], dim=1)

        # 优化后的自回归循环
        for _ in range(steps - 1):
            # 并行计算所有投影
            x_dbl = self.x_proj(x_vis)
            x_proj = self.in_proj(x_vis)

            # 优化split操作
            delta, B_mat, C = torch.split(
                x_dbl,
                [self.args.dt_rank, n, n],
                dim=-1
            )
            delta = F.softplus(self.dt_proj(delta))

            # 兼容旧版本的einsum计算
            deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
            deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B_mat, x_proj[..., :d_in])

            # 更新状态
            hidden_state = deltaA[:, -1] * hidden_state + deltaB_u[:, -1]

            # 预测下一个token
            next_token = torch.einsum('bdn,bn->bd', hidden_state, C[:, -1, :])
            next_token = self.out_proj(next_token).unsqueeze(1)
            x_vis = torch.cat([x_vis, next_token], dim=1)

        full_seq = x
        # 5. 按原始indexs重新排序（如果需要）
        if indexs is not None:
            for i in range(B):
                # 实现按indexs将生成的token插入到原掩码位置
                full_seq[i][indexs] = x_vis[i][:steps, :]

        return full_seq

    def forward(self, img_x, text_x):
        B, C, H, W = img_x.shape
        # 跨模态融合
        img_x = rearrange(img_x, "b c h w -> b (h w) c")  # B, L, C
        # 添加
        # img2text = self.CrossAttention1(text_x, img_x)
        text2img = self.CrossAttention2(img_x, text_x)

        # # 动态掩码
        # with torch.no_grad():
        #     img_invis, img_vis, img_indexs = self.token_masking(text2img)
        #     # text_invis, text_vis, text_indexs = self.token_masking(img2text)

        # todo: 移动窗口
        # ## 是否连同 text + img 序列共同推理
        # spa_rec = self.recons_tokens(text2img, img_vis, img_indexs)

        # # 保存需要计算损失的张量
        # self.spa_rec = spa_rec
        # self.text2img = text2img
        # # self.img2text = img2text
        text2img = rearrange(text2img, " b (h w) c -> b c h w", h=H, w=W)

        # loss = self.compute_loss()

        return text2img
