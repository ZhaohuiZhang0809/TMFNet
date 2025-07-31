import torch
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


class Text_Encoder(nn.Module):
    def __init__(
            self,
            version=r'/mnt/d/PyCode/zzh/LMFNet/models/clip-vit-large-patch14',
            max_length=77,
            freeze=False,
            output_dim=384,
            load_pretrained=True,  # 新增：是否加载预训练权重
    ): # 或者./clip-vit-large-patch14'
        super(Text_Encoder, self).__init__()

        # 加载 tokenizer（仍然需要预训练的词表）
        self.tokenizer = CLIPTokenizer.from_pretrained(version, clean_up_tokenization_spaces=False)

        # 加载模型配置
        config = CLIPTextConfig.from_pretrained(version)
        config.max_position_embeddings = max_length  # 确保 max_length 与配置一致

        if load_pretrained:
            # 加载预训练权重
            self.transformer = CLIPTextModel.from_pretrained(version, config=config)
        else:
            # 随机初始化模型（不加载预训练权重）
            self.transformer = CLIPTextModel(config)

        # 投影层
        self.proj = nn.Linear(768, output_dim)

        self.max_length = max_length

        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        tokens = batch_encoding['input_ids'].to(device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=False)
        out = outputs.last_hidden_state
        out = self.proj(out)
        return out


if __name__ == '__main__':
    # 初始化 Text_Encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = Text_Encoder(max_length=77).to(device)

    # 测试输入文本
    text = ("This is a sample text for CLIP tokenizer.", "Another example sentence.")

    out = text_encoder(text)

    print(out.size())

    # 使用 torchsummary 查看模型结构
    # 注意：torchsummary 主要用于分析模型的输入输出，而不是内部结构
    # summary(text_encoder, input_size=(len(text),), device=text_encoder.device)

    import time
    import numpy as np

    def calculate_fps(num_iterations=100):
        t_all = []

        # 运行推理多次
        with torch.no_grad():
            for _ in range(num_iterations):
                # 启动计时器
                start_time = time.time()
                out = text_encoder(text)
                # 计算总时间
                total_time = time.time() - start_time
                t_all.append(total_time)

        print('average time:', np.mean(t_all) / 1)
        print('average fps:', 1 / np.mean(t_all))

        print('fastest time:', min(t_all) / 1)
        print('fastest fps:', 1 / min(t_all))

        print('slowest time:', max(t_all) / 1)
        print('slowest fps:', 1 / max(t_all))


    calculate_fps(num_iterations=10)
