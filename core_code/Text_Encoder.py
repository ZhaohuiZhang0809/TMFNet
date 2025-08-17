import torch
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


class Text_Encoder(nn.Module):
    def __init__(
            self,
            version=r'/mnt/PyCode/zzh/LMFNet/models/clip-vit-large-patch14',  # Path to the pre-trained model
            max_length=777,  # Maximum length of input text
            freeze=False,  # Whether to freeze the model weights
            output_dim=384,  # Dimension of the output features
            load_pretrained=True,  # Whether to load pre-trained weights
    ):  # Initialize the Text_Encoder module
        super(Text_Encoder, self).__init__()

        # Load the tokenizer (still requires a pre-trained vocabulary)
        self.tokenizer = CLIPTokenizer.from_pretrained(version, clean_up_tokenization=False)

        # Load the model configuration
        config = CLIPTextConfig.from_pretrained(version)
        config.max_position_embeddings = max_length  # Ensure max_length is consistent with the configuration

        if load_pretrained:
            # Load pre-trained weights
            self.transformer = CLIPTextModel.from_pretrained(version, config=config)
        else:
            # Randomly initialize the model (do not load pre-trained weights)
            self.transformer = CLIPTextModel(config)

        # Projection layer
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
