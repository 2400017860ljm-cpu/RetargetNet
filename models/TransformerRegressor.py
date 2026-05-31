import torch
import torch.nn as nn

class TransformerRegressor(nn.Module):
    """基于 Transformer 编码器的运动重定向模型。

    Args:
        input_dim (int): 输入特征维度 (默认 69)
        output_dim (int): 输出特征维度 (默认 36)
        d_model (int): Transformer 隐藏维度 (默认 256)
        nhead (int): 多头注意力头数 (默认 8)
        num_layers (int): 编码器层数 (默认 4)
        dropout (float): Dropout 概率 (默认 0.2)
    """
    def __init__(self, input_dim=69, output_dim=36, d_model=256, nhead=8,
                 num_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (B, 69)
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        x = x + self.pos_embed
        x = self.encoder(x)                  # (B, 1, d_model)
        x = self.layer_norm(x.squeeze(1))    # (B, d_model)
        return self.output_proj(x)           # (B, 36)

    def name(self):
        return "TransformerRegressor"