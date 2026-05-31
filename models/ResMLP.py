import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """残差块：线性 -> LayerNorm -> GELU -> Dropout -> 线性 -> LayerNorm -> GELU -> Dropout"""
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class ResMLP(nn.Module):
    """改进版 MLP：引入残差连接和 LayerNorm，更有利于旋转部分的精确学习。

    Args:
        hidden_dim (int): 隐藏层宽度（默认 256）
        num_blocks (int): 残差块数量（默认 2）
        dropout (float): Dropout 概率（默认 0.5）
    """
    def __init__(self, input_dim=69, output_dim=36, hidden_dim=256, num_blocks=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def name(self):
        return "ResMLP"

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, 69)
        Returns:
            torch.Tensor: (batch_size, 36)
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)