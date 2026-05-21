# models/RegLSTM.py
import torch
import torch.nn as nn
from typing import Tuple, Optional

class RegLSTM(nn.Module):
    """强正则化的轻量 LSTM，用于人体运动重定向。

    相较 SimpleLSTM 的改进:
    - hidden_dim: 256 → 128 (减少容量)
    - num_layers: 2 → 1 (防止过拟合)
    - dropout: 0.1 → 0.4 (强正则化)
    - 输出部分增加 LayerNorm + 全连接头，增强稳定性
    - 移除过于复杂的初始化，保留 Xavier 初始化

    输入/输出接口与 SimpleLSTM 完全一致:
    - input:  (batch, seq_len, 69)
    - output: (batch, seq_len, 36)
    """
    
    def __init__(
        self,
        input_dim: int = 69,
        hidden_dim: int = 128,
        num_layers: int = 1,
        output_dim: int = 36,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 核心 LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 输出映射头（LayerNorm + Dropout + Linear）
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """合理的参数初始化，避免梯度爆炸/消失"""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        # 全连接层使用默认初始化即可（PyTorch 默认 Kaiming Uniform）

    def name(self) -> str:
        return "RegLSTM"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 69)
        Returns:
            output: (batch, seq_len, 36)
        """
        # LSTM 前向
        lstm_out, _ = self.lstm(x)            # (B, T, hidden_dim)
        # LayerNorm + Dropout 逐时间步应用
        B, T, H = lstm_out.shape
        flat = lstm_out.reshape(B * T, H)
        flat = self.layer_norm(flat)
        flat = self.dropout(flat)
        # 输出映射
        out = self.output_proj(flat)           # (B*T, output_dim)
        out = out.reshape(B, T, self.output_dim)
        return out