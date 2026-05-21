# TemporalModel.py (修复安全约束问题)
import torch
import torch.nn as nn
from typing import Tuple, Optional

class SimpleLSTM(nn.Module):
    """Temporal motion retargeting model: Uses LSTM to learn temporal dependencies in human motion data for generating smooth robot control commands.
    
    Key improvements:
    - Input: Continuous human pose sequence (batch, seq_len, 69)
    - Output: Synchronized robot target sequence (batch, seq_len, 36)
    - Temporal modeling: Captures motion continuity, eliminating jitter issues from frame-wise MLP approaches
    - Physics-aware: LayerNorm stabilizes rotation-sensitive components
    
    Fully compatible interface with SimpleMLP for seamless integration into existing training/inference pipelines.
    """
    
    def __init__(
        self,
        input_dim: int = 69,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 36,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # Core LSTM network
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection layer (maps LSTM features to robot pose space)
        # CRITICAL FIX 1: 禁用 LayerNorm 的可学习参数
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False)  # ← 必须设置为 False
        )
        
        # Advanced initialization (2026 best practices)
        self._init_weights()
    
    def _init_weights(self):
        """Proper LSTM weight initialization to prevent gradient explosion/vanishing"""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        
        #CRITICAL FIX 2: 强制初始化输出层 bias 为安全姿态
        with torch.no_grad():
            # UR5 安全姿态中值 (6关节 × 6自由度 = 36维)
            # 注意：根据实际机器人调整数值 (此处为UR5示例)
            safe_bias = torch.tensor([
                0.0, 0.0, -1.57, 0.0, 1.57, 0.0,  # 关节1
                0.0, 0.0, -1.57, 0.0, 1.57, 0.0,  # 关节2
                0.0, 0.0, -1.57, 0.0, 1.57, 0.0,  # 关节3
                0.0, 0.0, -1.57, 0.0, 1.57, 0.0,  # 关节4
                0.0, 0.0, -1.57, 0.0, 1.57, 0.0,  # 关节5
                0.0, 0.0, -1.57, 0.0, 1.57, 0.0,  # 关节6
            ])
            # 验证维度匹配
            assert safe_bias.shape[0] == self.output_dim, \
                f"safe_bias dim {safe_bias.shape[0]} != output_dim {self.output_dim}"
            self.output_proj[0].bias.copy_(safe_bias)  # ← 锚定安全输出
    
    
    def name(self) -> str:
        return f"SimpleLSTM_h{self.hidden_dim}_l{self.num_layers}{'_bi' if self.bidirectional else ''}"
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        lstm_out, _ = self.lstm(x, hidden)
        output = self.output_proj(lstm_out)
        return output
    
    def predict_next(
        self,
        x: torch.Tensor,
        n_steps: int = 1
    ) -> torch.Tensor:
        """Generates predictions for n_steps into the future (for robot motion planning)
        
        Args:
            x: Current history sequence (batch_size, seq_len, 69)
            n_steps: Number of future steps to predict
        
        Returns:
            Predicted sequence (batch_size, n_steps, 36)
        """
        device = x.device
        batch_size = x.size(0)
        predictions = []
        
        # Initialize with last input frame
        current_input = x[:, -1:, :]  # (B, 1, 69)
        
        # Recursive prediction
        for _ in range(n_steps):
            out = self.forward(current_input)  # (B, 1, 36)
            predictions.append(out)
            # Use prediction as next input (in real deployment, should incorporate feedback control)
            current_input = torch.cat([
                current_input[:, :, 3:],  # Discard old position data
                out                       # Append new pose prediction
            ], dim=-1)
        
        return torch.cat(predictions, dim=1)  # (B, n_steps, 36)