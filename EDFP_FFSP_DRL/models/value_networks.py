# models/value_networks.py (添加GPU支持)
import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, device=None):
        super().__init__()
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        # 移动到设备
        if device:
            self.to(device)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        # 确保输入在正确的设备上
        if self.device:
            state_features = state_features.to(self.device)

        # 全局平均池化获取整体特征
        if len(state_features.shape) > 2:
            state_features = state_features.mean(dim=1)

        return self.network(state_features)