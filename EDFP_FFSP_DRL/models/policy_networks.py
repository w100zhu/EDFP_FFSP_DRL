# models/policy_networks.py (添加GPU支持)
import torch
import torch.nn as nn
import torch.nn.functional as F


class JobSelectionPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int, device=None):
        super().__init__()
        self.device = device

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # 移动到设备
        if device:
            self.to(device)

    def forward(self, state_features: torch.Tensor):
        # 确保输入在正确的设备上
        if self.device:
            state_features = state_features.to(self.device)

        # 特征提取
        features = self.feature_net(state_features)

        # 全局平均池化获取整体特征
        if len(features.shape) > 2:
            features = features.mean(dim=1)

        # 策略头
        logits = self.policy_head(features)

        return torch.distributions.Categorical(logits=logits)


class MachineAllocationPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int, device=None):
        super().__init__()
        self.device = device

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # 移动到设备
        if device:
            self.to(device)

    def forward(self, state_features: torch.Tensor):
        # 确保输入在正确的设备上
        if self.device:
            state_features = state_features.to(self.device)

        # 特征提取
        features = self.feature_net(state_features)

        # 全局平均池化获取整体特征
        if len(features.shape) > 2:
            features = features.mean(dim=1)

        # 策略头
        logits = self.policy_head(features)

        return torch.distributions.Categorical(logits=logits)