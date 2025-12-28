# models/cross_attention.py (添加GPU支持)
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.device = device

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"

        # 线性变换层
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 移动到设备
        if device:
            self.to(device)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)

        # 确保输入在正确的设备上
        if self.device:
            query = query.to(self.device)
            key = key.to(self.device)
            value = value.to(self.device)

        # 线性变换并分头
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=self.device))
        attention_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # 输出变换
        output = self.out_linear(attended)
        output = self.layer_norm(output + query)  # 残差连接

        return output


class CrossAttentionFeatureExtractor(nn.Module):
    def __init__(self, job_state_dim: int, machine_state_dim: int, hidden_dim: int, num_heads: int, device=None):
        super().__init__()
        self.device = device

        self.job_embedding = nn.Sequential(
            nn.Linear(job_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.machine_embedding = nn.Sequential(
            nn.Linear(machine_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.cross_attention = MultiHeadCrossAttention(hidden_dim, num_heads, device)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 移动到设备
        if device:
            self.to(device)

    def forward(self, job_state: torch.Tensor, machine_state: torch.Tensor) -> torch.Tensor:
        # 确保输入在正确的设备上
        if self.device:
            job_state = job_state.to(self.device)
            machine_state = machine_state.to(self.device)

        # 嵌入层
        job_embedded = self.job_embedding(job_state)
        machine_embedded = self.machine_embedding(machine_state)

        # 交叉注意力
        attended_features = self.cross_attention(job_embedded, machine_embedded, machine_embedded)

        # 前馈网络
        output = self.feed_forward(attended_features)

        return output