# training/multihead_trainer.py
from EDFP_FFSP_DRL.training.ippo_trainer import IPPOTrainer
import torch
import torch.nn as nn
import numpy as np


class MultiHeadTrainer(IPPOTrainer):
    """多头注意力训练器 - 使用注意力机制"""

    def __init__(self, config, job_agent, machine_agent, env):
        super().__init__(config, job_agent, machine_agent, env)

        # 添加注意力机制
        self.use_attention = True
        self.attention_dim = 64
        self.num_heads = 4

        # 初始化注意力层
        self._init_attention_layers()

        print("✅ Multi-Head Attention PPO Trainer Initialized")
        print(f"🔍 Attention: {self.num_heads} heads, dim={self.attention_dim}")

    def _init_attention_layers(self):
        """初始化注意力层"""
        # 作业注意力层
        self.job_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 机器注意力层
        self.machine_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 移动到设备
        device = torch.device(self.config.DEVICE)
        self.job_attention.to(device)
        self.machine_attention.to(device)
        self.cross_attention.to(device)

    def _process_state_with_attention(self, state_batch):
        """使用注意力机制处理状态"""
        if not self.use_attention:
            return state_batch

        try:
            batch_size = state_batch.size(0)
            state_dim = state_batch.size(1)

            # 重塑为序列形式（假设状态可以分成作业和机器部分）
            seq_length = self.config.NUM_JOBS + self.config.NUM_MACHINES

            # 创建位置编码
            pos_encoding = self._create_positional_encoding(seq_length, self.attention_dim)
            pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1).to(state_batch.device)

            # 将状态转换为序列
            if state_dim >= seq_length * self.attention_dim:
                # 重塑为序列
                seq_state = state_batch.view(batch_size, seq_length, self.attention_dim)
                seq_state = seq_state + pos_encoding

                # 自注意力处理
                attended_state, _ = self.job_attention(seq_state, seq_state, seq_state)

                # 全局池化
                global_features = attended_state.mean(dim=1)

                return global_features
            else:
                # 状态维度不够，跳过注意力
                return state_batch

        except Exception as e:
            print(f"⚠️ Attention processing error: {e}")
            return state_batch

    def _create_positional_encoding(self, seq_length, d_model):
        """创建位置编码"""
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pos_encoding = torch.zeros(seq_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def collect_rollout(self):
        """收集经验 - 带注意力处理"""
        batch, episode_reward, steps, _ = super().collect_rollout()

        # 对状态应用注意力
        if batch is not None and 'states' in batch:
            batch['states'] = self._process_state_with_attention(batch['states'])

        return batch, episode_reward, steps, {}