# agents/machine_agent.py (完整改进版 - 增强探索和多样性正则化)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import deque


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        """
        x: [batch_size, state_dim]
        mask: [batch_size, action_dim], 1=合法, 0=非法
        """
        logits = self.network(x)

        if mask is not None:
            mask = mask.to(logits.device)
            logits = logits.masked_fill(mask == 0, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs + 1e-8


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.network(x)


class MachineAgent:
    def __init__(self, config):
        self.config = config
        self.state_dim = config.STATE_DIM
        self.action_dim = config.MACHINE_ACTION_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.device = config.DEVICE

        # 探索参数
        self.exploration_strategy = getattr(config, 'EXPLORATION_STRATEGY', 'adaptive')
        self.initial_epsilon = getattr(config, 'INITIAL_EPSILON', 0.3)
        self.epsilon = self.initial_epsilon
        self.epsilon_decay = getattr(config, 'EPSILON_DECAY', 0.995)
        self.min_epsilon = getattr(config, 'MIN_EPSILON', 0.05)

        # 温度参数用于softmax探索
        self.initial_temperature = getattr(config, 'INITIAL_TEMPERATURE', 1.0)
        self.temperature = self.initial_temperature
        self.temperature_decay = getattr(config, 'TEMPERATURE_DECAY', 0.998)
        self.min_temperature = getattr(config, 'MIN_TEMPERATURE', 0.1)

        # 自适应探索参数
        self.adaptive_exploration = True
        self.exploration_stats = {"epsilon": 0, "entropy": 0, "softmax": 0, "greedy": 0, "random": 0}
        self.reward_history = deque(maxlen=100)
        self.action_diversity_history = deque(maxlen=50)

        # 多样性正则化权重
        self.diversity_weight = getattr(config, 'DIVERSITY_WEIGHT', 0.01)

        # 学习率调度
        self.learning_rate = config.LEARNING_RATE
        self.lr_decay_steps = getattr(config, 'LR_DECAY_STEPS', 1000)
        self.lr_decay_rate = getattr(config, 'LR_DECAY_RATE', 0.95)

        print(f"🔧 Machine Agent初始化: 状态维度={self.state_dim}, 动作维度={self.action_dim}")
        print(f"  探索策略: {self.exploration_strategy}, ε={self.epsilon:.3f}, τ={self.temperature:.3f}")

        self.actor = ActorNetwork(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate
        )

        self.entropy_coef = config.ENTROPY_COEF
        self.value_coef = config.VALUE_COEF
        self.clip_eps = config.PPO_CLIP_EPS

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,
            gamma=0.9
        )

        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'exploration_rates': [],
            'action_entropies': [],
            'diversity_scores': []
        }

    def _get_action_mask(self, env):
        """生成动作掩码: 1=合法, 0=非法"""
        mask = torch.ones(self.action_dim).to(self.device)

        if env is not None:
            try:
                # 获取可用动作 (jobs, machines)
                _, available_machines = env.get_available_actions()

                mask = torch.zeros(self.action_dim).to(self.device)

                if available_machines:
                    mask[available_machines] = 1.0
                else:
                    mask[0] = 1.0

            except Exception as e:
                print(f"⚠️ Machine Mask生成失败: {e}")
                mask = torch.ones(self.action_dim).to(self.device)

        return mask

    def act(self, state, env=None, training=True, episode=None):
        """增强版动作选择"""
        if not training:
            return self.greedy_act(state, env)

        try:
            state_tensor = self._process_state(state)
            mask = self._get_action_mask(env)

            # 获取可用动作索引
            valid_actions = torch.where(mask == 1)[0].cpu().numpy()

            with torch.no_grad():
                # 计算动作概率
                action_probs = self.actor(state_tensor, mask=mask.unsqueeze(0))

                # 探索策略选择
                explore_type = self._select_exploration_strategy(episode)
                self.exploration_stats[explore_type] += 1

                if explore_type == "random" and len(valid_actions) > 0:
                    # 完全随机探索
                    action_idx = np.random.choice(valid_actions)
                    action = torch.tensor([action_idx], device=self.device)
                    dist = torch.distributions.Categorical(action_probs)
                    log_prob = dist.log_prob(action)

                elif explore_type == "epsilon" and np.random.random() < self.epsilon:
                    # ε-贪心探索
                    if len(valid_actions) > 0:
                        action_idx = np.random.choice(valid_actions)
                        action = torch.tensor([action_idx], device=self.device)
                    else:
                        dist = torch.distributions.Categorical(action_probs)
                        action = dist.sample()
                    dist = torch.distributions.Categorical(action_probs)
                    log_prob = dist.log_prob(action)

                elif explore_type == "softmax":
                    # 温度探索
                    logits = torch.log(action_probs + 1e-8) / self.temperature
                    softmax_probs = F.softmax(logits, dim=-1)

                    # 应用掩码
                    if mask is not None:
                        softmax_probs = softmax_probs * mask.unsqueeze(0)
                        softmax_probs = softmax_probs / (softmax_probs.sum(dim=-1, keepdim=True) + 1e-8)

                    dist = torch.distributions.Categorical(softmax_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                else:
                    # 标准策略梯度探索
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                value = self.critic(state_tensor)

                # 计算动作熵用于多样性统计
                action_entropy = dist.entropy().item()
                self.action_diversity_history.append(action_entropy)

                return action.item(), log_prob, value

        except Exception as e:
            print(f"❌ Machine Agent动作选择错误: {e}")
            # 返回随机有效动作
            if env is not None:
                _, available_machines = env.get_available_actions()
                if available_machines:
                    return np.random.choice(available_machines), torch.tensor([0.0]).to(self.device), torch.tensor(
                        [0.0]).to(self.device)
            return np.random.randint(self.action_dim), torch.tensor([0.0]).to(self.device), torch.tensor([0.0]).to(
                self.device)

    def _select_exploration_strategy(self, episode):
        """选择探索策略"""
        if self.exploration_strategy == "epsilon":
            return "epsilon" if np.random.random() < 0.3 else "policy"
        elif self.exploration_strategy == "adaptive":
            # 自适应探索策略
            if len(self.reward_history) < 20:
                # 早期更多探索
                return np.random.choice(["epsilon", "random", "softmax"], p=[0.4, 0.3, 0.3])
            else:
                # 根据奖励进展调整探索
                recent_rewards = list(self.reward_history)[-20:]
                reward_std = np.std(recent_rewards)

                if reward_std < 0.1:  # 奖励变化小，可能是局部最优，增加探索
                    return np.random.choice(["epsilon", "random", "softmax"], p=[0.5, 0.3, 0.2])
                else:
                    return np.random.choice(["epsilon", "policy", "softmax"], p=[0.2, 0.6, 0.2])
        else:
            # 默认策略
            return np.random.choice(["epsilon", "policy", "softmax"], p=[0.3, 0.5, 0.2])

    def greedy_act(self, state, env=None):
        """贪婪动作 (测试/推理用)"""
        try:
            state_tensor = self._process_state(state)
            mask = self._get_action_mask(env)

            with torch.no_grad():
                # 传入 mask
                action_probs = self.actor(state_tensor, mask=mask.unsqueeze(0))
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    print("⚠️ Machine Agent 动作概率包含NaN或Inf")
                    # 返回随机有效动作
                    if env is not None:
                        _, available_machines = env.get_available_actions()
                        if available_machines:
                            return np.random.choice(available_machines)
                    return np.random.randint(self.action_dim)

                action = torch.argmax(action_probs, dim=-1)

            return action.item()

        except Exception as e:
            print(f"❌ Machine Agent 贪婪动作选择错误: {e}")
            # 返回随机有效动作
            if env is not None:
                _, available_machines = env.get_available_actions()
                if available_machines:
                    return np.random.choice(available_machines)
            return np.random.randint(self.action_dim)

    def update(self, batch):
        if batch is None or len(batch['states']) == 0:
            return 0.0

        try:
            total_policy_loss = 0.0
            total_diversity_loss = 0.0

            # 获取 mask
            masks = batch.get('machine_masks', None)

            for _ in range(self.config.PPO_EPOCHS):
                action_probs = self.actor(batch['states'], mask=masks)
                values = self.critic(batch['states'])

                # 如果 values 是标量，将其转换为 [1] 的形状
                if values.dim() == 0:  # 标量
                    values = values.unsqueeze(0)

                # 展平 values 以匹配 returns 的形状
                values = values.view(-1)
                returns = batch['returns'].view(-1)

                # 现在计算损失，确保形状匹配
                value_loss = F.mse_loss(values, returns)

                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch['machine_actions'])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch['machine_log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch['returns'])

                # 计算多样性损失
                diversity_loss = self._compute_diversity_loss(action_probs)

                total_loss = (policy_loss +
                              self.value_coef * value_loss -
                              self.entropy_coef * entropy +
                              self.diversity_weight * diversity_loss)

                self.optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.GRAD_CLIP
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_diversity_loss += diversity_loss.item()

            # 记录训练统计
            avg_policy_loss = total_policy_loss / self.config.PPO_EPOCHS
            avg_diversity_loss = total_diversity_loss / self.config.PPO_EPOCHS

            return avg_policy_loss

        except Exception as e:
            print(f"❌ Machine Agent update error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _compute_diversity_loss(self, action_probs):
        """计算多样性损失 - 鼓励探索不同动作"""
        batch_size = action_probs.shape[0]

        # 1. 负熵奖励 (鼓励分散)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        neg_entropy_loss = -entropy.mean()  # 负号是因为我们要最大化熵

        # 2. 最大概率惩罚 (防止过于确定)
        max_probs, _ = torch.max(action_probs, dim=-1)
        max_prob_penalty = max_probs.mean()

        # 3. 批次内动作分布相似性惩罚
        if batch_size > 1:
            # 计算批次内动作分布的余弦相似性
            probs_flat = action_probs.view(batch_size, -1)
            norm = torch.norm(probs_flat, dim=1, keepdim=True)
            probs_normalized = probs_flat / (norm + 1e-8)
            similarity = torch.mm(probs_normalized, probs_normalized.t())

            # 取上三角部分（排除对角线）
            similarity_loss = torch.triu(similarity, diagonal=1).mean()
        else:
            similarity_loss = 0.0

        # 组合多样性损失
        diversity_loss = neg_entropy_loss + 0.5 * max_prob_penalty + 0.1 * similarity_loss

        return diversity_loss

    def update_exploration_params(self, episode):
        """更新探索参数"""
        # ε衰减
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # 温度衰减
        self.temperature = max(self.min_temperature,
                               self.temperature * self.temperature_decay)

        # 自适应调整：如果奖励停滞，增加探索
        if len(self.reward_history) > 50:
            recent_rewards = list(self.reward_history)[-50:]
            reward_std = np.std(recent_rewards)

            if reward_std < 0.1:  # 奖励变化小，可能是局部最优
                # 临时增加探索
                self.epsilon = min(0.5, self.epsilon * 1.1)
                self.temperature = min(2.0, self.temperature * 1.1)

                # 调整多样性权重
                self.diversity_weight = min(0.05, self.diversity_weight * 1.2)
                print(f"📈 Episode {episode}: 增加探索 (ε={self.epsilon:.3f}, τ={self.temperature:.3f})")

    def reset_exploration(self):
        """重置探索参数"""
        self.epsilon = min(0.5, self.initial_epsilon * 1.5)
        self.temperature = min(2.0, self.initial_temperature * 1.5)
        print(f"🔄 重置探索参数: ε={self.epsilon:.3f}, τ={self.temperature:.3f}")

    def record_reward(self, reward):
        """记录奖励历史"""
        self.reward_history.append(reward)

    def get_exploration_stats(self):
        """获取探索统计"""
        total = sum(self.exploration_stats.values())
        if total == 0:
            return {k: 0.0 for k in self.exploration_stats.keys()}

        stats = {k: v / total for k, v in self.exploration_stats.items()}
        stats.update({
            'epsilon_value': self.epsilon,
            'temperature': self.temperature,
            'diversity_weight': self.diversity_weight
        })
        return stats

    def _process_state(self, state):
        """处理状态输入"""
        try:
            if isinstance(state, torch.Tensor):
                state_tensor = state.clone().detach()
            elif isinstance(state, dict):
                if 'combined_state' in state:
                    state_data = state['combined_state']
                else:
                    state_data = list(state.values())[0]

                if isinstance(state_data, torch.Tensor):
                    state_tensor = state_data.clone().detach()
                else:
                    state_tensor = torch.FloatTensor(state_data)
            else:
                state_tensor = torch.FloatTensor(state)

            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)

            return state_tensor.to(self.device)

        except Exception as e:
            print(f"❌ Machine Agent 状态处理错误: {e}")
            return torch.zeros((1, self.state_dim), device=self.device)

    def save(self, filepath):
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'temperature': self.temperature,
                'diversity_weight': self.diversity_weight,
                'exploration_stats': self.exploration_stats,
            }, filepath)
        except Exception as e:
            print(f"❌ 模型保存错误: {e}")

    def load(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载探索参数
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'temperature' in checkpoint:
                self.temperature = checkpoint['temperature']
            if 'diversity_weight' in checkpoint:
                self.diversity_weight = checkpoint['diversity_weight']
            if 'exploration_stats' in checkpoint:
                self.exploration_stats = checkpoint['exploration_stats']

        except Exception as e:
            print(f"❌ 模型加载错误: {e}")