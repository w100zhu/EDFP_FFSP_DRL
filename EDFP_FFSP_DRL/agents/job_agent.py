# agents/job_agent.py (利润导向优化版 - 修复版)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import deque
import random


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
            # 确保 mask 和 logits 在同一设备
            mask = mask.to(logits.device)
            # 将非法动作的概率压到极小 (使用 -1e9 而不是 -inf 以防计算错误)
            logits = logits.masked_fill(mask == 0, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs + 1e-8  # 避免零概率


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


class JobAgent:
    def __init__(self, config):
        self.config = config
        self.state_dim = config.STATE_DIM
        self.action_dim = config.JOB_ACTION_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.device = config.DEVICE

        # 探索参数
        self.exploration_strategy = getattr(config, 'EXPLORATION_STRATEGY', 'profit_guided')
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
        # 初始化所有可能的探索类型
        self.exploration_stats = {
            "epsilon": 0, "entropy": 0, "softmax": 0, "greedy": 0,
            "random": 0, "profit_guided": 0, "policy": 0
        }
        self.reward_history = deque(maxlen=100)
        self.action_diversity_history = deque(maxlen=50)

        # 多样性正则化权重
        self.diversity_weight = getattr(config, 'DIVERSITY_WEIGHT', 0.01)

        # 利润导向探索
        self.profit_focused_exploration = True
        self.profit_estimation = {}  # 动作利润估计
        self.profit_learning_rate = 0.1  # 利润估计学习率
        self.action_profit_history = {}  # 记录每个动作的利润历史

        # 学习率调度
        self.learning_rate = config.LEARNING_RATE
        self.lr_decay_steps = getattr(config, 'LR_DECAY_STEPS', 1000)
        self.lr_decay_rate = getattr(config, 'LR_DECAY_RATE', 0.95)

        # 突变探索机制
        self.mutation_enabled = True
        self.mutation_probability = 0.05
        self.mutation_strength = 0.3

        # 局部最优突破机制
        self.no_improvement_count = 0
        self.best_reward = -float('inf')
        self.breakout_threshold = 100

        print(f"🔧 Job Agent初始化: 状态维度={self.state_dim}, 动作维度={self.action_dim}")
        print(f"  探索策略: {self.exploration_strategy}, ε={self.epsilon:.3f}, τ={self.temperature:.3f}")
        print(f"  利润导向探索: {'启用' if self.profit_focused_exploration else '禁用'}")

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
            'diversity_scores': [],
            'profit_estimates': []
        }

        # 动作使用计数
        self.action_usage_count = {}

    def _get_action_mask(self, env):
        """生成动作掩码: 1=合法, 0=非法"""
        mask = torch.ones(self.action_dim).to(self.device)

        if env is not None:
            try:
                # 获取可用动作 (jobs, machines)
                available_jobs, _ = env.get_available_actions()

                # 初始化全0掩码
                mask = torch.zeros(self.action_dim).to(self.device)

                if len(available_jobs) > 0:
                    # 将可用工件对应的位置设为1
                    mask[available_jobs] = 1.0
                else:
                    # 如果没有可用工件，防止全0导致报错，默认允许动作0
                    mask[0] = 1.0

            except Exception as e:
                print(f"⚠️ Job Mask生成失败: {e}")
                mask = torch.ones(self.action_dim).to(self.device)

        return mask

    def act(self, state, env=None, training=True, episode=None):
        """利润导向的动作选择 (训练用)"""
        if not training:
            return self.greedy_act(state, env)

        try:
            state_tensor = self._process_state(state)
            mask = self._get_action_mask(env)

            # 获取可用动作索引
            valid_actions = torch.where(mask == 1)[0].cpu().numpy()

            # 检查是否需要突变探索（突破局部最优）
            if (self.mutation_enabled and episode is not None and
                    self._should_perform_mutation(episode)):
                action_idx = self._mutation_exploration(valid_actions, env)
                if action_idx is not None:
                    action = torch.tensor([action_idx], device=self.device)
                    dist = torch.distributions.Categorical(self.actor(state_tensor, mask=mask.unsqueeze(0)))
                    log_prob = dist.log_prob(action)
                    value = self.critic(state_tensor)
                    self.exploration_stats['random'] += 1
                    return action.item(), log_prob, value

            with torch.no_grad():
                # 计算动作概率
                action_probs = self.actor(state_tensor, mask=mask.unsqueeze(0))

                # 利润导向探索（40%概率）
                if (self.profit_focused_exploration and
                        np.random.random() < 0.4 and
                        len(valid_actions) > 0):

                    profit_action = self._profit_guided_exploration(valid_actions, env)
                    if profit_action is not None:
                        action = torch.tensor([profit_action], device=self.device)
                        dist = torch.distributions.Categorical(action_probs)
                        log_prob = dist.log_prob(action)
                        value = self.critic(state_tensor)
                        self.exploration_stats['profit_guided'] += 1
                        return action.item(), log_prob, value

                # 探索策略选择
                explore_type = self._select_exploration_strategy(episode)
                # 确保 explore_type 是 Python 字符串类型
                explore_type = str(explore_type) if not isinstance(explore_type, str) else explore_type

                # 确保探索类型在统计字典中
                if explore_type not in self.exploration_stats:
                    self.exploration_stats[explore_type] = 0

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
                    # 标准策略梯度探索（包括"policy"类型）
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                value = self.critic(state_tensor)

                # 计算动作熵用于多样性统计
                action_entropy = dist.entropy().item()
                self.action_diversity_history.append(action_entropy)

                # 记录动作使用
                action_item = action.item()
                self.action_usage_count[action_item] = self.action_usage_count.get(action_item, 0) + 1

                return action.item(), log_prob, value

        except Exception as e:
            print(f"❌ Job Agent动作选择错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回随机有效动作
            if env is not None:
                available_jobs, _ = env.get_available_actions()
                if len(available_jobs) > 0:
                    random_action = np.random.choice(available_jobs)
                    return random_action, torch.tensor([0.0]).to(self.device), torch.tensor([0.0]).to(self.device)
            random_action = np.random.randint(self.action_dim)
            return random_action, torch.tensor([0.0]).to(self.device), torch.tensor([0.0]).to(self.device)

    def _select_exploration_strategy(self, episode):
        """选择探索策略"""
        if self.exploration_strategy == "epsilon":
            return "epsilon" if np.random.random() < 0.3 else "policy"
        elif self.exploration_strategy == "profit_guided":
            # 利润导向策略：早期更多探索，后期更多利用
            if episode is None or episode < 500:
                # 早期：更多随机探索
                strategies = ["epsilon", "random", "softmax", "policy"]
                weights = [0.3, 0.3, 0.2, 0.2]
                chosen_idx = np.random.choice(len(strategies), p=weights)
                return strategies[chosen_idx]
            elif episode < 2000:
                # 中期：平衡探索和利用
                strategies = ["epsilon", "random", "softmax", "policy"]
                weights = [0.2, 0.2, 0.2, 0.4]
                chosen_idx = np.random.choice(len(strategies), p=weights)
                return strategies[chosen_idx]
            else:
                # 后期：更多利用，偶尔探索
                strategies = ["epsilon", "random", "softmax", "policy"]
                weights = [0.1, 0.1, 0.1, 0.7]
                chosen_idx = np.random.choice(len(strategies), p=weights)
                return strategies[chosen_idx]
        else:
            # 默认策略
            strategies = ["epsilon", "policy", "softmax"]
            weights = [0.3, 0.5, 0.2]
            chosen_idx = np.random.choice(len(strategies), p=weights)
            return strategies[chosen_idx]

    def _profit_guided_exploration(self, valid_actions, env=None):
        """利润导向的探索"""
        if len(valid_actions) == 0:
            return None

        # 如果有利润估计，使用UCB算法
        if self.profit_estimation and len(valid_actions) > 1:
            ucb_scores = []
            for action in valid_actions:
                if action in self.profit_estimation:
                    profit_mean, profit_count = self.profit_estimation[action]
                    total_count = sum(c for _, c in self.profit_estimation.values())
                    exploration_bonus = np.sqrt(2 * np.log(total_count + 1) / (profit_count + 1e-8))
                    ucb_score = profit_mean + exploration_bonus * 0.5  # 降低探索系数
                else:
                    ucb_score = 8.0  # 未探索动作的默认高分
                ucb_scores.append(ucb_score)

            # 选择UCB分数最高的动作
            best_idx = np.argmax(ucb_scores)
            return valid_actions[best_idx]

        return None

    def _should_perform_mutation(self, episode):
        """判断是否应该进行突变探索"""
        # 每100轮进行一次突变探索
        if episode % 100 == 0:
            return np.random.random() < 0.3

        # 如果连续很多轮没有进步，增加突变概率
        if self.no_improvement_count > self.breakout_threshold:
            increased_prob = min(0.5, self.mutation_probability *
                                 (1 + self.no_improvement_count / self.breakout_threshold))
            return np.random.random() < increased_prob

        return False

    def _mutation_exploration(self, valid_actions, env=None):
        """突变探索：跳出局部最优"""
        if len(valid_actions) == 0:
            return None

        # 策略1：选择最少使用的动作
        if self.action_usage_count:
            usage_counts = [self.action_usage_count.get(a, 0) for a in valid_actions]
            min_usage = min(usage_counts) if usage_counts else 0
            least_used_actions = [a for a, c in zip(valid_actions, usage_counts) if c == min_usage]
            if len(least_used_actions) > 0:
                return np.random.choice(least_used_actions)

        # 策略2：完全随机
        return np.random.choice(valid_actions)

    def greedy_act(self, state, env=None):
        """贪婪动作 (测试/推理用)"""
        try:
            state_tensor = self._process_state(state)
            mask = self._get_action_mask(env)

            with torch.no_grad():
                # 传入 mask
                action_probs = self.actor(state_tensor, mask=mask.unsqueeze(0))

                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    print("⚠️ Job Agent 动作概率包含NaN或Inf")
                    # 返回随机有效动作
                    if env is not None:
                        available_jobs, _ = env.get_available_actions()
                        if len(available_jobs) > 0:
                            return np.random.choice(available_jobs)
                    return np.random.randint(self.action_dim)

                action = torch.argmax(action_probs, dim=-1)

            return action.item()

        except Exception as e:
            print(f"❌ Job Agent 贪婪动作选择错误: {e}")
            # 返回随机有效动作
            if env is not None:
                available_jobs, _ = env.get_available_actions()
                if len(available_jobs) > 0:
                    return np.random.choice(available_jobs)
            return np.random.randint(self.action_dim)

    def update(self, batch):
        if batch is None or len(batch['states']) == 0:
            return 0.0

        try:
            total_policy_loss = 0.0
            total_diversity_loss = 0.0

            # 检查 batch 中是否有 mask 信息，如果没有则默认全1
            masks = batch.get('job_masks', None)

            for _ in range(self.config.PPO_EPOCHS):
                # 传入 masks 进行更新
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
                new_log_probs = dist.log_prob(batch['job_actions'])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch['job_log_probs'])
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
            print(f"❌ Job Agent update error: {e}")
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

    def update_exploration_params(self, episode, episode_reward):
        """更新探索参数"""
        # 更新无进步计数
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

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

                # 增加突变概率
                self.mutation_probability = min(0.15, self.mutation_probability * 1.3)

                if episode % 100 == 0:
                    print(f"📈 Episode {episode}: 增加探索 (ε={self.epsilon:.3f}, τ={self.temperature:.3f})")

    def update_profit_estimation(self, action, profit):
        """更新动作的利润估计"""
        if action not in self.profit_estimation:
            self.profit_estimation[action] = (profit, 1)
        else:
            old_mean, old_count = self.profit_estimation[action]
            new_count = old_count + 1
            new_mean = old_mean + (profit - old_mean) / new_count
            self.profit_estimation[action] = (new_mean, new_count)

        # 记录利润历史
        if action not in self.action_profit_history:
            self.action_profit_history[action] = []
        self.action_profit_history[action].append(profit)
        if len(self.action_profit_history[action]) > 20:
            self.action_profit_history[action] = self.action_profit_history[action][-20:]

    def reset_exploration(self):
        """重置探索参数"""
        self.epsilon = min(0.5, self.initial_epsilon * 1.5)
        self.temperature = min(2.0, self.initial_temperature * 1.5)
        self.no_improvement_count = 0
        print(f"🔄 重置探索参数: ε={self.epsilon:.3f}, τ={self.temperature:.3f}")

    def record_reward(self, reward):
        """记录奖励历史"""
        self.reward_history.append(reward)

    def get_exploration_stats(self):
        """获取探索统计"""
        # 清理空的统计项
        stats_to_remove = []
        for key in self.exploration_stats:
            if self.exploration_stats[key] == 0 and key not in ["epsilon", "entropy", "softmax", "greedy", "random",
                                                                "profit_guided", "policy"]:
                stats_to_remove.append(key)

        for key in stats_to_remove:
            del self.exploration_stats[key]

        total = sum(self.exploration_stats.values())
        if total == 0:
            return {k: 0.0 for k in self.exploration_stats.keys()}

        stats = {k: v / total for k, v in self.exploration_stats.items()}
        stats.update({
            'epsilon_value': self.epsilon,
            'temperature': self.temperature,
            'diversity_weight': self.diversity_weight,
            'no_improvement_count': self.no_improvement_count,
            'best_reward': self.best_reward
        })
        return stats

    def get_profit_estimation_stats(self):
        """获取利润估计统计"""
        if not self.profit_estimation:
            return {"count": 0, "avg_profit": 0.0}

        total_count = sum(count for _, count in self.profit_estimation.values())
        avg_profits = [mean for mean, _ in self.profit_estimation.values()]
        avg_profit = np.mean(avg_profits) if avg_profits else 0.0

        return {
            "count": len(self.profit_estimation),
            "total_count": total_count,
            "avg_profit": avg_profit,
            "max_profit": max([mean for mean, _ in self.profit_estimation.values()]) if self.profit_estimation else 0.0
        }

    def _process_state(self, state):
        """处理状态输入"""
        try:
            if isinstance(state, torch.Tensor):
                state_tensor = state.clone().detach()
            elif isinstance(state, dict):
                # 兼容字典输入
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
            print(f"❌ Job Agent 状态处理错误: {e}")
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
                'profit_estimation': self.profit_estimation,
                'best_reward': self.best_reward,
                'no_improvement_count': self.no_improvement_count,
                'action_usage_count': self.action_usage_count,
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
            if 'profit_estimation' in checkpoint:
                self.profit_estimation = checkpoint['profit_estimation']
            if 'best_reward' in checkpoint:
                self.best_reward = checkpoint['best_reward']
            if 'no_improvement_count' in checkpoint:
                self.no_improvement_count = checkpoint['no_improvement_count']
            if 'action_usage_count' in checkpoint:
                self.action_usage_count = checkpoint['action_usage_count']

        except Exception as e:
            print(f"❌ 模型加载错误: {e}")