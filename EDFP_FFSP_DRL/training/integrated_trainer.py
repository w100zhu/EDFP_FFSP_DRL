# EDFP_FFSP_DRL/training/integrated_trainer.py (完整增强版)
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from EDFP_FFSP_DRL.training.curriculum_trainer import CurriculumTrainer
import matplotlib.pyplot as plt
from collections import deque
import copy


class IntegratedTrainer(CurriculumTrainer):
    """
    【利润率优先集成版本 - 增强版】
    训练目标：最大化单位时间利润 (Profit Rate = Profit / Makespan)

    架构特点：
    1. 增强型注意力：Self-Attention (内部建模) + Cross-Attention (交互建模)
    2. 评价标准：以 Profit Rate 为第一保存标准
    3. 动态权重：支持根据训练阶段动态调整 P/M 权重
    """

    def __init__(self, config, job_agent, machine_agent, env):
        # 1. 初始化课程学习基础
        super().__init__(config, job_agent, machine_agent, env)

        # 2. 明确训练目标：利润率 (Profit Rate)
        self.primary_objective = "profit_rate"

        # 初始权重配置 (用于 Reward Shaping)
        self.profit_weight = getattr(config, 'PROFIT_WEIGHT', 0.7)
        self.makespan_weight = getattr(config, 'MAKESPAN_WEIGHT', 0.2)
        self.balance_weight = getattr(config, 'BALANCE_WEIGHT', 0.1)

        # 同步环境权重
        if hasattr(env, 'update_reward_weights'):
            env.update_reward_weights(self.profit_weight, self.makespan_weight, self.balance_weight)

        # 3. 初始化增强型注意力机制
        self.use_attention = getattr(config, 'USE_ATTENTION', True)
        self.attention_dim = 64
        self.num_heads = 4

        # 维度计算
        self.num_jobs = config.NUM_JOBS
        self.num_machines = config.NUM_MACHINES
        self.machine_feat_dim = 4
        self.global_feat_dim = 7

        # 动态计算 Job 特征维度
        remaining_dim = config.STATE_DIM - (self.num_machines * self.machine_feat_dim) - self.global_feat_dim
        self.job_feat_dim = remaining_dim // self.num_jobs

        print(f"🧠 Attention Enhanced: Self+Cross Mode")
        print(f"   Dims: Job={self.job_feat_dim}, Mach={self.machine_feat_dim}, Attn={self.attention_dim}")

        if self.use_attention:
            self._init_attention_layers()
            self._register_params_to_optimizer()

        # 4. 训练统计记录
        self.training_history = {
            'episode': [], 'rewards': [], 'profits': [], 'makespans': [],
            'profit_rates': [], 'job_losses': [], 'machine_losses': [], 'steps': []
        }

        # 最佳记录跟踪 (以利润率为核心)
        self.best_profit_rate = 0.0
        self.best_profit = 0.0
        self.best_makespan = float('inf')
        self.best_combined_score = 0.0

        # 性能窗口 (用于动态调整)
        self.performance_window = deque(maxlen=50)

        print("✅ Integrated Trainer (Profit-Rate Focused) Initialized")

    def _init_attention_layers(self):
        """初始化双重注意力层 (Self + Cross)"""
        dev = torch.device(self.config.DEVICE)

        # 1. 输入投影 (Projection)
        self.job_in_proj = nn.Linear(self.job_feat_dim, self.attention_dim).to(dev)
        self.machine_in_proj = nn.Linear(self.machine_feat_dim, self.attention_dim).to(dev)

        # 2. Self-Attention (理解内部结构：工件间依赖、机器间状态)
        self.job_self_attn = nn.MultiheadAttention(self.attention_dim, self.num_heads, batch_first=True).to(dev)
        self.machine_self_attn = nn.MultiheadAttention(self.attention_dim, self.num_heads, batch_first=True).to(dev)

        # Self-Attention 的归一化层
        self.norm_job = nn.LayerNorm(self.attention_dim).to(dev)
        self.norm_machine = nn.LayerNorm(self.attention_dim).to(dev)

        # 3. Cross-Attention (Job 查询 Machine 匹配度)
        self.cross_attention = nn.MultiheadAttention(self.attention_dim, self.num_heads, batch_first=True).to(dev)
        self.norm_cross = nn.LayerNorm(self.attention_dim).to(dev)

        # 4. 输出还原 (Back Projection)
        self.job_out_proj = nn.Linear(self.attention_dim, self.job_feat_dim).to(dev)

    def _register_params_to_optimizer(self):
        """注册 Attention 参数到优化器"""
        if hasattr(self.job_agent, 'optimizer'):
            params = list(self.job_in_proj.parameters()) + \
                     list(self.machine_in_proj.parameters()) + \
                     list(self.job_self_attn.parameters()) + \
                     list(self.machine_self_attn.parameters()) + \
                     list(self.norm_job.parameters()) + \
                     list(self.norm_machine.parameters()) + \
                     list(self.cross_attention.parameters()) + \
                     list(self.norm_cross.parameters()) + \
                     list(self.job_out_proj.parameters())
            self.job_agent.optimizer.add_param_group({'params': params})
            print("🔗 Attention params registered to Job Agent Optimizer")

    def _process_state_with_attention(self, state_batch):
        """
        核心增强逻辑：State -> [Job, Machine] -> Self-Attn -> Cross-Attn -> Enhanced State
        """
        if not self.use_attention: return state_batch

        try:
            if not isinstance(state_batch, torch.Tensor):
                state_batch = torch.FloatTensor(state_batch).to(self.config.DEVICE)
            if len(state_batch.shape) == 1: state_batch = state_batch.unsqueeze(0)

            B = state_batch.shape[0]

            # 切片
            j_end = self.num_jobs * self.job_feat_dim
            m_end = j_end + (self.num_machines * self.machine_feat_dim)

            job_flat = state_batch[:, :j_end]
            mach_flat = state_batch[:, j_end:m_end]
            global_part = state_batch[:, m_end:]

            # Reshape -> [Batch, Seq, Dim]
            job_seq = job_flat.view(B, self.num_jobs, self.job_feat_dim)
            mach_seq = mach_flat.view(B, self.num_machines, self.machine_feat_dim)

            # 1. 投影
            j_emb = self.job_in_proj(job_seq)
            m_emb = self.machine_in_proj(mach_seq)

            # 2. Self-Attention (理解内部依赖)
            j_sa, _ = self.job_self_attn(j_emb, j_emb, j_emb)
            j_emb = self.norm_job(j_emb + j_sa)  # Residual

            m_sa, _ = self.machine_self_attn(m_emb, m_emb, m_emb)
            m_emb = self.norm_machine(m_emb + m_sa)  # Residual

            # 3. Cross-Attention (Job 查询 Machine)
            attn_out, _ = self.cross_attention(query=j_emb, key=m_emb, value=m_emb)
            j_emb = self.norm_cross(j_emb + attn_out)  # Residual

            # 4. 还原
            j_delta = self.job_out_proj(j_emb)
            job_seq_final = job_seq + j_delta  # Residual connection to input

            # 5. 重组
            enhanced_state = torch.cat([
                job_seq_final.view(B, -1),
                mach_flat,
                global_part
            ], dim=1)

            return enhanced_state

        except Exception as e:
            # print(f"Attn Fail: {e}")
            return state_batch

    def collect_rollout(self):
        """收集经验 - 插入 Attention 处理"""
        batch_states = []
        batch_job_actions = []
        batch_machine_actions = []
        batch_job_log_probs = []
        batch_machine_log_probs = []
        batch_rewards = []
        batch_dones = []
        batch_job_masks = []
        batch_machine_masks = []

        state = self.env._get_state_representation()
        episode_reward = 0
        steps = 0

        for _ in range(self.config.ROLLOUT_LENGTH):
            # 1. Attention 增强
            with torch.no_grad():
                processed_state = self._process_state_with_attention(state)

            # 2. 获取 Mask
            job_mask = self.job_agent._get_action_mask(self.env)
            machine_mask = self.machine_agent._get_action_mask(self.env)

            # 3. 决策
            if hasattr(self.job_agent, 'act'):
                j_action, j_log_prob, _ = self.job_agent.act(processed_state, self.env, training=True,
                                                             episode=self.training_history['episode'][-1] if
                                                             self.training_history['episode'] else 0)
            else:
                j_action = self.job_agent.select_action(processed_state)
                j_log_prob = torch.tensor(0.0)

            if hasattr(self.machine_agent, 'act'):
                m_action, m_log_prob, _ = self.machine_agent.act(processed_state, self.env, training=True,
                                                                 episode=self.training_history['episode'][-1] if
                                                                 self.training_history['episode'] else 0)
            else:
                m_action = self.machine_agent.select_action(processed_state)
                m_log_prob = torch.tensor(0.0)

            # 4. 执行
            next_state, reward, done, info = self.env.step(j_action, m_action)
            if 'net_profit' in info:
                self.job_agent.update_profit_estimation(j_action, info['net_profit'])

            # 5. 存储
            if isinstance(processed_state, torch.Tensor):
                batch_states.append(processed_state.cpu().numpy().flatten())
            else:
                batch_states.append(processed_state)

            batch_job_actions.append(j_action)
            batch_machine_actions.append(m_action)
            batch_job_log_probs.append(j_log_prob.item())
            batch_machine_log_probs.append(m_log_prob.item())
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_job_masks.append(job_mask.cpu().numpy())
            batch_machine_masks.append(machine_mask.cpu().numpy())

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                state = self.env.reset()
                break

        # 计算 Returns
        returns = []
        R = 0
        for r, d in zip(reversed(batch_rewards), reversed(batch_dones)):
            if d: R = 0
            R = r + self.config.GAMMA * R
            returns.insert(0, R)

        tensor_batch = {
            'states': torch.FloatTensor(np.array(batch_states)).to(self.config.DEVICE),
            'job_actions': torch.LongTensor(batch_job_actions).to(self.config.DEVICE),
            'machine_actions': torch.LongTensor(batch_machine_actions).to(self.config.DEVICE),
            'job_log_probs': torch.FloatTensor(batch_job_log_probs).to(self.config.DEVICE),
            'machine_log_probs': torch.FloatTensor(batch_machine_log_probs).to(self.config.DEVICE),
            'returns': torch.FloatTensor(returns).to(self.config.DEVICE),
            'advantages': torch.FloatTensor(returns).to(self.config.DEVICE),
            'job_masks': torch.FloatTensor(np.array(batch_job_masks)).to(self.config.DEVICE),
            'machine_masks': torch.FloatTensor(np.array(batch_machine_masks)).to(self.config.DEVICE)
        }

        # Value Baseline
        if hasattr(self.job_agent, 'critic'):
            with torch.no_grad():
                values = self.job_agent.critic(tensor_batch['states']).view(-1)
                tensor_batch['advantages'] = tensor_batch['returns'] - values

        return tensor_batch, episode_reward, steps, {}

    def train(self, num_episodes=None):
        """利润率优先的主训练循环"""
        total_episodes = num_episodes or self.config.NUM_EPISODES
        print(f"🎯 Starting Profit-Rate Focused training for {total_episodes} episodes")

        episode_counter = 0

        while episode_counter < total_episodes and self.current_stage < len(self.curriculum_stages):
            stage = self.curriculum_stages[self.current_stage]
            self._create_environment_for_stage(stage)
            self._check_and_update_agents_for_new_stage()

            stage_episodes = min(stage['episodes'], total_episodes - episode_counter)

            for episode_in_stage in range(stage_episodes):
                episode = episode_counter + episode_in_stage
                try:
                    batch, episode_reward, steps, info = self.collect_rollout()

                    # 关键：计算 Profit Rate
                    current_profit = self.env.total_profit
                    current_makespan = self.env._calculate_makespan()
                    # 防止除零
                    safe_makespan = max(1.0, current_makespan)
                    current_profit_rate = current_profit / safe_makespan

                    # 记录历史
                    self.training_history['episode'].append(episode)
                    self.training_history['rewards'].append(episode_reward)
                    self.training_history['profits'].append(current_profit)
                    self.training_history['makespans'].append(current_makespan)
                    self.training_history['profit_rates'].append(current_profit_rate)  # 新增记录
                    self.training_history['steps'].append(steps)

                    # 最佳模型判定：核心是 Profit Rate
                    if current_profit_rate > self.best_profit_rate:
                        self.best_profit_rate = current_profit_rate
                        self.best_profit = current_profit
                        self.best_makespan = current_makespan
                        self.save_checkpoint(episode)
                        print(
                            f"🏆 New Record! Rate={current_profit_rate:.2f} (P={current_profit:.0f}, M={current_makespan:.0f})")

                    # 更新 Agent
                    if batch is not None:
                        j_loss = self.job_agent.update(batch)
                        m_loss = self.machine_agent.update(batch)
                        self.training_history['job_losses'].append(j_loss)
                        self.training_history['machine_losses'].append(m_loss)

                    # 监控输出
                    if episode % self.config.MONITOR_INTERVAL == 0:
                        print(f"📈 Ep {episode} | Rew: {episode_reward:.1f} | Rate: {current_profit_rate:.2f} | "
                              f"Prof: {current_profit:.0f} | Mksp: {current_makespan:.0f} | BestRate: {self.best_profit_rate:.2f}")

                    # 定期保存
                    if episode % self.config.SAVE_INTERVAL == 0:
                        self.save_checkpoint(episode)
                        if episode % 500 == 0: self._save_training_curves(episode)

                    episode_counter += 1

                except Exception as e:
                    print(f"❌ Ep {episode} Error: {e}")
                    import traceback;
                    traceback.print_exc()
                    continue

            self.current_stage += 1

        self._save_training_curves(episode_counter)
        return self.training_history

    def save_checkpoint(self, episode):
        """保存包含 Attention 参数的检查点"""
        checkpoint_dir = getattr(self.config, 'CHECKPOINT_DIR', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")

        state_dict = {
            'episode': episode,
            'best_profit_rate': self.best_profit_rate,
            'best_profit': self.best_profit,
            # Attention
            'job_in_proj': self.job_in_proj.state_dict(),
            'machine_in_proj': self.machine_in_proj.state_dict(),
            'job_self_attn': self.job_self_attn.state_dict(),
            'machine_self_attn': self.machine_self_attn.state_dict(),
            'cross_attention': self.cross_attention.state_dict(),
            'norm_job': self.norm_job.state_dict(),
            'norm_machine': self.norm_machine.state_dict(),
            'norm_cross': self.norm_cross.state_dict(),
            'job_out_proj': self.job_out_proj.state_dict(),
            # Agents
            'job_actor_state': self.job_agent.actor.state_dict(),
            'job_critic_state': self.job_agent.critic.state_dict(),
            'machine_actor_state': self.machine_agent.actor.state_dict(),
            'machine_critic_state': self.machine_agent.critic.state_dict(),
        }
        torch.save(state_dict, path)
        # 额外保存为 best_checkpoint 如果是最佳
        if self.best_profit_rate == state_dict.get('best_profit_rate', 0):
            torch.save(state_dict, os.path.join(checkpoint_dir, "best_checkpoint.pth"))
        print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        if not os.path.exists(path): return
        ckpt = torch.load(path, map_location=self.config.DEVICE)

        # Load Attention
        if 'job_in_proj' in ckpt:
            self.job_in_proj.load_state_dict(ckpt['job_in_proj'])
            self.machine_in_proj.load_state_dict(ckpt['machine_in_proj'])
            self.job_self_attn.load_state_dict(ckpt['job_self_attn'])
            self.machine_self_attn.load_state_dict(ckpt['machine_self_attn'])
            self.cross_attention.load_state_dict(ckpt['cross_attention'])
            self.norm_job.load_state_dict(ckpt['norm_job'])
            self.norm_machine.load_state_dict(ckpt['norm_machine'])
            self.norm_cross.load_state_dict(ckpt['norm_cross'])
            self.job_out_proj.load_state_dict(ckpt['job_out_proj'])

        # Load Agents
        self.job_agent.actor.load_state_dict(ckpt['job_actor_state'])
        self.job_agent.critic.load_state_dict(ckpt['job_critic_state'])
        self.machine_agent.actor.load_state_dict(ckpt['machine_actor_state'])
        self.machine_agent.critic.load_state_dict(ckpt['machine_critic_state'])

        self.best_profit_rate = ckpt.get('best_profit_rate', 0.0)
        print(f"♻️ Loaded checkpoint from {path} (Best Rate: {self.best_profit_rate:.2f})")

    def _save_training_curves(self, episode):
        """保存包含 Profit Rate 的曲线"""
        save_dir = getattr(self.config, 'RESULTS_DIR', 'training_results')
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Rewards
        axes[0, 0].plot(self.training_history['episode'], self.training_history['rewards'], 'b-')
        axes[0, 0].set_title('Rewards')

        # 2. Profit Rate (核心指标)
        axes[0, 1].plot(self.training_history['episode'], self.training_history['profit_rates'], 'g-', linewidth=2)
        axes[0, 1].set_title('Profit Rate (Profit/Makespan)')
        axes[0, 1].axhline(y=self.best_profit_rate, color='r', linestyle='--',
                           label=f'Best: {self.best_profit_rate:.2f}')
        axes[0, 1].legend()

        # 3. Profit & Makespan
        ax2 = axes[1, 0].twinx()
        axes[1, 0].plot(self.training_history['episode'], self.training_history['profits'], 'y-', label='Profit')
        ax2.plot(self.training_history['episode'], self.training_history['makespans'], 'r--', label='Makespan')
        axes[1, 0].set_title('Profit vs Makespan')
        axes[1, 0].legend(loc='upper left')
        ax2.legend(loc='upper right')

        # 4. Losses
        axes[1, 1].plot(self.training_history['job_losses'], label='Job Loss')
        axes[1, 1].plot(self.training_history['machine_losses'], label='Mach Loss')
        axes[1, 1].set_title('Losses')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"training_curves_{episode}.png"))
        plt.close()