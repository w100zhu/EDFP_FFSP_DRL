# training/base_ppo_trainer.py
import torch
import numpy as np
from collections import deque
import os
import traceback


class BasePPOTrainer:
    """基础PPO训练器 - 基准版本 (已修复属性缺失错误)"""

    def __init__(self, config, job_agent, machine_agent, env):
        self.config = config
        self.job_agent = job_agent
        self.machine_agent = machine_agent
        self.env = env

        # 修复：从配置中初始化关键属性
        self.max_steps = getattr(config, 'MAX_STEPS', 100)
        self.state_dim = getattr(config, 'STATE_DIM', 64)
        self.device = getattr(config, 'DEVICE', 'cpu')
        self.gamma = getattr(config, 'GAMMA', 0.99)

        # 训练统计
        self.training_history = {
            'episode': [],
            'rewards': [],
            'profits': [],
            'makespans': [],
            'job_losses': [],
            'machine_losses': [],
            'steps': []
        }

        print("✅ Base PPO Trainer Initialized (Fixed)")

    def train(self, num_episodes):
        """训练循环 - 基础版本"""
        print(f"🎯 Starting Base PPO training for {num_episodes} episodes")

        for episode in range(num_episodes):
            try:
                # 收集经验
                result = self.collect_rollout()
                if result is None:
                    continue

                batch, episode_reward, steps, *_ = result

                # 记录统计
                self.training_history['episode'].append(episode)
                self.training_history['rewards'].append(episode_reward)
                self.training_history['steps'].append(steps)

                # 获取环境统计
                current_profit = self.env.total_profit if hasattr(self.env, 'total_profit') else 0
                self.training_history['profits'].append(current_profit)

                current_makespan = self.env._calculate_makespan()
                self.training_history['makespans'].append(current_makespan)

                # 更新智能体
                if batch is not None:
                    job_loss, machine_loss = self.update_agents(batch)
                    self.training_history['job_losses'].append(job_loss)
                    self.training_history['machine_losses'].append(machine_loss)
                else:
                    job_loss, machine_loss = 0.0, 0.0

                # 输出进度
                if episode % self.config.MONITOR_INTERVAL == 0:
                    self._print_progress(episode, episode_reward, steps, job_loss, machine_loss)

                # 保存检查点
                if episode % self.config.SAVE_INTERVAL == 0 and episode > 0:
                    self._save_checkpoint(episode)

            except Exception as e:
                print(f"❌ Training episode {episode} error: {e}")
                traceback.print_exc()
                continue

        return self.training_history

    def collect_rollout(self):
        """收集经验 - 基础版本"""
        try:
            batch_states, batch_actions, batch_rewards = [], [], []
            batch_next_states, batch_dones, batch_log_probs, batch_values = [], [], [], []

            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < self.max_steps:
                # 获取动作
                job_result = self.job_agent.act(state)
                machine_result = self.machine_agent.act(state)

                # 处理 Agent 返回值
                if isinstance(job_result, tuple):
                    job_action, job_log_prob, job_value = job_result
                else:
                    job_action, job_log_prob, job_value = job_result, 0.0, 0.0

                if isinstance(machine_result, tuple):
                    machine_action, machine_log_prob, machine_value = machine_result
                else:
                    machine_action, machine_log_prob, machine_value = machine_result, 0.0, 0.0

                # 确保动作是标量
                if hasattr(job_action, 'item'): job_action = job_action.item()
                if hasattr(machine_action, 'item'): machine_action = machine_action.item()

                # 执行动作
                step_result = self.env.step(job_action, machine_action)
                next_state = step_result[0]
                reward = step_result[1]
                done = step_result[2]
                info = step_result[-1]  # info 通常在最后

                # 确保存储的数据类型正确
                if hasattr(reward, 'item'): reward = reward.item()
                if hasattr(job_log_prob, 'item'): job_log_prob = job_log_prob.item()
                if hasattr(machine_log_prob, 'item'): machine_log_prob = machine_log_prob.item()
                if hasattr(job_value, 'item'): job_value = job_value.item()
                if hasattr(machine_value, 'item'): machine_value = machine_value.item()

                # 存储经验
                batch_states.append(state)
                batch_actions.append([job_action, machine_action])
                batch_rewards.append(reward)
                batch_next_states.append(next_state)
                batch_dones.append(done)
                batch_log_probs.append([job_log_prob, machine_log_prob])

                # 存储价值
                avg_value = (job_value + machine_value) / 2
                batch_values.append(avg_value)

                state = next_state
                episode_reward += reward
                steps += 1

            # 转换为numpy数组
            if len(batch_states) > 0:
                if isinstance(batch_states[0], torch.Tensor):
                    batch_states_np = torch.stack(batch_states).cpu().numpy()
                    batch_next_states_np = torch.stack(batch_next_states).cpu().numpy()
                else:
                    batch_states_np = np.array(batch_states)
                    batch_next_states_np = np.array(batch_next_states)
            else:
                return None

            batch = (
                batch_states_np,
                np.array(batch_actions),
                np.array(batch_rewards),
                batch_next_states_np,
                np.array(batch_dones),
                np.array(batch_log_probs),
                np.array(batch_values)
            )

            return batch, episode_reward, steps

        except Exception as e:
            print(f"Error in collect_rollout: {e}")
            traceback.print_exc()
            return None

    def update_agents(self, batch):
        """更新智能体 - 基础版本"""
        states, actions, rewards, next_states, dones, log_probs, values = batch

        if len(states) == 0:
            return 0.0, 0.0

        try:
            # 重构 batch 字典以适配 Agent.update 接口
            tensor_batch = {
                'states': torch.FloatTensor(states).to(self.device),
                'job_actions': torch.LongTensor(actions[:, 0]).to(self.device),
                'machine_actions': torch.LongTensor(actions[:, 1]).to(self.device),
                'job_log_probs': torch.FloatTensor(log_probs[:, 0]).to(self.device),
                'machine_log_probs': torch.FloatTensor(log_probs[:, 1]).to(self.device),
                'rewards': torch.FloatTensor(rewards).to(self.device),
                'dones': torch.FloatTensor(dones).to(self.device),
                'values': torch.FloatTensor(values).to(self.device)
            }

            # 计算优势函数
            returns = []
            gae = 0
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + self.gamma * (values[i + 1] if i + 1 < len(rewards) else 0) * (1 - dones[i]) - \
                        values[i]
                gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
                returns.insert(0, gae + values[i])

            tensor_batch['returns'] = torch.FloatTensor(returns).to(self.device)
            tensor_batch['advantages'] = tensor_batch['returns'] - tensor_batch['values']

            # 更新 Agents
            job_loss = self.job_agent.update(tensor_batch)
            machine_loss = self.machine_agent.update(tensor_batch)

            return job_loss, machine_loss

        except Exception as e:
            print(f"Error calculating returns or updating: {e}")
            traceback.print_exc()
            return 0.0, 0.0

    def evaluate(self, num_episodes=10):
        """评估方法"""
        print(f"📊 Evaluating BasePPO for {num_episodes} episodes...")
        rewards = []

        # 临时切换到评估模式
        if hasattr(self.job_agent, 'policy'): self.job_agent.policy.eval()
        if hasattr(self.machine_agent, 'policy'): self.machine_agent.policy.eval()

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                # 简单贪婪策略或采样
                with torch.no_grad():
                    job_act = self.job_agent.select_action(state)[0]
                    mac_act = self.machine_agent.select_action(state)[0]
                state, r, done, _ = self.env.step(job_act, mac_act)
                ep_reward += r
            rewards.append(ep_reward)

        # 恢复训练模式
        if hasattr(self.job_agent, 'policy'): self.job_agent.policy.train()
        if hasattr(self.machine_agent, 'policy'): self.machine_agent.policy.train()

        avg_reward = sum(rewards) / len(rewards)
        print(f"   Avg Reward: {avg_reward:.2f}")
        return {'rewards': rewards, 'avg_reward': avg_reward}

    def _print_progress(self, episode, reward, steps, job_loss, machine_loss):
        print(
            f"📈 Episode {episode}: Reward={reward:.2f}, Steps={steps}, Losses=[Job: {job_loss:.4f}, Machine: {machine_loss:.4f}]")

    def _save_checkpoint(self, episode):
        try:
            job_path = os.path.join(self.config.CHECKPOINT_DIR, f'job_agent_base_{episode}.pth')
            machine_path = os.path.join(self.config.CHECKPOINT_DIR, f'machine_agent_base_{episode}.pth')
            self.job_agent.save(job_path)
            self.machine_agent.save(machine_path)
        except Exception as e:
            pass