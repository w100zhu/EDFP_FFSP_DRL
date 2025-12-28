# training/balanced_ppo_trainer.py
from EDFP_FFSP_DRL.training.ippo_trainer import IPPOTrainer


class BalancedPPOTrainer(IPPOTrainer):
    """平衡PPO训练器 - 继承自IPPO，带奖励平衡"""

    def __init__(self, config, job_agent, machine_agent, env):
        super().__init__(config, job_agent, machine_agent, env)
        print("✅ Balanced PPO Trainer Initialized")

        # 在环境中设置奖励权重
        if hasattr(env, 'current_profit_weight'):
            env.current_profit_weight = getattr(config, 'PROFIT_WEIGHT', 0.5)
            env.current_makespan_weight = getattr(config, 'MAKESPAN_WEIGHT', 0.05)

    def evaluate(self, num_episodes=10):
        """
        评估模型性能（新增方法）
        """
        # 1. 切换到评估模式 (关闭 Dropout/Batch Norm 等)
        if hasattr(self.job_agent, 'policy'):
            self.job_agent.policy.eval()
        if hasattr(self.machine_agent, 'policy'):
            self.machine_agent.policy.eval()

        rewards = []
        profits = []
        makespans = []
        completed_jobs = []

        print(f"📊 正在评估 BasePPO 模型 ({num_episodes} 轮)...")

        import torch  # 确保导入了 torch

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    # 评估时不需要计算梯度
                    job_action, _ = self.job_agent.select_action(state)
                    machine_action, _ = self.machine_agent.select_action(state)

                next_state, reward, done, info = self.env.step(job_action, machine_action)
                episode_reward += reward
                state = next_state

            # 记录单轮结果
            rewards.append(episode_reward)

            # 尝试获取环境指标 (根据你的环境实现，属性名可能不同)
            # 这里使用了 getattr 防止报错，如果没有属性则记录为 0
            profits.append(getattr(self.env, 'current_profit', 0))
            makespans.append(getattr(self.env, 'current_makespan', 0))

            # 尝试获取完成工件数
            completed_count = 0
            if hasattr(self.env, 'completed_jobs'):
                completed_count = len(self.env.completed_jobs)
            elif hasattr(self.env, 'completed_jobs_count'):
                completed_count = self.env.completed_jobs_count
            completed_jobs.append(completed_count)

        # 2. 恢复训练模式
        if hasattr(self.job_agent, 'policy'):
            self.job_agent.policy.train()
        if hasattr(self.machine_agent, 'policy'):
            self.machine_agent.policy.train()

        avg_reward = sum(rewards) / len(rewards)
        print(f"   平均奖励: {avg_reward:.2f}")

        return {
            'rewards': rewards,
            'profits': profits,
            'makespans': makespans,
            'completed_jobs': completed_jobs
        }