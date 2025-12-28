# training/curriculum_trainer.py
from EDFP_FFSP_DRL.training.ippo_trainer import IPPOTrainer
import numpy as np
import torch
from EDFP_FFSP_DRL.agents.job_agent import JobAgent
from EDFP_FFSP_DRL.agents.machine_agent import MachineAgent

class CurriculumTrainer(IPPOTrainer):
    """课程学习训练器 - 逐步增加难度"""

    def __init__(self, config, job_agent, machine_agent, env):
        super().__init__(config, job_agent, machine_agent, env)

        # 课程学习参数
        self.curriculum_stages = [
            {'num_jobs': max(3, config.NUM_JOBS // 4), 'num_machines': config.NUM_MACHINES, 'episodes': 1000},
            {'num_jobs': max(5, config.NUM_JOBS // 2), 'num_machines': config.NUM_MACHINES, 'episodes': 1000},
            {'num_jobs': max(7, 3 * config.NUM_JOBS // 4), 'num_machines': config.NUM_MACHINES, 'episodes': 1000},
            {'num_jobs': config.NUM_JOBS, 'num_machines': config.NUM_MACHINES, 'episodes': 2000}
        ]

        self.current_stage = 0
        self.stage_episode = 0

        print("✅ Curriculum PPO Trainer Initialized")
        print(f"📚 Curriculum Stages: {len(self.curriculum_stages)}")

    def _check_and_update_agents_for_new_stage(self):
        """
        【新增方法】检查并动态调整状态维度，如果需要则重新创建智能体。
        在每个课程阶段开始时调用。
        """
        # 1. 检查新的环境状态维度
        state = self.env.reset()

        # 确保能正确获取状态维度，处理 Tensor 和 Numpy 数组
        if not isinstance(state, torch.Tensor) and hasattr(state, 'shape'):
            state = torch.tensor(state)
        elif not isinstance(state, torch.Tensor) and isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        # 获取实际状态维度 (处理一维向量)
        actual_state_dim = state.shape[0] if hasattr(state, 'shape') and len(state.shape) > 0 else len(state)

        # 2. 如果维度不匹配，动态调整配置并重新创建智能体
        if actual_state_dim != self.config.STATE_DIM:
            print(f"🔄 状态维度不匹配: 旧={self.config.STATE_DIM}, 新={actual_state_dim}. 动态调整...")

            # 更新配置
            self.config.STATE_DIM = actual_state_dim
            self.config.JOB_STATE_DIM = actual_state_dim
            self.config.MACHINE_STATE_DIM = actual_state_dim

            # 重新创建智能体，使其神经网络的输入层与新的状态维度匹配
            self.job_agent = JobAgent(self.config)
            self.machine_agent = MachineAgent(self.config)
            print(f"✅ 智能体已重新创建以适应新的状态维度: {self.config.STATE_DIM}")
        else:
            # 即使维度匹配，也重置环境以确保状态正确
            self.env.reset()
            print(f"✅ 状态维度匹配: {self.config.STATE_DIM}. 重新设置环境.")

    def train(self, num_episodes=None):
        """课程学习训练"""
        total_episodes = num_episodes or self.config.NUM_EPISODES

        print(f"🎯 Starting Curriculum training for {total_episodes} episodes")

        episode_counter = 0

        while episode_counter < total_episodes and self.current_stage < len(self.curriculum_stages):
            stage = self.curriculum_stages[self.current_stage]

            print(f"\n📖 Curriculum Stage {self.current_stage + 1}: "
                  f"Jobs={stage['num_jobs']}, Machines={stage['num_machines']}, "
                  f"Episodes={stage['episodes']}")

            # 重新创建环境（改变规模）
            self._create_environment_for_stage(stage)

            # 2. 【修复关键点】检查并更新智能体以匹配新的状态维度
            self._check_and_update_agents_for_new_stage()

            # 训练当前阶段
            stage_episodes = min(stage['episodes'], total_episodes - episode_counter)

            for episode in range(stage_episodes):
                try:
                    # 收集经验
                    batch, episode_reward, steps, _ = self.collect_rollout()

                    # 记录统计
                    self.training_history['episode'].append(episode_counter)
                    self.training_history['rewards'].append(episode_reward)
                    self.training_history['steps'].append(steps)
                    self.training_history['profits'].append(self.env.total_profit)
                    self.training_history['makespans'].append(self.env._calculate_makespan())

                    # 更新智能体
                    if batch is not None:
                        job_loss, machine_loss = self.update_agents(batch)
                        self.training_history['job_losses'].append(job_loss)
                        self.training_history['machine_losses'].append(machine_loss)
                    else:
                        job_loss, machine_loss = 0.0, 0.0

                    # 输出进度
                    if episode % self.config.MONITOR_INTERVAL == 0:
                        stage_info = f"(Stage {self.current_stage + 1}, {stage['num_jobs']}J/{stage['num_machines']}M)"
                        print(f"📈 Episode {episode_counter} {stage_info}: "
                              f"Reward={episode_reward:.2f}, Steps={steps}")

                    episode_counter += 1
                    self.stage_episode += 1

                except Exception as e:
                    print(f"❌ Training episode {episode_counter} error: {e}")
                    continue

            # 进入下一阶段
            self.current_stage += 1
            self.stage_episode = 0

        return self.training_history

    def _create_environment_for_stage(self, stage):
        """为当前阶段创建环境"""
        from EDFP_FFSP_DRL.environment.dffsp_env import DFFSPEnvironment

        # 更新环境规模
        self.env = DFFSPEnvironment(
            num_jobs=stage['num_jobs'],
            num_stages=self.config.NUM_STAGES,
            num_machines=stage['num_machines'],
            config=self.config
        )