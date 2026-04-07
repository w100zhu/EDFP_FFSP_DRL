# 新建 EDFP_FFSP_DRL/experiments/sensor_robustness_analysis.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EDFP_FFSP_DRL.environment.dffsp_env import DFFSPEnvironment
from EDFP_FFSP_DRL.agents.job_agent import JobAgent
from EDFP_FFSP_DRL.agents.machine_agent import MachineAgent


class SensorRobustnessAnalyzer:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]  # 传感器误差百分比

    def run_analysis(self):
        print(f"🔬 启动传感器鲁棒性分析 - 模型: {os.path.basename(self.model_path)}")
        results = []

        # 加载 DRL 模型
        ckpt = torch.load(self.model_path, map_location='cpu')
        job_agent = JobAgent(self.config)
        machine_agent = MachineAgent(self.config)
        # 加载逻辑需与 run_comparison.py 保持一致
        # ... (加载 actor 权重代码)

        for noise in self.noise_levels:
            print(f"  🌊 测试观测噪声等级: {noise * 100}%")
            rates = []
            for seed in [42, 100, 2023]:  # 多种子平均
                np.random.seed(seed)
                env = DFFSPEnvironment(self.config.NUM_JOBS, self.config.NUM_STAGES, self.config.NUM_MACHINES,
                                       self.config)
                rate = self._evaluate_with_noise(env, job_agent, machine_agent, noise)
                rates.append(rate)

            results.append({'Noise_Level': noise, 'Avg_Profit_Rate': np.mean(rates)})

        df = pd.DataFrame(results)
        df.to_csv("sensor_robustness_report.csv", index=False)
        self._plot_results(df)

    def _evaluate_with_noise(self, env, job_agent, machine_agent, noise_std):
        state = env.reset()
        done = False
        while not done:
            # 注入噪声模拟传感器不准
            if isinstance(state, torch.Tensor):
                noisy_state = state + torch.randn_like(state) * noise_std
            else:
                noisy_state = state + np.random.normal(0, noise_std, size=state.shape)

            # 推理时必须使用 Action Masking 防止死锁
            # 获取当前可用动作作为掩码
            job_action, _, _ = job_agent.act(noisy_state, env=env, training=False)
            machine_action, _, _ = machine_agent.act(noisy_state, env=env, training=False)

            state, reward, done, info = env.step(job_action, machine_action)

        return env.total_profit / max(1.0, env._calculate_makespan())

    def _plot_results(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df['Noise_Level'], df['Avg_Profit_Rate'], 'o-', linewidth=2, color='darkorange')
        plt.title('System Robustness Analysis: Profit Rate vs. Observation Noise')
        plt.xlabel('Sensor Noise Level (Std Dev)')
        plt.ylabel('Average Profit Rate (Profit/Time)')
        plt.grid(True, alpha=0.3)
        plt.savefig("sensor_robustness_curve.png")
        print("📊 鲁棒性分析曲线已生成: sensor_robustness_curve.png")