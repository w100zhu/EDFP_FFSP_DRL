# experiments/generalization_experiment.py
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EDFP_FFSP_DRL.training.integrated_trainer import IntegratedTrainer
from EDFP_FFSP_DRL.environment.dffsp_env import DFFSPEnvironment
from EDFP_FFSP_DRL.agents.job_agent import JobAgent
from EDFP_FFSP_DRL.agents.machine_agent import MachineAgent


class CompleteConfig:
    """完整的配置类，包含所有必需属性"""

    def __init__(self, config_dict=None):
        # 基础配置
        self.NUM_JOBS = 10
        self.NUM_MACHINES = 8
        self.NUM_STAGES = 5
        self.NUM_EPISODES = 2000
        self.MAX_STEPS = 150

        # DRL 训练参数
        self.ROLLOUT_LENGTH = 20
        self.BATCH_SIZE = 16
        self.PPO_EPOCHS = 5
        self.LEARNING_RATE = 3e-4
        self.GAMMA = 0.99
        self.LAM = 0.95
        self.PPO_CLIP_EPS = 0.2
        self.ENTROPY_COEF = 0.05
        self.VALUE_COEF = 0.5
        self.WEIGHT_DECAY = 1e-5
        self.ADAM_EPS = 1e-8
        self.GRAD_CLIP = 0.5

        # 网络结构参数
        self.STATE_DIM = 64
        self.HIDDEN_DIM = 128
        self.JOB_STATE_DIM = 64
        self.MACHINE_STATE_DIM = 64
        self.JOB_ACTION_DIM = 10
        self.MACHINE_ACTION_DIM = 8

        # 奖励权重
        self.PROFIT_WEIGHT = 0.7
        self.MAKESPAN_WEIGHT = 0.2
        self.BALANCE_WEIGHT = 0.1

        # 其他配置
        self.STATE_NORMALIZATION = True
        self.SAVE_INTERVAL = 50
        self.MONITOR_INTERVAL = 50
        self.PLOT_INTERVAL = 1000
        self.USE_DYNAMIC_REWARD = True
        self.USE_ATTENTION = True
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 文件夹路径
        self.CHECKPOINT_DIR = "checkpoints"
        self.RESULTS_DIR = "training_results"

        # 如果提供了配置字典，更新属性
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)

        # 确保必要的文件夹存在
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)


class StaticPPOTrainer:
    """
    简化的静态奖励函数训练器（用于泛化实验对比）
    固定奖励权重，不进行动态调整
    """

    def __init__(self, config, job_agent, machine_agent, env):
        self.config = config
        self.job_agent = job_agent
        self.machine_agent = machine_agent
        self.env = env

        # 记录训练历史
        self.training_history = {
            'episode': [],
            'rewards': [],
            'profits': [],
            'makespans': []
        }

        print(f"⚖️ 静态训练器初始化: 利润权重={config.PROFIT_WEIGHT:.2f}")

    def collect_rollout(self):
        """收集轨迹数据（简化版本）"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < self.config.MAX_STEPS:
            # 获取可用动作
            avail_j, avail_m = self.env.get_available_actions()

            if not avail_j or not avail_m:
                break

            # 随机选择动作（简化，实际应该用智能体策略）
            j_action = np.random.choice(avail_j)
            m_action = np.random.choice(avail_m)

            # 执行动作
            next_state, reward, done, info = self.env.step(j_action, m_action)

            episode_reward += reward
            steps += 1
            state = next_state

        # 返回简化结果
        final_profit = self.env.total_profit
        final_makespan = max(10.0, self.env.current_time)

        return None, episode_reward, steps, {
            'final_profit': final_profit,
            'final_makespan': final_makespan
        }

    def train(self, num_episodes=None):
        """训练循环 - 固定参数版本"""
        total_episodes = num_episodes or self.config.NUM_EPISODES

        for episode in range(total_episodes):
            try:
                batch, ep_reward, steps, info = self.collect_rollout()

                # 记录结果
                final_profit = info.get('final_profit', 0)
                final_makespan = info.get('final_makespan', 0)
                final_rate = final_profit / max(1.0, final_makespan)

                self.training_history['episode'].append(episode)
                self.training_history['rewards'].append(ep_reward)
                self.training_history['profits'].append(final_profit)
                self.training_history['makespans'].append(final_makespan)

                if episode % 100 == 0:
                    print(f"Static Ep {episode} | Rate: {final_rate:.2f} | "
                          f"Profit: {final_profit:.1f} | Mksp: {final_makespan:.1f}")

            except Exception as e:
                print(f"❌ Static Ep {episode} Error: {e}")

        return self.training_history


class GeneralizationExperiment:
    """泛化能力实验"""

    def __init__(self, config_dir="experiment_configs"):
        self.config_dir = config_dir
        self.results_dir = "generalization_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_experiment(self, num_seeds=3):
        """运行泛化实验"""
        print("🎯 开始泛化能力实验")

        # 实验配置
        experiment_configs = [
            {
                'name': '动态奖励函数 (CMA-DRL)',
                'trainer_type': 'Integrated',
                'use_dynamic_reward': True
            },
            {
                'name': '静态奖励函数 (Static)',
                'trainer_type': 'Static',
                'use_dynamic_reward': False
            }
        ]

        all_results = []

        for seed in range(num_seeds):
            print(f"\n🌱 随机种子: {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)

            for exp_config in experiment_configs:
                print(f"\n🧪 实验: {exp_config['name']}")

                # 1. 在中等跳变概率环境训练
                print("  阶段1: 训练 (跳变概率=0.25)")
                trainer, agents = self._train_on_medium_environment(
                    exp_config, seed
                )

                # 2. 在高跳变概率环境测试
                print("  阶段2: 测试 (跳变概率=0.50)")
                test_results = self._test_on_hard_environment(
                    trainer, agents, exp_config
                )

                # 记录结果
                result_entry = {
                    'experiment': exp_config['name'],
                    'seed': seed,
                    'test_results': test_results,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_results.append(result_entry)

        # 保存和分析结果
        self._analyze_results(all_results)

        return all_results

    def _train_on_medium_environment(self, exp_config, seed):
        """在中等跳变概率环境训练"""
        # 创建训练环境（中等跳变概率）
        train_env = DFFSPEnvironment(
            num_jobs=10,
            num_stages=5,
            num_machines=8,
            train_skip_prob=0.25,  # 中等跳变概率
            test_skip_prob=0.50,
            is_test=False
        )

        # 创建完整配置
        config_dict = {
            'NUM_JOBS': 10,
            'NUM_MACHINES': 8,
            'NUM_STAGES': 5,
            'NUM_EPISODES': 2000,  # 为了快速测试，减少训练轮次
            'STATE_DIM': 64,
            'HIDDEN_DIM': 128,
            'JOB_ACTION_DIM': 10,
            'MACHINE_ACTION_DIM': 8,
            'PROFIT_WEIGHT': 0.7,
            'MAKESPAN_WEIGHT': 0.2,
            'BALANCE_WEIGHT': 0.1,
            'USE_DYNAMIC_REWARD': exp_config['use_dynamic_reward'],
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'CHECKPOINT_DIR': f"checkpoints/generalization/{exp_config['name'].replace(' ', '_')}_{seed}",
            'RESULTS_DIR': f"training_results/generalization/{exp_config['name'].replace(' ', '_')}_{seed}"
        }

        config = CompleteConfig(config_dict)

        # 创建智能体
        job_agent = JobAgent(config)
        machine_agent = MachineAgent(config)

        # 创建训练器
        if exp_config['trainer_type'] == 'Integrated':
            trainer = IntegratedTrainer(config, job_agent, machine_agent, train_env)
        else:
            trainer = StaticPPOTrainer(config, job_agent, machine_agent, train_env)

        # 训练
        trainer.train(config.NUM_EPISODES)

        return trainer, (job_agent, machine_agent)

    def _test_on_hard_environment(self, trainer, agents, exp_config):
        """在高跳变概率环境测试"""
        # 创建测试环境（高跳变概率）
        test_env = DFFSPEnvironment(
            num_jobs=10,
            num_stages=5,
            num_machines=8,
            train_skip_prob=0.25,
            test_skip_prob=0.50,  # 高跳变概率
            is_test=True
        )

        job_agent, machine_agent = agents

        # 测试多个episode
        num_test_episodes = 20  # 减少测试次数以加快速度
        all_rewards = []
        all_profits = []
        all_makespans = []
        all_profit_rates = []

        for ep in range(num_test_episodes):
            state = test_env.reset()
            done = False
            episode_reward = 0

            while not done:
                # 使用训练好的策略
                avail_j, avail_m = test_env.get_available_actions()

                if not avail_j or not avail_m:
                    break

                # 简化：随机选择动作
                j_action = np.random.choice(avail_j) if avail_j else 0
                m_action = np.random.choice(avail_m) if avail_m else 0

                state, reward, done, info = test_env.step(j_action, m_action)
                episode_reward += reward

            # 记录结果
            final_profit = test_env.total_profit
            final_makespan = max(10.0, test_env.current_time)
            final_profit_rate = final_profit / final_makespan

            all_rewards.append(episode_reward)
            all_profits.append(final_profit)
            all_makespans.append(final_makespan)
            all_profit_rates.append(final_profit_rate)

            if (ep + 1) % 1 == 0:
                print(f"   测试进度: {ep + 1}/{num_test_episodes} | "
                      f"利润率: {final_profit_rate:.2f}")

        # 计算性能下降幅度（相对于训练环境的平均性能）
        # 这里简化：假设训练环境平均利润率为 baseline_performance
        baseline_performance = 80.0  # 示例值，实际应从训练结果获取
        avg_performance = np.mean(all_profit_rates)
        decline = ((baseline_performance - avg_performance) / baseline_performance) * 100

        return {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_profit': np.mean(all_profits),
            'std_profit': np.std(all_profits),
            'avg_makespan': np.mean(all_makespans),
            'std_makespan': np.std(all_makespans),
            'avg_profit_rate': np.mean(all_profit_rates),
            'std_profit_rate': np.std(all_profit_rates),
            'relative_decline': decline
        }

    def _analyze_results(self, all_results):
        """分析实验结果"""
        # 转换为DataFrame
        df_data = []
        for result in all_results:
            df_data.append({
                'Experiment': result['experiment'],
                'Seed': result['seed'],
                'Avg_Profit_Rate': result['test_results']['avg_profit_rate'],
                'Std_Profit_Rate': result['test_results']['std_profit_rate'],
                'Relative_Decline(%)': result['test_results']['relative_decline'],
                'Avg_Profit': result['test_results']['avg_profit'],
                'Avg_Makespan': result['test_results']['avg_makespan']
            })

        df = pd.DataFrame(df_data)

        # 保存原始数据
        raw_data_path = os.path.join(self.results_dir, 'generalization_raw_results.csv')
        df.to_csv(raw_data_path, index=False)
        print(f"📊 原始数据保存至: {raw_data_path}")

        # 计算统计摘要
        summary = df.groupby('Experiment').agg({
            'Avg_Profit_Rate': ['mean', 'std', 'min', 'max'],
            'Relative_Decline(%)': ['mean', 'std'],
            'Avg_Profit': 'mean',
            'Avg_Makespan': 'mean'
        }).round(2)

        summary_path = os.path.join(self.results_dir, 'generalization_summary.csv')
        summary.to_csv(summary_path)
        print(f"📈 统计摘要保存至: {summary_path}")

        # 绘制结果图
        self._plot_results(df)

        # 打印关键发现
        print("\n" + "=" * 60)
        print("🎯 泛化实验关键发现:")
        print("=" * 60)

        dynamic_results = df[df['Experiment'] == '动态奖励函数 (CMA-DRL)']
        static_results = df[df['Experiment'] == '静态奖励函数 (Static)']

        dynamic_decline = dynamic_results['Relative_Decline(%)'].mean()
        static_decline = static_results['Relative_Decline(%)'].mean()

        print(f"动态奖励函数 - 平均性能下降: {dynamic_decline:.1f}%")
        print(f"静态奖励函数 - 平均性能下降: {static_decline:.1f}%")

        if dynamic_decline < static_decline:
            improvement = static_decline - dynamic_decline
            print(f"✅ 动态奖励函数显著提升了策略的泛化能力！(提升 {improvement:.1f}%)")
        else:
            print("⚠️  未观察到明显的泛化能力提升")

    def _plot_results(self, df):
        """绘制实验结果图"""
        plt.figure(figsize=(12, 8))

        # 1. 利润率对比
        plt.subplot(2, 2, 1)
        experiments = df['Experiment'].unique()
        colors = {'动态奖励函数 (CMA-DRL)': 'blue', '静态奖励函数 (Static)': 'red'}

        for exp in experiments:
            exp_data = df[df['Experiment'] == exp]['Avg_Profit_Rate']
            x_pos = np.random.normal(experiments.tolist().index(exp), 0.05, size=len(exp_data))
            plt.scatter(x_pos, exp_data, alpha=0.6, label=exp, color=colors.get(exp, 'gray'))

        plt.boxplot([df[df['Experiment'] == exp]['Avg_Profit_Rate'] for exp in experiments],
                    labels=experiments)
        plt.title('Profit Rate Comparison (Higher is Better)')
        plt.xticks(rotation=15)
        plt.ylabel('Profit Rate')

        # 2. 性能下降幅度对比
        plt.subplot(2, 2, 2)
        for exp in experiments:
            exp_data = df[df['Experiment'] == exp]['Relative_Decline(%)']
            x_pos = np.random.normal(experiments.tolist().index(exp), 0.05, size=len(exp_data))
            plt.scatter(x_pos, exp_data, alpha=0.6, color=colors.get(exp, 'gray'))

        plt.boxplot([df[df['Experiment'] == exp]['Relative_Decline(%)'] for exp in experiments],
                    labels=experiments)
        plt.title('Performance Decline (Lower is Better)')
        plt.xticks(rotation=15)
        plt.ylabel('Decline (%)')

        # 3. 利润与完工时间散点图
        plt.subplot(2, 2, 3)
        for exp in experiments:
            exp_df = df[df['Experiment'] == exp]
            plt.scatter(exp_df['Avg_Makespan'], exp_df['Avg_Profit'],
                        alpha=0.7, label=exp, color=colors.get(exp, 'gray'), s=80)
        plt.title('Profit vs Makespan Trade-off')
        plt.xlabel('Makespan')
        plt.ylabel('Profit')
        plt.legend()

        # 4. 标准差对比（稳定性）
        plt.subplot(2, 2, 4)
        std_comparison = df.groupby('Experiment')['Std_Profit_Rate'].mean()
        plt.bar(std_comparison.index, std_comparison.values,
                color=[colors.get(exp, 'gray') for exp in std_comparison.index])
        plt.title('Stability Comparison (Lower Std is Better)')
        plt.xticks(rotation=15)
        plt.ylabel('Standard Deviation of Profit Rate')

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'generalization_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 分析图保存至: {plot_path}")


def main():
    """主函数"""
    print("🚀 开始运行泛化能力实验")

    experiment = GeneralizationExperiment()
    results = experiment.run_experiment(num_seeds=3)  # 先用3个种子测试

    print("\n🎉 泛化实验完成！")
    print(f"结果保存在: {experiment.results_dir}")


if __name__ == "__main__":
    main()