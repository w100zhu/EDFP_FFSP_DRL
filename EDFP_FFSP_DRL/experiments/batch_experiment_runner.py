# experiments/batch_experiment_runner.py
import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

from EDFP_FFSP_DRL.training.multihead_trainer import MultiHeadTrainer
from EDFP_FFSP_DRL.training.balanced_ppo_trainer import BalancedPPOTrainer
from EDFP_FFSP_DRL.training.base_ppo_trainer import BasePPOTrainer
from EDFP_FFSP_DRL.training.curriculum_trainer import CurriculumTrainer
from EDFP_FFSP_DRL.training.integrated_trainer import IntegratedTrainer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BatchExperimentRunner:
    """批量实验运行器"""

    def __init__(self, configs_dir: str = "experiment_configs"):
        self.configs_dir = configs_dir
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)

        # 训练器工厂
        self.trainer_factory = TrainerFactory()

        # 实验历史
        self.experiment_history = []

    def run_experiment_group(self, group_name: str, num_repeats: int = 3):
        """运行实验组"""
        group_dir = os.path.join(self.configs_dir, group_name.replace(' ', '_'))

        if not os.path.exists(group_dir):
            print(f"实验组目录不存在: {group_dir}")
            return

        config_files = [f for f in os.listdir(group_dir) if f.endswith('.json')]

        print(f"🎯 开始运行实验组: {group_name}")
        print(f"📁 找到 {len(config_files)} 个实验配置")

        for config_file in config_files:
            config_path = os.path.join(group_dir, config_file)

            # 加载配置
            with open(config_path, 'r') as f:
                config = json.load(f)

            exp_id = config['EXPERIMENT_ID']

            print(f"\n🔬 开始实验: {exp_id}")

            # 重复实验
            for repeat in range(num_repeats):
                print(f"  重复 {repeat + 1}/{num_repeats}")

                try:
                    # 运行实验
                    results = self._run_single_experiment(config, repeat)

                    # 保存结果
                    self._save_experiment_results(exp_id, repeat, results, group_name)

                    # 记录到历史
                    self.experiment_history.append({
                        'experiment_id': exp_id,
                        'repeat': repeat,
                        'group': group_name,
                        'config': config,
                        'results': results,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                except Exception as e:
                    print(f"❌ 实验 {exp_id} 重复 {repeat} 失败: {e}")
                    traceback.print_exc()

        # 分析实验组结果
        self._analyze_group_results(group_name)

    def _run_single_experiment(self, config: Dict, repeat: int) -> Dict:
        """运行单个实验"""
        # 创建动态配置类
        DynamicConfig = self._create_dynamic_config(config)
        dynamic_config = DynamicConfig()

        # 创建环境
        from EDFP_FFSP_DRL.environment.dffsp_env import DFFSPEnvironment
        env = DFFSPEnvironment(
            num_jobs=config['NUM_JOBS'],
            num_stages=config['NUM_STAGES'],
            num_machines=config['NUM_MACHINES'],
            config=dynamic_config
        )

        sample_state = env.reset()
        if hasattr(sample_state, 'shape'):
            real_state_dim = sample_state.shape[0]
        else:
            real_state_dim = len(sample_state)

        print(f"🔄 检测到环境真实状态维度: {real_state_dim} (配置: {dynamic_config.STATE_DIM})")

        # 更新所有相关的维度配置
        dynamic_config.STATE_DIM = real_state_dim
        dynamic_config.JOB_STATE_DIM = real_state_dim
        dynamic_config.MACHINE_STATE_DIM = real_state_dim

        print(f"✅ 已更新配置状态维度为: {dynamic_config.STATE_DIM}")

        # 创建智能体
        from EDFP_FFSP_DRL.agents.job_agent import JobAgent
        from EDFP_FFSP_DRL.agents.machine_agent import MachineAgent

        job_agent = JobAgent(dynamic_config)
        machine_agent = MachineAgent(dynamic_config)

        # 创建训练器
        trainer_type = config.get('TRAINER_TYPE', 'BalancedPPO')
        trainer = self.trainer_factory.create_trainer(
            trainer_type, dynamic_config, job_agent, machine_agent, env
        )

        # 训练
        training_results = trainer.train(config['NUM_EPISODES'])

        # ==================== 【新增】强制保存逻辑 ====================
        print(f"💾 [Runner] 训练结束，正在强制保存最终模型 (Episode {config['NUM_EPISODES']})...")
        try:
            # 优先调用公共保存方法 (IntegratedTrainer 和修复后的 BasePPO 都有这个)
            if hasattr(trainer, 'save_checkpoint'):
                trainer.save_checkpoint(config['NUM_EPISODES'])
            # 兼容旧版 BasePPO 的私有方法
            elif hasattr(trainer, '_save_checkpoint'):
                trainer._save_checkpoint(config['NUM_EPISODES'])
            else:
                # 最后的兜底方案：手动保存
                save_dir = getattr(trainer.config, 'CHECKPOINT_DIR', 'checkpoints')
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"checkpoint_{config['NUM_EPISODES']}_forced.pth")

                # 尝试获取 policy
                state = {}
                if hasattr(trainer, 'job_agent') and hasattr(trainer.job_agent, 'policy'):
                    state['job_policy'] = trainer.job_agent.policy.state_dict()
                if hasattr(trainer, 'machine_agent') and hasattr(trainer.machine_agent, 'policy'):
                    state['machine_policy'] = trainer.machine_agent.policy.state_dict()

                if state:
                    torch.save(state, path)
                    print(f"⚠️ 使用兜底方案保存模型: {path}")
                else:
                    print("❌ 无法找到模型参数，保存失败")
        except Exception as e:
            print(f"❌ 强制保存失败: {e}")
        # ===========================================================

        # 测试
        test_results = trainer.evaluate(num_episodes=10)

        # 收集结果
        results = {
            'training_metrics': self._extract_training_metrics(training_results),
            'test_metrics': test_results,
            'config': config,
            'repeat': repeat
        }

        return results

    def _create_dynamic_config(self, config_dict: Dict):
        """创建动态配置类"""

        class DynamicConfig:
            def __init__(self):
                for key, value in config_dict.items():
                    setattr(self, key, value)

                # ==================== 修改区域：优化文件夹结构 ====================
                # 获取组名并替换空格，例如 "全面实验" -> "全面实验"
                group_name = config_dict.get('EXPERIMENT_GROUP', 'Default_Group').replace(' ', '_')
                exp_id = config_dict['EXPERIMENT_ID']

                # 修改路径结构为: checkpoints/组名/实验ID
                # 这样可以确保不同实验组的文件夹完全物理隔离
                self.CHECKPOINT_DIR = os.path.join("checkpoints", group_name, exp_id)

                # 结果目录同样隔离
                self.RESULTS_DIR = os.path.join("training_results", group_name, exp_id)

                # 确保目录存在
                os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
                os.makedirs(self.RESULTS_DIR, exist_ok=True)
                # ===============================================================

        return DynamicConfig

    def _extract_training_metrics(self, training_results) -> Dict:
        """从训练结果中提取指标"""
        # 假设training_results是一个包含训练历史的字典
        if not isinstance(training_results, dict):
            return {}

        # 兼容不同的键名 (rewards vs reward_history)
        rewards = training_results.get('rewards', training_results.get('reward_history', []))
        profits = training_results.get('profits', training_results.get('profit_history', []))
        makespans = training_results.get('makespans', training_results.get('makespan_history', []))

        metrics = {
            'final_reward': rewards[-1] if rewards else 0,
            'final_profit': profits[-1] if profits else 0,
            'final_makespan': makespans[-1] if makespans else 0,
            'convergence_episode': training_results.get('convergence_episode', 0),
            'training_time': training_results.get('training_time', 0),
            'reward_history': rewards,
            'profit_history': profits,
            'makespan_history': makespans
        }

        return metrics

    def _save_experiment_results(self, exp_id: str, repeat: int, results: Dict, group_name: str):
        """保存实验结果"""
        # 创建结果目录 (按组隔离)
        result_dir = os.path.join(self.results_dir, group_name.replace(' ', '_'), exp_id)
        os.makedirs(result_dir, exist_ok=True)

        # 保存详细结果
        result_file = os.path.join(result_dir, f"repeat_{repeat}.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 保存训练曲线数据（每50步）
        if 'training_metrics' in results and 'reward_history' in results['training_metrics']:
            self._save_training_curves(results['training_metrics'], result_dir, repeat)

    def _save_training_curves(self, metrics: Dict, result_dir: str, repeat: int):
        """保存训练曲线数据（每50步一个点）"""
        reward_history = metrics.get('reward_history', [])
        profit_history = metrics.get('profit_history', [])
        makespan_history = metrics.get('makespan_history', [])

        if not reward_history:
            return

        # 计算每50步的平均值
        window_size = 50
        reward_points = []
        profit_points = []
        makespan_points = []
        episode_points = []

        for i in range(0, len(reward_history), window_size):
            end_idx = min(i + window_size, len(reward_history))

            if end_idx - i >= window_size // 2:  # 至少有25个数据点
                avg_reward = np.mean(reward_history[i:end_idx])
                avg_profit = np.mean(profit_history[i:end_idx]) if profit_history else 0
                avg_makespan = np.mean(makespan_history[i:end_idx]) if makespan_history else 0

                reward_points.append(avg_reward)
                profit_points.append(avg_profit)
                makespan_points.append(avg_makespan)
                episode_points.append(end_idx)

        # 保存为CSV
        curve_data = pd.DataFrame({
            'episode': episode_points,
            'reward': reward_points,
            'profit': profit_points,
            'makespan': makespan_points
        })

        csv_path = os.path.join(result_dir, f"training_curves_{repeat}.csv")
        curve_data.to_csv(csv_path, index=False)

        # 绘制曲线图
        self._plot_training_curves(curve_data, result_dir, repeat)

    def _plot_training_curves(self, curve_data: pd.DataFrame, result_dir: str, repeat: int):
        """绘制训练曲线图"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 奖励曲线
        axes[0].plot(curve_data['episode'], curve_data['reward'], 'b-', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Reward Curve')
        axes[0].grid(True, alpha=0.3)

        # 利润曲线
        axes[1].plot(curve_data['episode'], curve_data['profit'], 'g-', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Profit')
        axes[1].set_title('Training Profit Curve')
        axes[1].grid(True, alpha=0.3)

        # 完工时长曲线
        axes[2].plot(curve_data['episode'], curve_data['makespan'], 'r-', linewidth=2)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Makespan')
        axes[2].set_title('Training Makespan Curve')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(result_dir, f"training_curves_{repeat}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_group_results(self, group_name: str):
        """分析实验组结果"""
        group_dir = os.path.join(self.results_dir, group_name.replace(' ', '_'))

        if not os.path.exists(group_dir):
            return

        # 收集所有实验结果
        all_results = []

        for exp_id in os.listdir(group_dir):
            exp_dir = os.path.join(group_dir, exp_id)

            if not os.path.isdir(exp_dir):
                continue

            # 读取重复实验的结果
            for result_file in os.listdir(exp_dir):
                if result_file.startswith('repeat_') and result_file.endswith('.json'):
                    result_path = os.path.join(exp_dir, result_file)

                    try:
                        with open(result_path, 'r') as f:
                            result = json.load(f)

                        # 提取关键指标
                        test_metrics = result.get('test_metrics', {})
                        config = result.get('config', {})

                        summary = {
                            'experiment_id': exp_id,
                            'group': group_name,
                            'avg_test_reward': np.mean(test_metrics.get('rewards', [0])),
                            'avg_test_profit': np.mean(test_metrics.get('profits', [0])),
                            'avg_test_makespan': np.mean(test_metrics.get('makespans', [0])),
                            'avg_completion_rate': np.mean(test_metrics.get('completed_jobs', [0])) / config.get(
                                'NUM_JOBS', 10),
                            'num_jobs': config.get('NUM_JOBS', 10),
                            'num_machines': config.get('NUM_MACHINES', 8),
                            'profit_weight': config.get('PROFIT_WEIGHT', 0.5),
                            'makespan_weight': config.get('MAKESPAN_WEIGHT', 0.05),
                            'trainer_type': config.get('TRAINER_TYPE', 'BalancedPPO')
                        }

                        all_results.append(summary)
                    except:
                        continue

        if not all_results:
            return

        # 创建DataFrame
        df = pd.DataFrame(all_results)

        # 保存汇总结果
        summary_path = os.path.join(group_dir, 'experiment_summary.csv')
        df.to_csv(summary_path, index=False)

        # 绘制分析图
        self._plot_group_analysis(df, group_dir, group_name)

    def _plot_group_analysis(self, df: pd.DataFrame, group_dir: str, group_name: str):
        """绘制实验组分析图"""
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))

        if '环境规模实验' in group_name:
            # 环境规模实验分析
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 按产品数量的平均利润
            sns.barplot(x='num_jobs', y='avg_test_profit', data=df, ax=axes[0, 0])
            axes[0, 0].set_title('Average Profit by Number of Jobs')
            axes[0, 0].set_xlabel('Number of Jobs')
            axes[0, 0].set_ylabel('Average Profit')

            # 按机器数量的平均利润
            sns.barplot(x='num_machines', y='avg_test_profit', data=df, ax=axes[0, 1])
            axes[0, 1].set_title('Average Profit by Number of Machines')
            axes[0, 1].set_xlabel('Number of Machines')
            axes[0, 1].set_ylabel('Average Profit')

            # 按产品数量的平均完工时长
            sns.barplot(x='num_jobs', y='avg_test_makespan', data=df, ax=axes[1, 0])
            axes[1, 0].set_title('Average Makespan by Number of Jobs')
            axes[1, 0].set_xlabel('Number of Jobs')
            axes[1, 0].set_ylabel('Average Makespan')

            # 按机器数量的完成率
            sns.barplot(x='num_machines', y='avg_completion_rate', data=df, ax=axes[1, 1])
            axes[1, 1].set_title('Completion Rate by Number of Machines')
            axes[1, 1].set_xlabel('Number of Machines')
            axes[1, 1].set_ylabel('Completion Rate')

        elif '奖励权重实验' in group_name:
            # 奖励权重实验分析
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 利润权重对利润的影响
            sns.scatterplot(x='profit_weight', y='avg_test_profit', data=df, ax=axes[0, 0])
            axes[0, 0].set_title('Profit Weight vs Average Profit')
            axes[0, 0].set_xlabel('Profit Weight')
            axes[0, 0].set_ylabel('Average Profit')

            # 完工时长权重对完工时长的影响
            sns.scatterplot(x='makespan_weight', y='avg_test_makespan', data=df, ax=axes[0, 1])
            axes[0, 1].set_title('Makespan Weight vs Average Makespan')
            axes[0, 1].set_xlabel('Makespan Weight')
            axes[0, 1].set_ylabel('Average Makespan')

            # 权重组合对奖励的影响
            pivot = df.pivot_table(values='avg_test_reward',
                                   index='profit_weight',
                                   columns='makespan_weight')
            sns.heatmap(pivot, annot=True, fmt='.2f', ax=axes[1, 0])
            axes[1, 0].set_title('Reward Heatmap by Weights')

            # 权重组合对完成率的影响
            pivot_completion = df.pivot_table(values='avg_completion_rate',
                                              index='profit_weight',
                                              columns='makespan_weight')
            sns.heatmap(pivot_completion, annot=True, fmt='.3f', ax=axes[1, 1])
            axes[1, 1].set_title('Completion Rate Heatmap by Weights')


        elif '算法对比实验' in group_name or '消融实验' in group_name or '全面实验' in group_name:
            # 算法/消融/全面实验分析
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            # 算法对比柱状图
            metrics = ['avg_test_reward', 'avg_test_profit', 'avg_test_makespan', 'avg_completion_rate']
            titles = ['Average Reward', 'Average Profit', 'Average Makespan', 'Completion Rate']
            for idx, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[idx // 2, idx % 2]
                # 使用柱状图展示不同 Trainer 的性能差异
                if 'trainer_type' in df.columns:
                    sns.barplot(x='trainer_type', y=metric, data=df, ax=ax, palette='viridis')
                    ax.set_title(f'{title} by Trainer Type')
                    ax.set_xlabel('Trainer Type')
                    ax.set_ylabel(title)
                    ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_path = os.path.join(group_dir, 'experiment_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


class TrainerFactory:
    """训练器工厂"""

    def create_trainer(self, trainer_type: str, config, job_agent, machine_agent, env):
        """创建训练器"""
        if trainer_type == 'BasePPO':
            return BasePPOTrainer(config, job_agent, machine_agent, env)
        elif trainer_type == 'BalancedPPO':
            return BalancedPPOTrainer(config, job_agent, machine_agent, env)
        elif trainer_type == 'Curriculum':
            return CurriculumTrainer(config, job_agent, machine_agent, env)
        elif trainer_type == 'MultiHead':
            return MultiHeadTrainer(config, job_agent, machine_agent, env)
        elif trainer_type == 'Integrated':
            return IntegratedTrainer(config, job_agent, machine_agent, env)
        else:
            # 默认使用BalancedPPO
            return BalancedPPOTrainer(config, job_agent, machine_agent, env)