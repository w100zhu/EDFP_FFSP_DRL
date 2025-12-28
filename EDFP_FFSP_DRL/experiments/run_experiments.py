# experiments/run_experiments.py
import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EDFP_FFSP_DRL.experiments.experiment_config_generator import ExperimentConfigGenerator
from EDFP_FFSP_DRL.experiments.batch_experiment_runner import BatchExperimentRunner


def main():
    print("🚀 Starting Disassembly Workshop DRL Experiments")

    # ==================== 配置运行目标 ====================
    # 定义要运行的实验组 (只保留你想跑的)
    target_groups = [
        # '环境规模实验',
        # '奖励权重实验',
        # '算法对比实验',
        '全面实验',  # <--- 目前只运行这个
        # '消融实验'
    ]
    # ====================================================

    # 初始化生成器和运行器
    generator = ExperimentConfigGenerator()
    runner = BatchExperimentRunner()

    # === 全局清理：只在程序刚开始时清理一次旧的配置文件夹 ===
    config_root_dir = "experiment_configs"
    if os.path.exists(config_root_dir):
        print(f"🧹 初始化清理: 删除旧的 {config_root_dir}")
        try:
            shutil.rmtree(config_root_dir)
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")
    # =======================================================

    # 循环处理每个目标组
    for group_name in target_groups:
        print(f"\n{'=' * 60}")
        print(f"🎯 Processing Experiment Group: {group_name}")
        print('=' * 60)

        # 1. 只生成当前组的配置
        print(f"📋 Generating configs for: {group_name}...")
        group_configs = generator.generate_configs_by_group_name(group_name)

        if not group_configs:
            print(f"❌ No configs generated for {group_name}, skipping.")
            continue

        # 保存配置 (注意：clear_output_dir=False，因为我们在最上面已经清空过了)
        generator.save_configs(group_configs, output_dir=config_root_dir, clear_output_dir=False)
        print(f"✅ Generated {len(group_configs)} configs for {group_name}")

        # 2. 运行当前组
        print(f"🏃 Running: {group_name}...")
        # num_repeats=1 表示每个实验只跑一次
        runner.run_experiment_group(group_name, num_repeats=1)

    print("\n🎉 All target experiments completed!")

    # 3. 生成综合报告
    print("\n📊 Step 3: Generating comprehensive report...")
    generate_comprehensive_report()


def generate_comprehensive_report():
    """生成综合实验报告"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    results_dir = "experiment_results"

    # 收集所有实验结果
    all_results = []

    if not os.path.exists(results_dir):
        print("❌ No experiment results directory found")
        return

    for group in os.listdir(results_dir):
        group_dir = os.path.join(results_dir, group)

        if not os.path.isdir(group_dir):
            continue

        summary_file = os.path.join(group_dir, 'experiment_summary.csv')

        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            df['group'] = group
            all_results.append(df)

    if not all_results:
        print("❌ No experiment results found")
        return

    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)

    # 保存综合结果
    combined_path = os.path.join(results_dir, 'all_experiments_summary.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"✅ Combined results saved: {combined_path}")

    # 生成综合分析图
    generate_comprehensive_plots(combined_df, results_dir)


def generate_comprehensive_plots(df, output_dir):
    """生成综合分析图"""
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 15))

    # 1. 算法性能对比
    plt.subplot(2, 2, 1)
    if 'trainer_type' in df.columns:
        metric_data = []
        for trainer in df['trainer_type'].unique():
            trainer_df = df[df['trainer_type'] == trainer]
            metric_data.append({
                'trainer': trainer,
                'avg_reward': trainer_df['avg_test_reward'].mean(),
                'avg_profit': trainer_df['avg_test_profit'].mean(),
                'avg_makespan': trainer_df['avg_test_makespan'].mean(),
                'avg_completion': trainer_df['avg_completion_rate'].mean()
            })

        metric_df = pd.DataFrame(metric_data)

        # 绘制多指标雷达图（简化版：并列柱状图）
        x = range(len(metric_df))
        width = 0.2

        plt.bar([i - 1.5 * width for i in x], metric_df['avg_reward'], width, label='Reward')
        plt.bar([i - 0.5 * width for i in x], metric_df['avg_profit'], width, label='Profit')
        plt.bar([i + 0.5 * width for i in x], metric_df['avg_completion'], width, label='Completion')

        plt.xticks(x, metric_df['trainer'], rotation=45)
        plt.title('Algorithm Performance Comparison')
        plt.legend()

    # 2. 环境规模影响
    plt.subplot(2, 2, 2)
    if 'num_jobs' in df.columns:
        # 按产品数量分组
        job_groups = df.groupby('num_jobs')

        metrics = ['avg_test_profit', 'avg_test_makespan', 'avg_completion_rate']
        for metric in metrics:
            if metric in df.columns:
                mean_values = job_groups[metric].mean()
                plt.plot(mean_values.index, mean_values.values, 'o-', label=metric)

        plt.xlabel('Number of Jobs')
        plt.ylabel('Metric Value')
        plt.title('Impact of Job Number on Performance')
        plt.legend()

    # 3. 奖励权重分析
    plt.subplot(2, 2, 3)
    if 'profit_weight' in df.columns and 'makespan_weight' in df.columns:
        # 创建热力图数据
        heatmap_data = df.pivot_table(
            values='avg_test_reward',
            index='profit_weight',
            columns='makespan_weight',
            aggfunc='mean'
        )

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Reward Heatmap by Weights')
        plt.xlabel('Makespan Weight')
        plt.ylabel('Profit Weight')

    # 4. 相关性分析
    plt.subplot(2, 2, 4)
    # 选择数值列进行相关性分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['repeat', 'episode']]

    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()

        # 只显示部分相关性强的特征
        if len(corr_matrix) > 10:
            # 选择与奖励相关性最强的特征
            if 'avg_test_reward' in corr_matrix.columns:
                reward_corr = corr_matrix['avg_test_reward'].abs().sort_values(ascending=False)
                top_features = reward_corr.index[:10].tolist()
                corr_matrix = df[top_features].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comprehensive_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive analysis plot saved: {plot_path}")

    # 5. 训练曲线对比（从保存的CSV中加载）
    plot_training_curves_comparison(df, output_dir)


def plot_training_curves_comparison(df, output_dir):
    """绘制训练曲线对比图"""
    import glob

    # 查找训练曲线数据
    curve_files = glob.glob(os.path.join(output_dir, '**', 'training_curves_*.csv'), recursive=True)

    if not curve_files:
        return

    # 按实验组分类
    curve_data = {}

    for curve_file in curve_files[:10]:  # 限制数量，避免过多
        try:
            # 从路径提取实验信息
            path_parts = curve_file.split(os.sep)
            if len(path_parts) >= 3:
                group = path_parts[-3]
                exp_id = path_parts[-2]

                # 读取曲线数据
                curve_df = pd.read_csv(curve_file)

                if group not in curve_data:
                    curve_data[group] = {}

                if exp_id not in curve_data[group]:
                    curve_data[group][exp_id] = []

                curve_data[group][exp_id].append(curve_df)
        except:
            continue

    # 绘制曲线对比
    for group, experiments in curve_data.items():
        if not experiments:
            continue

        plt.figure(figsize=(15, 5))

        metrics = ['reward', 'profit', 'makespan']

        for idx, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, idx)

            for exp_id, curves in experiments.items():
                if curves:
                    # 取第一个重复实验的曲线
                    curve_df = curves[0]
                    if 'episode' in curve_df.columns and metric in curve_df.columns:
                        plt.plot(curve_df['episode'], curve_df[metric], label=exp_id, alpha=0.7)

            plt.xlabel('Episode')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} Curves - {group}')
            plt.legend(fontsize='small')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{group}_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()