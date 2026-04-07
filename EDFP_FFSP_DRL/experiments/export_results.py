# export_results.py
import pandas as pd
import os
import json
import numpy as np
from pathlib import Path


def export_experiment_results():
    """导出所有实验结果到CSV文件"""
    results_dir = "experiment_results"
    output_file = "all_experiments_complete.csv"

    all_data = []

    # 遍历所有实验组
    for group in os.listdir(results_dir):
        group_dir = os.path.join(results_dir, group)
        if not os.path.isdir(group_dir):
            continue

        # 遍历每个实验
        for exp_id in os.listdir(group_dir):
            exp_dir = os.path.join(group_dir, exp_id)
            if not os.path.isdir(exp_dir):
                continue

            # 读取重复实验的结果
            repeat_results = []
            for repeat_file in os.listdir(exp_dir):
                if repeat_file.startswith('repeat_') and repeat_file.endswith('.json'):
                    repeat_path = os.path.join(exp_dir, repeat_file)

                    try:
                        with open(repeat_path, 'r') as f:
                            result = json.load(f)

                        # 从训练历史中提取最终性能
                        training_metrics = result.get('training_metrics', {})
                        test_metrics = result.get('test_metrics', {})
                        config = result.get('config', {})

                        # 获取最后的训练奖励（最后10轮平均）
                        reward_history = training_metrics.get('reward_history', [])
                        profit_history = training_metrics.get('profit_history', [])
                        makespan_history = training_metrics.get('makespan_history', [])

                        # 计算最终性能
                        if len(reward_history) > 10:
                            final_reward = np.mean(reward_history[-10:])
                            final_profit = np.mean(profit_history[-10:]) if profit_history else 0
                            final_makespan = np.mean(makespan_history[-10:]) if makespan_history else 0
                        else:
                            final_reward = np.mean(reward_history) if reward_history else 0
                            final_profit = np.mean(profit_history) if profit_history else 0
                            final_makespan = np.mean(makespan_history) if makespan_history else 0

                        # 计算利润率
                        final_profit_rate = final_profit / max(1.0, final_makespan)

                        # 从测试结果获取
                        test_rewards = test_metrics.get('rewards', [])
                        test_profits = test_metrics.get('profits', [])
                        test_makespans = test_metrics.get('makespans', [])

                        # 准备数据行
                        row = {
                            'experiment_id': exp_id,
                            'group': group,
                            'repeat': repeat_file.replace('repeat_', '').replace('.json', ''),

                            # 训练最终性能
                            'final_train_reward': final_reward,
                            'final_train_profit': final_profit,
                            'final_train_makespan': final_makespan,
                            'final_train_profit_rate': final_profit_rate,

                            # 测试性能
                            'avg_test_reward': np.mean(test_rewards) if test_rewards else 0,
                            'avg_test_profit': np.mean(test_profits) if test_profits else 0,
                            'avg_test_makespan': np.mean(test_makespans) if test_makespans else 0,
                            'avg_test_completion': test_metrics.get('avg_completion_rate', 0),

                            # 配置参数
                            'num_jobs': config.get('NUM_JOBS', 10),
                            'num_machines': config.get('NUM_MACHINES', 8),
                            'profit_weight': config.get('PROFIT_WEIGHT', 0.5),
                            'makespan_weight': config.get('MAKESPAN_WEIGHT', 0.05),
                            'trainer_type': config.get('TRAINER_TYPE', 'Unknown'),

                            # 其他信息
                            'config_path': exp_dir
                        }

                        repeat_results.append(row)

                    except Exception as e:
                        print(f"Error processing {repeat_path}: {e}")
                        continue

            # 如果有重复结果，计算平均值
            if repeat_results:
                # 添加每个重复实验
                all_data.extend(repeat_results)

                # 计算该实验的平均值
                avg_row = {}
                if repeat_results:
                    for key in repeat_results[0].keys():
                        if key not in ['experiment_id', 'group', 'repeat', 'config_path']:
                            values = [r[key] for r in repeat_results if isinstance(r[key], (int, float))]
                            avg_row[key] = np.mean(values) if values else 0
                        else:
                            avg_row[key] = repeat_results[0][key]

                    avg_row['repeat'] = 'average'
                    all_data.append(avg_row)

    # 创建DataFrame并保存
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 实验结果已导出到: {output_file}")
        print(f"📊 总记录数: {len(df)}")

        # 按实验组和训练器类型分组统计
        print("\n📈 分组统计结果:")
        if 'group' in df.columns and 'trainer_type' in df.columns and 'final_train_profit_rate' in df.columns:
            grouped = df[df['repeat'] == 'average'].groupby(['group', 'trainer_type'])['final_train_profit_rate'].mean()
            print(grouped)
    else:
        print("❌ 没有找到实验结果")


def check_training_curves():
    """检查训练曲线数据"""
    import matplotlib.pyplot as plt

    results_dir = "experiment_results"
    curves_found = []

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('training_curves_0.csv'):
                curve_path = os.path.join(root, file)
                curves_found.append(curve_path)

                # 加载并显示一个样本曲线
                try:
                    df = pd.read_csv(curve_path)
                    print(f"\n📊 训练曲线样本 ({curve_path}):")
                    print(f"   数据点数量: {len(df)}")
                    print(f"   指标: {', '.join(df.columns.tolist())}")

                    # 显示最后几个数据点
                    if len(df) > 5:
                        print(f"   最后5个episode的奖励: {df['reward'].tail(5).values}")
                        if 'profit_rate' in df.columns:
                            print(f"   最后5个episode的利润率: {df['profit_rate'].tail(5).values}")
                except:
                    pass

    print(f"\n🔍 找到 {len(curves_found)} 个训练曲线文件")
    return curves_found


if __name__ == "__main__":
    print("🚀 开始导出实验结果...")

    # 检查训练曲线
    curves = check_training_curves()

    # 导出结果
    export_experiment_results()

    # 如果存在汇总文件，也显示
    summary_files = [
        "experiment_results/消融实验/experiment_summary.csv",
        "all_experiments_summary.csv"
    ]

    for file in summary_files:
        if os.path.exists(file):
            print(f"\n📋 现有汇总文件: {file}")
            df = pd.read_csv(file)
            print(f"   记录数: {len(df)}")
            print(f"   列名: {df.columns.tolist()}")

            # 显示前几个实验
            if len(df) > 0:
                print("   前几个实验:")
                for i in range(min(3, len(df))):
                    exp_id = df.iloc[i]['experiment_id']
                    reward = df.iloc[i]['avg_test_reward']
                    print(f"     {exp_id}: 奖励={reward:.2f}")