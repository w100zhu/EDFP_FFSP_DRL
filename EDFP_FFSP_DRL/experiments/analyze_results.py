# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def analyze_experiment_results():
    """分析实验结果并生成报告"""

    # 读取导出结果
    if os.path.exists("all_experiments_complete.csv"):
        df = pd.read_csv("all_experiments_complete.csv")
    else:
        # 尝试从现有文件读取
        df = pd.read_csv("all_experiments_summary.csv")

    print(f"📊 总实验记录数: {len(df)}")
    print(f"📋 可用列: {df.columns.tolist()}")

    # 过滤出平均值行
    if 'repeat' in df.columns:
        avg_df = df[df['repeat'] == 'average'].copy()
    else:
        avg_df = df.copy()

    print(f"📈 平均结果记录数: {len(avg_df)}")

    # 创建分析报告目录
    report_dir = "experiment_analysis_report"
    os.makedirs(report_dir, exist_ok=True)

    # 1. 算法性能对比
    if 'trainer_type' in avg_df.columns and 'final_train_profit_rate' in avg_df.columns:
        plt.figure(figsize=(12, 6))

        # 按训练器类型分组
        trainer_performance = avg_df.groupby('trainer_type').agg({
            'final_train_profit_rate': 'mean',
            'final_train_profit': 'mean',
            'final_train_makespan': 'mean',
            'avg_test_reward': 'mean'
        }).reset_index()

        # 绘制柱状图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = [
            ('final_train_profit_rate', 'Profit Rate (Train)'),
            ('final_train_profit', 'Total Profit (Train)'),
            ('final_train_makespan', 'Makespan (Train)'),
            ('avg_test_reward', 'Test Reward')
        ]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            trainer_performance.sort_values(metric, ascending=False).plot(
                x='trainer_type', y=metric, kind='bar', ax=ax,
                color='skyblue', edgecolor='black'
            )
            ax.set_title(title)
            ax.set_xlabel('Trainer Type')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'algorithm_performance.png'), dpi=300, bbox_inches='tight')

        # 保存性能表格
        trainer_performance.to_csv(os.path.join(report_dir, 'algorithm_performance.csv'), index=False)

    # 2. 环境规模影响分析
    if 'num_jobs' in avg_df.columns:
        plt.figure(figsize=(14, 10))

        # 按工件数量分组
        job_groups = avg_df.groupby('num_jobs').agg({
            'final_train_profit_rate': ['mean', 'std'],
            'final_train_profit': ['mean', 'std'],
            'final_train_makespan': ['mean', 'std']
        }).reset_index()

        # 绘制折线图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        job_groups.columns = ['_'.join(col).strip('_') for col in job_groups.columns]

        # 利润率 vs 工件数量
        axes[0, 0].errorbar(
            job_groups['num_jobs'],
            job_groups['final_train_profit_rate_mean'],
            yerr=job_groups['final_train_profit_rate_std'],
            marker='o', capsize=5, linewidth=2
        )
        axes[0, 0].set_title('Profit Rate vs Number of Jobs')
        axes[0, 0].set_xlabel('Number of Jobs')
        axes[0, 0].set_ylabel('Profit Rate')
        axes[0, 0].grid(True, alpha=0.3)

        # 总利润 vs 工件数量
        axes[0, 1].errorbar(
            job_groups['num_jobs'],
            job_groups['final_train_profit_mean'],
            yerr=job_groups['final_train_profit_std'],
            marker='s', capsize=5, linewidth=2, color='green'
        )
        axes[0, 1].set_title('Total Profit vs Number of Jobs')
        axes[0, 1].set_xlabel('Number of Jobs')
        axes[0, 1].set_ylabel('Total Profit')
        axes[0, 1].grid(True, alpha=0.3)

        # 完工时长 vs 工件数量
        axes[1, 0].errorbar(
            job_groups['num_jobs'],
            job_groups['final_train_makespan_mean'],
            yerr=job_groups['final_train_makespan_std'],
            marker='^', capsize=5, linewidth=2, color='red'
        )
        axes[1, 0].set_title('Makespan vs Number of Jobs')
        axes[1, 0].set_xlabel('Number of Jobs')
        axes[1, 0].set_ylabel('Makespan')
        axes[1, 0].grid(True, alpha=0.3)

        # 效率（利润/时间/工件）vs 工件数量
        if 'final_train_profit_mean' in job_groups.columns and 'final_train_makespan_mean' in job_groups.columns:
            efficiency = job_groups['final_train_profit_mean'] / (
                        job_groups['final_train_makespan_mean'] * job_groups['num_jobs'])
            axes[1, 1].plot(job_groups['num_jobs'], efficiency, marker='d', linewidth=2, color='purple')
            axes[1, 1].set_title('Efficiency (Profit/Time/Job) vs Number of Jobs')
            axes[1, 1].set_xlabel('Number of Jobs')
            axes[1, 1].set_ylabel('Efficiency')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'scale_impact.png'), dpi=300, bbox_inches='tight')

        # 保存规模影响表格
        job_groups.to_csv(os.path.join(report_dir, 'scale_impact.csv'), index=False)

    # 3. 训练器详细对比报告
    if 'trainer_type' in avg_df.columns:
        detailed_report = []

        for trainer in avg_df['trainer_type'].unique():
            trainer_df = avg_df[avg_df['trainer_type'] == trainer]

            if len(trainer_df) > 0:
                report = {
                    'trainer_type': trainer,
                    'num_experiments': len(trainer_df),
                    'avg_profit_rate': trainer_df['final_train_profit_rate'].mean(),
                    'std_profit_rate': trainer_df['final_train_profit_rate'].std(),
                    'avg_total_profit': trainer_df['final_train_profit'].mean(),
                    'avg_makespan': trainer_df['final_train_makespan'].mean(),
                    'avg_test_reward': trainer_df['avg_test_reward'].mean(),
                    'best_experiment': trainer_df.loc[trainer_df['final_train_profit_rate'].idxmax(), 'experiment_id']
                    if 'experiment_id' in trainer_df.columns else 'N/A'
                }
                detailed_report.append(report)

        # 创建详细报告DataFrame
        detailed_df = pd.DataFrame(detailed_report)
        detailed_df.to_csv(os.path.join(report_dir, 'trainer_detailed_report.csv'), index=False)

        # 打印报告摘要
        print("\n" + "=" * 60)
        print("🏆 训练器性能排行榜")
        print("=" * 60)

        detailed_df = detailed_df.sort_values('avg_profit_rate', ascending=False)
        for idx, row in detailed_df.iterrows():
            print(f"{idx + 1}. {row['trainer_type']}:")
            print(f"   平均利润率: {row['avg_profit_rate']:.4f} (±{row['std_profit_rate']:.4f})")
            print(f"   平均总利润: {row['avg_total_profit']:.2f}")
            print(f"   平均完工时长: {row['avg_makespan']:.2f}")
            print(f"   最佳实验: {row['best_experiment']}")
            print()

    # 4. 生成HTML报告
    generate_html_report(report_dir, avg_df)

    print(f"\n✅ 分析报告已生成到目录: {report_dir}")
    print(f"📊 包含文件:")
    for file in os.listdir(report_dir):
        print(f"   - {file}")


def generate_html_report(report_dir, df):
    """生成HTML格式的报告"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>实验分析报告</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            .metric { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .highlight { background: #e8f4f8; padding: 10px; border-left: 4px solid #3498db; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>🧪 DRL拆解车间调度实验分析报告</h1>

        <div class="highlight">
            <p><strong>报告生成时间:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p><strong>总实验数:</strong> """ + str(len(df)) + """</p>
        </div>

        <h2>📊 关键发现</h2>
        <div class="metric">
    """

    # 添加关键指标
    if 'final_train_profit_rate' in df.columns:
        best_exp = df.loc[df['final_train_profit_rate'].idxmax()]
        html_content += f"""
            <p><strong>🏆 最佳实验:</strong> {best_exp.get('experiment_id', 'N/A')}</p>
            <p><strong>🎯 最佳利润率:</strong> {best_exp['final_train_profit_rate']:.4f}</p>
            <p><strong>💰 对应利润:</strong> {best_exp.get('final_train_profit', 0):.2f}</p>
            <p><strong>⏱️ 对应完工时长:</strong> {best_exp.get('final_train_makespan', 0):.2f}</p>
        """

    html_content += """
        </div>

        <h2>📈 性能对比</h2>
        <p>算法性能对比图:</p>
        <img src="algorithm_performance.png" alt="算法性能对比">

        <h2>📊 环境规模影响</h2>
        <p>工件数量对性能的影响:</p>
        <img src="scale_impact.png" alt="规模影响分析">

        <h2>📋 详细数据</h2>
        <p>详细数据可下载:</p>
        <ul>
            <li><a href="algorithm_performance.csv">算法性能数据</a></li>
            <li><a href="scale_impact.csv">规模影响数据</a></li>
            <li><a href="trainer_detailed_report.csv">训练器详细报告</a></li>
        </ul>

        <h2>🔍 分析说明</h2>
        <div class="metric">
            <p><strong>利润率 (Profit Rate):</strong> 利润与完工时长的比值，反映单位时间的盈利能力</p>
            <p><strong>总利润 (Total Profit):</strong> 单次调度任务获得的总利润</p>
            <p><strong>完工时长 (Makespan):</strong> 完成所有工件所需的总时间</p>
            <p><strong>效率 (Efficiency):</strong> 每工件每时间单位的利润</p>
        </div>
    </body>
    </html>
    """

    # 保存HTML报告
    with open(os.path.join(report_dir, 'analysis_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    print("🔬 开始分析实验结果...")
    analyze_experiment_results()