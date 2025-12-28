import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple


class GanttChart:
    """
    甘特图可视化工具
    用于可视化调度结果
    """

    def __init__(self, schedule: List[List[Tuple]], num_machines: int, num_jobs: int):
        self.schedule = schedule
        self.num_machines = num_machines
        self.num_jobs = num_jobs

    def plot(self, filename: str = None, show: bool = True) -> None:
        """绘制甘特图"""
        fig, ax = plt.subplots(figsize=(15, 8))

        # 为每个作业分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_jobs))

        # 绘制每个机器的调度
        for machine_id, machine_schedule in enumerate(self.schedule):
            for job_info in machine_schedule:
                if len(job_info) >= 3:
                    job_id, start_time, end_time = job_info[0], job_info[1], job_info[2]
                    stage = job_info[3] if len(job_info) > 3 else 0

                    duration = end_time - start_time

                    # 绘制矩形
                    rect = patches.Rectangle(
                        (start_time, machine_id - 0.4), duration, 0.8,
                        linewidth=1, edgecolor='black',
                        facecolor=colors[job_id % len(colors)],
                        alpha=0.7
                    )
                    ax.add_patch(rect)

                    # 添加文本标注
                    ax.text(start_time + duration / 2, machine_id,
                            f'J{job_id}-S{stage}',
                            ha='center', va='center', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))

        # 设置坐标轴
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('机器', fontsize=12)
        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([f'机器 {i}' for i in range(self.num_machines)])

        # 设置网格
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # 设置标题
        makespan = max([job_info[2] for machine_schedule in self.schedule for job_info in machine_schedule])
        ax.set_title(f'调度甘特图 (最大完成时间: {makespan:.2f})', fontsize=14)

        # 设置图例
        legend_elements = [patches.Patch(facecolor=colors[i],
                                         label=f'作业 {i}')
                           for i in range(min(10, self.num_jobs))]  # 只显示前10个作业的图例
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"甘特图已保存到: {filename}")

        if show:
            plt.show()

        return fig, ax

    def save(self, filename: str) -> None:
        """保存甘特图"""
        self.plot(filename=filename, show=False)