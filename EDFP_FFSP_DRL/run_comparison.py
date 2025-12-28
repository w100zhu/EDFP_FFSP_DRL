# EDFP_FFSP_DRL/run_comparison.py (升级版：支持利润率评估 + 严格配置对齐)
import sys
import os
import glob
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import re

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from EDFP_FFSP_DRL.config import Config
from EDFP_FFSP_DRL.environment.dffsp_env import DFFSPEnvironment
from EDFP_FFSP_DRL.heuristics import HeuristicSolver
from EDFP_FFSP_DRL.agents.job_agent import JobAgent
from EDFP_FFSP_DRL.agents.machine_agent import MachineAgent
from EDFP_FFSP_DRL.training.integrated_trainer import IntegratedTrainer  # 用于加载 Attention 结构

# ================= 1. 实验配置区域 =================

GROUP_NAME = "全面实验"
TARGET_EPISODE = 200  # 或根据实际 Checkpoint 选择

# 【核心】确保与训练参数一致
TRAIN_PARAMS = {
    'NUM_STAGES': 5,
    'HIDDEN_DIM': 64,
    'MAX_STEPS': 500
}

SCENARIOS = [
    {'NUM_JOBS': 5, 'NUM_MACHINES': 8, 'NAME': 'Small (5x8)'},
    {'NUM_JOBS': 10, 'NUM_MACHINES': 8, 'NAME': 'Medium (10x8)'},
    {'NUM_JOBS': 20, 'NUM_MACHINES': 8, 'NAME': 'Large (20x8)'},
    # {'NUM_JOBS': 30, 'NUM_MACHINES': 8, 'NAME': 'Extra Large (30x8)'},
]

TEST_SEEDS = [42, 100, 2023, 7, 99]  # 多测几个种子取平均

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(BASE_PATH, 'experiments', 'checkpoints', GROUP_NAME)

print(f"📂 脚本搜索的根目录: {CHECKPOINT_ROOT}")


# ================= 2. 辅助函数 =================

def find_best_checkpoint(folder_path, prefix, target_ep):
    if not os.path.exists(folder_path): return None, "目录不存在"

    # 优先找 best_checkpoint.pth
    best_pth = os.path.join(folder_path, "best_checkpoint.pth")
    if os.path.exists(best_pth):
        return best_pth, "🏆 找到最佳模型 (best_checkpoint.pth)"

    # 其次找 target_ep
    target_pth = os.path.join(folder_path, f"{prefix}_{target_ep}.pth")
    if os.path.exists(target_pth):
        return target_pth, f"🎯 找到指定轮次 ({target_ep})"

    # 最后找最新的
    files = glob.glob(os.path.join(folder_path, f"{prefix}*.pth"))
    if not files: return None, "无模型文件"
    return max(files, key=os.path.getmtime), "⚠️ 使用最新修改的文件"


def run_drl_agent(env, model_info, config):
    try:
        # 重新构建 Trainer 以正确加载 Attention 网络结构
        # 因为 Agent 本身不包含 Attention 层，它们在 Trainer 里
        job_agent = JobAgent(config)
        machine_agent = MachineAgent(config)

        # 临时创建一个 Trainer 实例来加载权重
        trainer = IntegratedTrainer(config, job_agent, machine_agent, env)
        trainer.load_checkpoint(model_info['path'])

        # 将 Trainer 里的 Agent 提取出来用于推理
        # 注意：推理时需要手动处理 Attention，最简单的方法是直接调用 Trainer 的 collect_rollout 逻辑
        # 或者我们把 Trainer 的 _process_state_with_attention 逻辑剥离出来。
        # 这里为了方便，我们直接复用 Trainer 的 _process_state_with_attention

        process_attn_func = trainer._process_state_with_attention

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return None

    state = env.reset()
    done = False
    start = time.time()
    steps = 0

    with torch.no_grad():
        while not done and steps < config.MAX_STEPS * 2:
            # 1. Attention 处理
            processed_state = process_attn_func(state)

            # 2. Agent 决策 (Greedy)
            if hasattr(job_agent, 'actor'):
                job_agent.actor.eval()
                machine_agent.actor.eval()

                # 贪婪选择：取概率最大的动作
                j_logits = job_agent.actor(processed_state)
                m_logits = machine_agent.actor(processed_state)

                # Masking
                j_mask = job_agent._get_action_mask(env).to(config.DEVICE)
                m_mask = machine_agent._get_action_mask(env).to(config.DEVICE)

                j_logits = j_logits + (j_mask + 1e-45).log()
                m_logits = m_logits + (m_mask + 1e-45).log()

                j_act = torch.argmax(j_logits, dim=-1).item()
                m_act = torch.argmax(m_logits, dim=-1).item()
            else:
                j_act = job_agent.select_action(processed_state)
                m_act = machine_agent.select_action(processed_state)

            state, _, done, _ = env.step(j_act, m_act)
            steps += 1

    makespan = env._calculate_makespan()
    profit = env.total_profit
    # 【新增】计算利润率
    profit_rate = profit / max(1.0, makespan)

    return {
        'Profit': profit,
        'Makespan': makespan,
        'Profit_Rate': profit_rate,
        'Time': time.time() - start
    }


# ================= 3. 主程序 =================

def run_comparison():
    all_results = []

    # 查找模型
    TRAINED_MODELS = {}
    for n_jobs in [s['NUM_JOBS'] for s in SCENARIOS]:
        folder = os.path.join(CHECKPOINT_ROOT, f'J{n_jobs}_M8_P0.5_Integrated')
        path, msg = find_best_checkpoint(folder, "checkpoint", TARGET_EPISODE)
        if path:
            TRAINED_MODELS[n_jobs] = {'path': path, 'msg': msg}
            print(f"✅ [J{n_jobs}] {msg}")
        else:
            print(f"❌ [J{n_jobs}] 未找到模型")

    for scenario in SCENARIOS:
        print(f"\n📊 测试场景: {scenario['NAME']}")

        # 配置初始化
        cfg = Config()
        cfg.NUM_JOBS = scenario['NUM_JOBS']
        cfg.NUM_MACHINES = scenario['NUM_MACHINES']
        cfg.NUM_STAGES = TRAIN_PARAMS['NUM_STAGES']
        cfg.HIDDEN_DIM = TRAIN_PARAMS['HIDDEN_DIM']
        cfg.MAX_STEPS = TRAIN_PARAMS['MAX_STEPS']
        cfg.DEVICE = 'cpu'  # 推理用 CPU 即可

        # 探测维度
        temp_env = DFFSPEnvironment(cfg.NUM_JOBS, cfg.NUM_STAGES, cfg.NUM_MACHINES, cfg)
        s = temp_env.reset()
        cfg.STATE_DIM = len(s) if not isinstance(s, torch.Tensor) else s.shape[0]
        cfg.JOB_STATE_DIM = cfg.STATE_DIM
        cfg.MACHINE_STATE_DIM = cfg.STATE_DIM
        cfg.JOB_ACTION_DIM = cfg.NUM_JOBS
        cfg.MACHINE_ACTION_DIM = cfg.NUM_MACHINES

        for seed in TEST_SEEDS:
            # 1. 运行启发式算法
            np.random.seed(seed)
            env_h = DFFSPEnvironment(cfg.NUM_JOBS, cfg.NUM_STAGES, cfg.NUM_MACHINES, cfg)
            solver = HeuristicSolver(env_h)

            for r in ['spt', 'random']:  # 简化对比
                res = solver.solve(r)
                # 计算 Rate
                p_rate = res['final_profit'] / max(1.0, res['final_makespan'])
                all_results.append({
                    'Scenario': scenario['NAME'],
                    'Algorithm': f'Heuristic-{r.upper()}',
                    'Profit': res['final_profit'],
                    'Makespan': res['final_makespan'],
                    'Profit_Rate': p_rate,
                    'Seed': seed
                })

            # 2. 运行 DRL (Efficiency Agent)
            if cfg.NUM_JOBS in TRAINED_MODELS:
                np.random.seed(seed)
                torch.manual_seed(seed)
                env_drl = DFFSPEnvironment(cfg.NUM_JOBS, cfg.NUM_STAGES, cfg.NUM_MACHINES, cfg)

                res = run_drl_agent(env_drl, TRAINED_MODELS[cfg.NUM_JOBS], cfg)
                if res:
                    all_results.append({
                        'Scenario': scenario['NAME'],
                        'Algorithm': 'Efficiency-Agent (DRL)',
                        'Profit': res['Profit'],
                        'Makespan': res['Makespan'],
                        'Profit_Rate': res['Profit_Rate'],
                        'Seed': seed
                    })

    # 保存与绘图
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv('final_efficiency_comparison.csv', index=False)
        print("\n💾 结果已保存: final_efficiency_comparison.csv")

        # 绘制 Profit Rate 对比图
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Scenario', y='Profit_Rate', hue='Algorithm', palette='viridis')
        plt.title('Efficiency Comparison: Profit Rate (Higher is Better)')
        plt.tight_layout()
        plt.savefig('efficiency_comparison.png')
        print("📊 图表已保存: efficiency_comparison.png")


if __name__ == '__main__':
    run_comparison()