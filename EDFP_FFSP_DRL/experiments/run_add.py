# experiments/run_experiments.py
import os
import sys
import json
import time
import random
import math
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import glob
import re

warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from EDFP_FFSP_DRL.experiments.experiment_config_generator import ExperimentConfigGenerator
from EDFP_FFSP_DRL.experiments.batch_experiment_runner import BatchExperimentRunner
from EDFP_FFSP_DRL.config import Config
from EDFP_FFSP_DRL.environment.dffsp_env import DFFSPEnvironment
from EDFP_FFSP_DRL.heuristics import HeuristicSolver
from EDFP_FFSP_DRL.agents.job_agent import JobAgent
from EDFP_FFSP_DRL.agents.machine_agent import MachineAgent
from EDFP_FFSP_DRL.training.integrated_trainer import IntegratedTrainer

# ✨ 导入多目标元启发算法 NSGA-II
try:
    from EDFP_FFSP_DRL.nsga2_solver import NSGA2Solver
except ImportError:
    try:
        from nsga2_solver import NSGA2Solver
    except ImportError:
        NSGA2Solver = None
        print("⚠️ 无法导入 NSGA2Solver，跳过 NSGA-II 对比")


# =====================================================================
# 辅助函数：加载与定位模型
# =====================================================================

def find_integrated_checkpoint(folder_path, prefix, target_ep):
    if not os.path.exists(folder_path): return None
    target_pth = os.path.join(folder_path, f"{prefix}_{target_ep}.pth")
    if os.path.exists(target_pth): return target_pth
    files = glob.glob(os.path.join(folder_path, f"{prefix}*.pth"))
    if files: return max(files, key=os.path.getmtime)
    best_pth = os.path.join(folder_path, "best_checkpoint.pth")
    if os.path.exists(best_pth): return best_pth
    return None


def find_base_ppo_checkpoints(folder_path, target_ep=None):
    if not os.path.exists(folder_path): return None, None

    if target_ep is not None:
        j_path = os.path.join(folder_path, f"job_agent_base_{target_ep}.pth")
        m_path = os.path.join(folder_path, f"machine_agent_base_{target_ep}.pth")
        if os.path.exists(j_path) and os.path.exists(m_path): return j_path, m_path

    j_files = glob.glob(os.path.join(folder_path, "job_agent_base_*.pth"))
    m_files = glob.glob(os.path.join(folder_path, "machine_agent_base_*.pth"))
    if not j_files or not m_files: return None, None

    def extract_ep(fp):
        m = re.search(r'_(\d+)\.pth$', os.path.basename(fp))
        return int(m.group(1)) if m else 0

    j_files = sorted(j_files, key=extract_ep, reverse=True)
    for jf in j_files:
        ep = extract_ep(jf)
        mf = os.path.join(folder_path, f"machine_agent_base_{ep}.pth")
        if os.path.exists(mf): return jf, mf
    return j_files[0], m_files[0]


# =====================================================================
# 算法执行器：包含时间统计与传感器噪声注入
# =====================================================================
def run_heuristic_agent(env_class, config, rule, seed):
    np.random.seed(seed);
    random.seed(seed)
    env = env_class(config.NUM_JOBS, config.NUM_STAGES, config.NUM_MACHINES, config)
    solver = HeuristicSolver(env)
    start_time = time.time()
    res = solver.solve(rule)
    elapsed_time = time.time() - start_time
    return {'Profit': res['final_profit'], 'Makespan': res['final_makespan'],
            'Profit_Rate': res['final_profit'] / max(1.0, res['final_makespan']), 'Time': elapsed_time}


def run_sa_spt_agent(env_class, config, seed):
    np.random.seed(seed);
    random.seed(seed)
    start_time = time.time()
    env_spt = env_class(config.NUM_JOBS, config.NUM_STAGES, config.NUM_MACHINES, config)
    spt_res = HeuristicSolver(env_spt).solve('spt')
    best_prt = spt_res['final_profit'] / max(1.0, spt_res['final_makespan'])
    best_p, best_m = spt_res['final_profit'], spt_res['final_makespan']

    T, T_min, alpha, iter_per_T = 100.0, 0.5, 0.95, 20
    curr_prt = best_prt
    while T > T_min:
        for _ in range(iter_per_T):
            env_new = env_class(config.NUM_JOBS, config.NUM_STAGES, config.NUM_MACHINES, config)
            rule = random.choice(['spt', 'sa-spt', 'random'])
            new_res = HeuristicSolver(env_new).solve(rule)
            new_prt = new_res['final_profit'] / max(1.0, new_res['final_makespan'])
            delta = new_prt - curr_prt
            if delta > 0 or math.exp(delta / T) > random.random():
                curr_prt = new_prt
                if curr_prt > best_prt: best_prt, best_p, best_m = curr_prt, new_res['final_profit'], new_res[
                    'final_makespan']
        T *= alpha
    elapsed_time = time.time() - start_time
    return {'Profit': best_p, 'Makespan': best_m, 'Profit_Rate': best_prt, 'Time': elapsed_time}


def run_nsga2_agent(env_class, config, seed):
    if NSGA2Solver is None: return None
    np.random.seed(seed);
    random.seed(seed);
    torch.manual_seed(seed)
    try:
        solver = NSGA2Solver(env_class, config, pop_size=40, n_gen=40)
        start_time = time.time()
        pareto_front, _ = solver.solve()
        elapsed_time = time.time() - start_time
        if not pareto_front: return None
        best_res = max(pareto_front, key=lambda x: x['profit'] / max(1.0, x['makespan']))
        return {'Profit': best_res['profit'], 'Makespan': best_res['makespan'],
                'Profit_Rate': best_res['profit'] / max(1.0, best_res['makespan']), 'Time': elapsed_time}
    except Exception as e:
        return None


def run_drl_inference(env_class, config, model_path, seed, use_attention=True, noise_std=0.0):
    """🚀 DRL 神经网络推理 (支持高斯噪声注入，用于测试鲁棒性)"""
    try:
        np.random.seed(seed);
        torch.manual_seed(seed)
        config.DEVICE = 'cpu'
        ckpt = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        saved_input_dim = config.STATE_DIM
        if 'config_info' in ckpt and ckpt['config_info'].get('state_dim'):
            saved_input_dim = ckpt['config_info']['state_dim']
        elif 'job_actor_state' in ckpt:
            for key in ckpt['job_actor_state']:
                if 'weight' in key and len(ckpt['job_actor_state'][key].shape) == 2:
                    saved_input_dim = ckpt['job_actor_state'][key].shape[1];
                    break

        config.USE_ATTENTION = use_attention
        original_state_dim = config.STATE_DIM
        config.STATE_DIM = config.JOB_STATE_DIM = config.MACHINE_STATE_DIM = saved_input_dim

        job_agent, machine_agent = JobAgent(config), MachineAgent(config)
        loaded = False
        for j_key, m_key in [('job_actor_state', 'machine_actor_state'), ('actor_state_dict', 'actor_state_dict'),
                             ('state_dict', 'state_dict')]:
            if j_key in ckpt and m_key in ckpt:
                try:
                    job_agent.actor.load_state_dict(ckpt[j_key]); machine_agent.actor.load_state_dict(
                        ckpt[m_key]); loaded = True; break
                except:
                    continue
        if not loaded:
            job_agent.load_state_dict(ckpt.get('job_agent', ckpt));
            machine_agent.load_state_dict(ckpt.get('machine_agent', ckpt))

        process_func = lambda x: x
        if use_attention:
            config.STATE_DIM = saved_input_dim
            trainer = IntegratedTrainer(config, job_agent, machine_agent, env_class(1, 1, 1, config))
            for layer_name in ['job_in_proj', 'machine_in_proj', 'job_self_attn', 'machine_self_attn',
                               'cross_attention', 'job_out_proj']:
                if layer_name in ckpt and hasattr(trainer, layer_name): getattr(trainer, layer_name).load_state_dict(
                    ckpt[layer_name])
            process_func = getattr(trainer, '_process_state_with_attention', lambda x: x)
        config.STATE_DIM = original_state_dim

        env = env_class(config.NUM_JOBS, config.NUM_STAGES, config.NUM_MACHINES, config)

        def adapt_and_add_noise(state, target_dim, std):
            st = torch.FloatTensor(state).to(config.DEVICE) if not isinstance(state, torch.Tensor) else state.to(
                config.DEVICE)
            if len(st.shape) == 1: st = st.unsqueeze(0)
            curr_dim = st.shape[-1]
            if curr_dim > target_dim:
                st = st[..., :target_dim]
            elif curr_dim < target_dim:
                st = torch.cat([st, torch.zeros(st.shape[0], target_dim - curr_dim).to(config.DEVICE)], dim=-1)

            # ✨ 核心：注入传感器测量误差 (高斯噪声)
            if std > 0:
                noise = torch.randn_like(st) * std
                st = torch.clamp(st + noise, -1.0, 1.0)  # 防止噪声导致状态值溢出

            return st

        state = env.reset()
        done, steps, total_reward = False, 0, 0
        job_agent.actor.eval();
        machine_agent.actor.eval()

        with torch.no_grad():
            dummy_state = torch.zeros(1, saved_input_dim).to(config.DEVICE)
            _ = job_agent.actor(process_func(dummy_state))
            _ = machine_agent.actor(process_func(dummy_state))

        start_time = time.time()
        with torch.no_grad():
            while not done and steps < config.MAX_STEPS * 2:
                avail_j, avail_m = env.get_available_actions()
                if not avail_j or not avail_m:
                    if hasattr(env, '_find_next_decision_point'):
                        env.current_time = env._find_next_decision_point()
                    else:
                        env.current_time += 1.0
                    if len(env.completed_jobs) == env.num_jobs: done = True
                    steps += 1;
                    continue

                # 注入噪声并处理状态
                st = adapt_and_add_noise(state, saved_input_dim, noise_std)
                processed_state = process_func(st)

                # Job Masking
                j_out = job_agent.actor(processed_state)
                j_logits = j_out.logits if hasattr(j_out, 'logits') else j_out
                if isinstance(j_logits, tuple): j_logits = j_logits[0]
                j_logits = j_logits.squeeze()
                if j_logits.dim() == 0: j_logits = j_logits.unsqueeze(0)
                j_mask = torch.full_like(j_logits, -float('inf'))
                valid_j = [j for j in avail_j if j < config.JOB_ACTION_DIM]
                if valid_j: j_mask[valid_j] = 0.0
                j_act = torch.argmax(j_logits + j_mask).item()

                # Machine Masking
                m_out = machine_agent.actor(processed_state)
                m_logits = m_out.logits if hasattr(m_out, 'logits') else m_out
                if isinstance(m_logits, tuple): m_logits = m_logits[0]
                m_logits = m_logits.squeeze()
                if m_logits.dim() == 0: m_logits = m_logits.unsqueeze(0)
                m_mask = torch.full_like(m_logits, -float('inf'))
                valid_m = [m for m in avail_m if m < config.MACHINE_ACTION_DIM]
                if valid_m: m_mask[valid_m] = 0.0
                m_act = torch.argmax(m_logits + m_mask).item()

                if j_act not in avail_j: j_act = random.choice(avail_j)
                if m_act not in avail_m: m_act = random.choice(avail_m)

                state, reward, done, _ = env.step(j_act, m_act)
                total_reward += reward;
                steps += 1

        elapsed_time = time.time() - start_time
        makespan = env._calculate_makespan()
        return {'Profit': env.total_profit, 'Makespan': makespan, 'Profit_Rate': env.total_profit / max(1.0, makespan),
                'Time': elapsed_time}
    except Exception as e:
        print(f"DRL Inference Error: {e}")
        return None


def run_drl_base_inference(env_class, config, job_path, machine_path, seed):
    """🚀 运行 BAPO 的推理 (无注意力机制)"""
    try:
        np.random.seed(seed);
        torch.manual_seed(seed)
        config.DEVICE = 'cpu'
        job_ckpt = torch.load(job_path, map_location=config.DEVICE, weights_only=False)
        machine_ckpt = torch.load(machine_path, map_location=config.DEVICE, weights_only=False)

        saved_input_dim = config.STATE_DIM
        if 'actor' in job_ckpt:
            for k in job_ckpt['actor']:
                if 'weight' in k and len(job_ckpt['actor'][k].shape) == 2:
                    saved_input_dim = job_ckpt['actor'][k].shape[1];
                    break
        orig_dim = config.STATE_DIM
        config.STATE_DIM = config.JOB_STATE_DIM = config.MACHINE_STATE_DIM = saved_input_dim
        job_agent, machine_agent = JobAgent(config), MachineAgent(config)
        try:
            job_agent.actor.load_state_dict(job_ckpt.get('actor', job_ckpt.get('actor_state_dict', job_ckpt)))
            machine_agent.actor.load_state_dict(
                machine_ckpt.get('actor', machine_ckpt.get('actor_state_dict', machine_ckpt)))
        except:
            return None
        config.STATE_DIM = config.JOB_STATE_DIM = config.MACHINE_STATE_DIM = orig_dim

        env = env_class(config.NUM_JOBS, config.NUM_STAGES, config.NUM_MACHINES, config)
        state = env.reset()
        done, steps = False, 0
        job_agent.actor.eval();
        machine_agent.actor.eval()

        with torch.no_grad():
            dummy_state = torch.zeros(1, saved_input_dim).to(config.DEVICE)
            _ = job_agent.actor(dummy_state)
            _ = machine_agent.actor(dummy_state)

        start_time = time.time()
        with torch.no_grad():
            while not done and steps < config.MAX_STEPS * 2:
                avail_j, avail_m = env.get_available_actions()
                if not avail_j or not avail_m:
                    if hasattr(env, '_find_next_decision_point'):
                        env.current_time = env._find_next_decision_point()
                    else:
                        env.current_time += 1.0
                    if len(env.completed_jobs) == env.num_jobs: done = True
                    steps += 1;
                    continue

                st = torch.FloatTensor(state).to(config.DEVICE).unsqueeze(0) if not isinstance(state,
                                                                                               torch.Tensor) else (
                    state.unsqueeze(0) if len(state.shape) == 1 else state)
                curr_dim = st.shape[-1]
                if curr_dim > saved_input_dim:
                    st = st[..., :saved_input_dim]
                elif curr_dim < saved_input_dim:
                    st = torch.cat([st, torch.zeros(st.shape[0], saved_input_dim - curr_dim).to(config.DEVICE)], dim=-1)

                j_out = job_agent.actor(st)
                j_logits = j_out.logits if hasattr(j_out, 'logits') else j_out
                if isinstance(j_logits, tuple): j_logits = j_logits[0]
                j_logits = j_logits.squeeze()
                if j_logits.dim() == 0: j_logits = j_logits.unsqueeze(0)
                j_mask = torch.full_like(j_logits, -float('inf'))
                valid_j = [j for j in avail_j if j < config.JOB_ACTION_DIM]
                if valid_j: j_mask[valid_j] = 0.0
                j_act = torch.argmax(j_logits + j_mask).item()

                m_out = machine_agent.actor(st)
                m_logits = m_out.logits if hasattr(m_out, 'logits') else m_out
                if isinstance(m_logits, tuple): m_logits = m_logits[0]
                m_logits = m_logits.squeeze()
                if m_logits.dim() == 0: m_logits = m_logits.unsqueeze(0)
                m_mask = torch.full_like(m_logits, -float('inf'))
                valid_m = [m for m in avail_m if m < config.MACHINE_ACTION_DIM]
                if valid_m: m_mask[valid_m] = 0.0
                m_act = torch.argmax(m_logits + m_mask).item()

                if j_act not in avail_j: j_act = random.choice(avail_j)
                if m_act not in avail_m: m_act = random.choice(avail_m)

                state, _, done, _ = env.step(j_act, m_act)
                steps += 1

        elapsed_time = time.time() - start_time
        makespan = env._calculate_makespan()
        return {'Profit': env.total_profit, 'Makespan': makespan, 'Profit_Rate': env.total_profit / max(1.0, makespan),
                'Time': elapsed_time}
    except Exception as e:
        return None


# =====================================================================
# 核心任务 1：综合算法大对比 (生成 Table 3)
# =====================================================================
def run_algorithm_comparison(group_name):
    print(f"\n⏳ 开始运行带【执行时间】的算法深度对比: {group_name}...")

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    config_root = os.path.join(BASE_PATH, "experiment_configs", group_name)
    checkpoint_root = os.path.join(BASE_PATH, "checkpoints", group_name)

    SCENARIOS = [
        {'NUM_JOBS': 5, 'NUM_MACHINES': 4, 'NAME': '5x4'},
        {'NUM_JOBS': 10, 'NUM_MACHINES': 4, 'NAME': '10x4'},
        {'NUM_JOBS': 20, 'NUM_MACHINES': 4, 'NAME': '20x4'},
        {'NUM_JOBS': 30, 'NUM_MACHINES': 4, 'NAME': '30x4'},
        {'NUM_JOBS': 5, 'NUM_MACHINES': 8, 'NAME': '5x8'},
        {'NUM_JOBS': 10, 'NUM_MACHINES': 8, 'NAME': '10x8'},
        {'NUM_JOBS': 20, 'NUM_MACHINES': 8, 'NAME': '20x8'},
        {'NUM_JOBS': 30, 'NUM_MACHINES': 8, 'NAME': '30x8'},
        {'NUM_JOBS': 5, 'NUM_MACHINES': 10, 'NAME': '5x10'},
        {'NUM_JOBS': 10, 'NUM_MACHINES': 10, 'NAME': '10x10'},
        {'NUM_JOBS': 20, 'NUM_MACHINES': 10, 'NAME': '20x10'},
        {'NUM_JOBS': 30, 'NUM_MACHINES': 10, 'NAME': '30x10'}
    ]
    TEST_SEEDS = [42, 100, 2023, 999, 777]
    TARGET_EPISODE = 500

    all_results = []
    ALGORITHM_TYPES = [
        {'name': 'Integrated-DRL', 'folder_suffix': 'Integrated', 'use_attention': True},
        {'name': 'Base-PPO', 'folder_suffix': 'BasePPO', 'use_attention': False},
    ]

    for scenario in SCENARIOS:
        n_jobs, n_machines, scenario_name = scenario['NUM_JOBS'], scenario['NUM_MACHINES'], scenario['NAME']
        print(f"\n{'=' * 70}\n📊 PK 场景: {scenario_name} (Jobs={n_jobs}, Machines={n_machines})\n{'=' * 70}")

        cfg = Config()
        cfg.NUM_JOBS, cfg.NUM_MACHINES = n_jobs, n_machines
        cfg.DEVICE = 'cpu'

        temp_env = DFFSPEnvironment(n_jobs, cfg.NUM_STAGES, n_machines, cfg)
        s = temp_env.reset()
        cfg.STATE_DIM = cfg.JOB_STATE_DIM = cfg.MACHINE_STATE_DIM = s.shape[0] if isinstance(s, torch.Tensor) else len(
            s)

        model_paths = {}
        folder_int = os.path.join(checkpoint_root, f"J{n_jobs}_M{n_machines}_P0.5_Integrated")
        pth_int = find_integrated_checkpoint(folder_int, "checkpoint", TARGET_EPISODE)
        if pth_int: model_paths['PPO (Ours)'] = {'path': pth_int, 'use_attention': True, 'type': 'integrated'}

        folder_base = os.path.join(checkpoint_root, f"J{n_jobs}_M{n_machines}_P0.5_BasePPO")
        jp, mp = find_base_ppo_checkpoints(folder_base)
        if jp and mp: model_paths['BAPO'] = {'job_path': jp, 'machine_path': mp, 'type': 'base'}

        scenario_results = []
        for seed_idx, seed in enumerate(TEST_SEEDS):
            print(f"\n  🌱 Seed {seed_idx + 1}/{len(TEST_SEEDS)}: {seed}")

            # 1. 经典启发式 (RA, SPT, LPT)
            for rule in ['spt', 'random', 'lpt']:
                res = run_heuristic_agent(DFFSPEnvironment, cfg, rule, seed)
                if res:
                    name = 'RA' if rule == 'random' else f'{rule.upper()}'
                    entry = {'Size': scenario_name, 'Algorithm': name, 'PRT': res['Profit_Rate'], 'GP': 0.0,
                             'Time': res['Time'], 'Seed': seed, 'Profit': res['Profit'], 'Makespan': res['Makespan']}
                    scenario_results.append(entry);
                    all_results.append(entry)
                    print(f"      🔹 {name:15s}: Rate={res['Profit_Rate']:6.2f} | Time: {res['Time']:6.4f}s")

            # 2. 局部搜索元启发式 (SA-SPT)
            res_sa = run_sa_spt_agent(DFFSPEnvironment, cfg, seed)
            if res_sa:
                scenario_results.append(
                    {'Size': scenario_name, 'Algorithm': 'SA-SPT', 'PRT': res_sa['Profit_Rate'], 'GP': 0.0,
                     'Time': res_sa['Time'], 'Seed': seed, 'Profit': res_sa['Profit'], 'Makespan': res_sa['Makespan']})
                print(
                    f"      🔥 {'SA-SPT':15s}: Rate={res_sa['Profit_Rate']:6.2f} | Time: {res_sa['Time']:6.2f}s  <-- 局部迭代")

            # 3. 全局进化元启发式 (NSGA-II)
            if NSGA2Solver is not None:
                res_nsga2 = run_nsga2_agent(DFFSPEnvironment, cfg, seed)
                if res_nsga2:
                    scenario_results.append(
                        {'Size': scenario_name, 'Algorithm': 'NSGA-II', 'PRT': res_nsga2['Profit_Rate'], 'GP': 0.0,
                         'Time': res_nsga2['Time'], 'Seed': seed, 'Profit': res_nsga2['Profit'],
                         'Makespan': res_nsga2['Makespan']})
                    print(
                        f"      🧬 {'NSGA-II':15s}: Rate={res_nsga2['Profit_Rate']:6.2f} | Time: {res_nsga2['Time']:6.2f}s  <-- 种群进化")

            # 4. DRL算法 (BAPO, PPO)
            for algo_name, info in model_paths.items():
                np.random.seed(seed);
                torch.manual_seed(seed)
                env_drl = DFFSPEnvironment(n_jobs, cfg.NUM_STAGES, n_machines, cfg)
                if info['type'] == 'integrated':
                    res = run_drl_inference(env_drl, cfg, info['path'], seed, info['use_attention'])
                else:
                    res = run_drl_base_inference(env_drl, cfg, info['job_path'], info['machine_path'], seed)

                if res:
                    scenario_results.append(
                        {'Size': scenario_name, 'Algorithm': algo_name, 'PRT': res['Profit_Rate'], 'GP': 0.0,
                         'Time': res['Time'], 'Seed': seed, 'Profit': res['Profit'], 'Makespan': res['Makespan']})
                    print(
                        f"      🚀 {algo_name:15s}: Rate={res['Profit_Rate']:6.2f} | Time: {res['Time']:6.4f}s  <-- 极速推理")

        if scenario_results:
            df_s = pd.DataFrame(scenario_results)
            print(f"\n📈 {scenario_name} 场景结果汇总:")
            print(df_s.groupby('Algorithm')[['Profit', 'Makespan', 'PRT', 'Time']].mean().round(2).sort_values('PRT',
                                                                                                               ascending=False))

    if all_results:
        df = pd.DataFrame(all_results)

        # 计算 GP (Gap Ratio) -> (最佳算法的PRT - 当前PRT) / 最佳算法的PRT
        for size in df['Size'].unique():
            for seed in df['Seed'].unique():
                mask = (df['Size'] == size) & (df['Seed'] == seed)
                best_prt = df.loc[mask, 'PRT'].max()
                if best_prt > 0:
                    df.loc[mask, 'GP'] = (best_prt - df.loc[mask, 'PRT']) / best_prt

        summary = df.groupby(['Size', 'Algorithm']).agg(
            {'Profit': 'mean', 'Makespan': 'mean', 'PRT': 'mean', 'GP': 'mean', 'Time': 'mean'}).round(4)
        summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns.values]

        os.makedirs(os.path.join(BASE_PATH, 'experiment_results'), exist_ok=True)
        out_csv = os.path.join(BASE_PATH, 'experiment_results', 'Paper_Table3_Final_Comparison.csv')
        summary.reset_index().to_csv(out_csv, index=False)
        print("\n" + "=" * 80)
        print("🏆 Table 3: Performance and Execution Time comparison (Generated)")
        print("=" * 80)
        print(summary[['PRT', 'GP', 'Time']])


# =====================================================================
# 核心任务 2：传感器误差 (Sensor Noise) 鲁棒性测试 (生成 Table 5)
# =====================================================================
def run_sensor_noise_robustness_test(group_name):
    print("\n" + "=" * 80)
    print("🌪️ 开始运行传感器误差(Sensor Noise)鲁棒性测试 (For Table 5)...")
    print("=" * 80)

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    checkpoint_root = os.path.join(BASE_PATH, "checkpoints", group_name)

    # 选取中等规模进行测试 (20x8)
    n_jobs, n_machines = 20, 8
    folder = os.path.join(checkpoint_root, f"J{n_jobs}_M{n_machines}_P0.5_Integrated")
    pth = find_integrated_checkpoint(folder, "checkpoint", 2000)

    if not pth:
        print("⚠️ 未找到用于噪声测试的 H-MADRL 模型，跳过此测试。")
        return

    cfg = Config()
    cfg.NUM_JOBS, cfg.NUM_MACHINES, cfg.DEVICE = n_jobs, n_machines, 'cpu'
    temp_env = DFFSPEnvironment(n_jobs, cfg.NUM_STAGES, n_machines, cfg)
    s = temp_env.reset()
    cfg.STATE_DIM = cfg.JOB_STATE_DIM = cfg.MACHINE_STATE_DIM = s.shape[0] if isinstance(s, torch.Tensor) else len(s)

    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20]
    noise_labels = ["0.00 (Perfect Sensor)", "0.05 (Slight Noise)", "0.10 (Moderate Noise)", "0.15 (High Uncertainty)",
                    "0.20 (Severe Distortion)"]

    TEST_SEEDS = [42, 100, 2023, 999, 777]  # 5个种子取平均

    results = []
    baseline_prt = None

    for std, label in zip(noise_levels, noise_labels):
        print(f"\n  📉 测试高斯噪声水平: σ = {std} ({label})")
        prts, profits, makespans = [], [], []

        for seed in TEST_SEEDS:
            env = DFFSPEnvironment(n_jobs, cfg.NUM_STAGES, n_machines, cfg)
            res = run_drl_inference(env, cfg, pth, seed, use_attention=True, noise_std=std)
            if res:
                prts.append(res['Profit_Rate'])
                profits.append(res['Profit'])
                makespans.append(res['Makespan'])

        if prts:
            avg_prt = np.mean(prts)
            avg_profit = np.mean(profits)
            avg_makespan = np.mean(makespans)

            if std == 0.00:
                baseline_prt = avg_prt
                decline_str = "Baseline"
            else:
                decline = ((baseline_prt - avg_prt) / baseline_prt) * 100
                decline_str = f"-{decline:.2f}%"

            results.append({
                'Sensor Noise Level (σ)': label,
                'Avg. Profit': avg_profit,
                'Avg. Makespan': avg_makespan,
                'Profit Rate (PRT)': avg_prt,
                'Relative Decline (%)': decline_str
            })
            print(f"      -> PRT: {avg_prt:.2f} | Profit: {avg_profit:.1f} | Makespan: {avg_makespan:.1f}")

    if results:
        df = pd.DataFrame(results)

        out_csv = os.path.join(BASE_PATH, 'experiment_results', 'Paper_Table5_Sensor_Noise_Robustness.csv')
        os.makedirs(os.path.join(BASE_PATH, 'experiment_results'), exist_ok=True)
        df.round(2).to_csv(out_csv, index=False)
        print(f"\n💾 传感器误差分析已保存: {out_csv}")
        print("\n" + "-" * 80)
        print("🏆 Table 5: Performance degradation under sensor uncertainty")
        print("-" * 80)
        try:
            print(df.to_markdown(index=False))
        except:
            print(df)


# =====================================================================
# Main 启动函数
# =====================================================================
def main():
    print("🚀 Starting Disassembly Workshop DRL Experiments Pipeline")

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    generator = ExperimentConfigGenerator()
    runner = BatchExperimentRunner()
    config_root_dir = os.path.join(BASE_PATH, "experiment_configs")

    # === 1. 首先确保模型被训练出来 ===
    target_groups = ['全面实验']

    if os.path.exists(config_root_dir):
        print(f"🧹 初始化清理: 删除旧的 {config_root_dir}")
        try:
            shutil.rmtree(config_root_dir)
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")

    for group_name in target_groups:
        print(f"\n{'=' * 60}\n🎯 Processing Experiment Group: {group_name}\n{'=' * 60}")
        group_configs = generator.generate_configs_by_group_name(group_name)
        if not group_configs: continue

        generator.save_configs(group_configs, output_dir=config_root_dir, clear_output_dir=False)

        # 运行 DRL 训练 (这将生成所需的 checkpoint)
        print(f"🏃 Training DRL Models for: {group_name}...")
        runner.run_experiment_group(group_name, num_repeats=1)

    print("\n🎉 DRL Training completed!")

    # === 2. 运行对比和传感器误差测试 ===
    for group_name in target_groups:
        # 1. 运行核心算法时间性能大乱斗 (Table 3)
        run_algorithm_comparison(group_name)

        # 2. 运行高斯传感器噪声鲁棒性测试 (Table 5)
        run_sensor_noise_robustness_test(group_name)

    print("\n✅ 所有任务执行完毕！")


if __name__ == '__main__':
    main()