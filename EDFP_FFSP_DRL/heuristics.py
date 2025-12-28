# EDFP_FFSP_DRL/heuristics.py
import numpy as np
import torch
import time


class HeuristicSolver:
    """
    增强版启发式求解器
    包含:
    1. Random: 随机选择
    2. SPT: 最短加工时间/最小成本 (基准)
    3. LPT: 最长加工时间/最大成本
    4. SA-SPT: 结构感知 SPT (Structure-Aware SPT) - 能够识别结构完整性并动态加权
    """

    def __init__(self, env):
        self.env = env

    def solve(self, rule='random', render=False):
        """
        运行一回合调度
        :param rule: 'random', 'spt', 'lpt', 'sa-spt'
        """
        # 重置环境
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0

        # 记录每一步的选择 (可选，用于调试)
        history = {
            'job_actions': [],
            'machine_actions': []
        }

        if render:
            print(f"🚀 开始运行启发式规则: {rule.upper()}")

        start_time = time.time()

        while not done:
            # 1. 获取当前可用的动作
            available_jobs, available_machines = self.env.get_available_actions()

            # --- 【全局防死锁机制】 ---
            # 如果没有可用动作（例如所有机器都在忙，或工件还没到），需要推进环境时间
            if not available_jobs or not available_machines:
                old_time = self.env.current_time
                # 调用环境内部方法寻找下一个决策点（机器释放或工件到达）
                # 注意：这里假设 env 有 _find_next_decision_point 方法
                # 如果没有，可以直接用 self.env.current_time += 1.0 替代
                if hasattr(self.env, '_find_next_decision_point'):
                    self.env.current_time = self.env._find_next_decision_point()
                else:
                    self.env.current_time += 1.0

                # 双重保险：如果时间没有变化（极少数情况），强制推进一小步
                if self.env.current_time <= old_time:
                    self.env.current_time += 1.0

                steps += 1
                continue
            # ---------------------------

            job_action = None
            machine_action = None

            # ====================================================
            # 🟢 规则 1: 随机 (Random)
            # ====================================================
            if rule == 'random':
                job_action = np.random.choice(available_jobs)
                machine_action = np.random.choice(available_machines)

            # ====================================================
            # 🔵 规则 2 & 3: 基础 SPT / LPT (只看成本)
            # ====================================================
            elif rule in ['spt', 'lpt']:
                best_pair = None
                # SPT找最小成本，LPT找最大成本
                best_metric = float('inf') if rule == 'spt' else -float('inf')

                for j in available_jobs:
                    # 获取该工件当前的工序索引
                    current_op = self.env.job_current_operations[j]
                    if current_op >= self.env.num_stages:
                        continue

                    for m in available_machines:
                        try:
                            # 读取成本矩阵: [jobs, stages, machines]
                            cost = self.env.disassembly_costs[j, current_op, m]

                            if rule == 'spt':
                                if cost < best_metric:
                                    best_metric = cost
                                    best_pair = (j, m)
                            else:  # lpt
                                if cost > best_metric:
                                    best_metric = cost
                                    best_pair = (j, m)
                        except IndexError:
                            continue

                # 如果找到了最优对，就执行；否则随机兜底
                if best_pair:
                    job_action, machine_action = best_pair
                else:
                    job_action = np.random.choice(available_jobs)
                    machine_action = np.random.choice(available_machines)

            # ====================================================
            # 🟠 规则 4: 结构感知 SPT (Structure-Aware SPT)
            # ====================================================
            elif rule == 'sa-spt':
                best_pair = None
                best_score = float('inf')  # 分数越低越好

                for j in available_jobs:
                    current_op = self.env.job_current_operations[j]
                    if current_op >= self.env.num_stages:
                        continue

                    # [新增] 读取该工件的结构完整性 (0.0 ~ 1.0)
                    structure = self.env.product_structure_completeness[j]
                    # [新增] 读取产品类型
                    p_type = self.env.product_types[j]

                    for m in available_machines:
                        try:
                            cost = self.env.disassembly_costs[j, current_op, m]

                            # --- 🧠 专家逻辑：给成本加权 ---

                            # 1. 结构完整性逻辑：
                            # 如果结构很差 (<0.4)，优先使用"耐用/重型"机器 (假设是 ID 0, 2)
                            penalty = 1.0
                            if structure < 0.4:
                                if m in [0, 2]:
                                    penalty = 0.8  # 鼓励 (打8折)
                                else:
                                    penalty = 1.5  # 惩罚 (涨价50%)

                            # 2. 精密加工逻辑：
                            # 如果是电子产品且结构尚好，优先用"精密"机器 (假设是 ID 1, 3)
                            if p_type == 'electronics' and structure > 0.7:
                                if m in [1, 3]:
                                    penalty = 0.7  # 鼓励

                            # 计算最终得分 (Score = Cost * Penalty)
                            # SPT 倾向于选 Score 小的
                            score = cost * penalty

                            if score < best_score:
                                best_score = score
                                best_pair = (j, m)

                        except IndexError:
                            continue

                if best_pair:
                    job_action, machine_action = best_pair
                else:
                    job_action = np.random.choice(available_jobs)
                    machine_action = np.random.choice(available_machines)

            # ====================================================
            # 执行动作
            # ====================================================
            state, reward, done, info = self.env.step(job_action, machine_action)
            total_reward += reward
            steps += 1

            history['job_actions'].append(job_action)
            history['machine_actions'].append(machine_action)

        # 循环结束，计算最终指标
        final_profit = self.env.total_profit
        final_makespan = self.env._calculate_makespan()
        elapsed_time = time.time() - start_time

        if render:
            print(f"🏁 {rule.upper()} 完成: 利润={final_profit:.2f}, 完工时间={final_makespan:.2f}")

        return {
            'rule': rule,
            'total_reward': total_reward,
            'final_profit': final_profit,  # 供 run_comparison 使用
            'final_makespan': final_makespan,  # 供 run_comparison 使用
            'steps': steps,
            'time': elapsed_time
        }