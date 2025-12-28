# environment/dffsp_env.py (利润优先优化版 - 修改成本公式和利润稳定性)
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import copy
import traceback


class DFFSPEnvironment:
    """
    退役产品拆解车间调度环境 - 利润优先优化版
    特点：
    1. 明确优先级：利润最大化 > 完工时间最小化
    2. 改进的奖励函数，突出利润目标
    3. 修改成本公式：成本 = 基础成本 * 处理时间
    4. 稳定利润范围：移除随机乘数，使利润更稳定
    """

    def __init__(self, num_jobs: int, num_stages: int, num_machines: int, config=None,
                 product_types=None, operation_sequences=None, machine_configs=None,
                 skip_probabilities=None, alternative_operations=None):
        print(f"🏭 初始化退役产品拆解车间环境: {num_jobs}个产品, {num_stages}个拆解工序, {num_machines}台设备")

        self.num_jobs = num_jobs
        self.num_stages = num_stages
        self.num_machines = num_machines
        self.config = config

        # ---------------------------------------------------------------------
        # 1. 基础配置初始化
        # ---------------------------------------------------------------------
        # 产品类型配置
        self.product_types = product_types or ['electronics', 'mechanical', 'mixed'] * (num_jobs // 3 + 1)
        self.product_types = self.product_types[:num_jobs]

        # 工序配置 - 确保所有工序都在有效范围内
        self.operation_sequences = operation_sequences or {
            'electronics': [0, 1, 2, 3],  # 拆卸外壳、分离电路板、提取贵金属、分类处理
            'mechanical': [0, 2, 4],  # 拆卸外壳、分解机械部件、分类处理
            'mixed': [0, 1, 2, 3, 4]  # 综合拆解流程
        }

        # 过滤无效工序
        for product_type, sequence in self.operation_sequences.items():
            self.operation_sequences[product_type] = [op for op in sequence if op < num_stages]

        # 机器配置
        self.machine_configs = machine_configs or {
            0: ['universal_disassembly', 'robust_disassembly'],
            1: ['precise_pcb_extractor', 'universal_pcb_extractor'],
            2: ['metal_separator_heavy', 'metal_separator_light'],
            3: ['precious_metal_extractor_precise', 'precious_metal_extractor_standard'],
            4: ['universal_sorting', 'fast_sorting']
        }
        self.machine_configs = {k: v for k, v in self.machine_configs.items() if k < num_stages}

        # 处理时间配置 - 不同工序在不同机器上的处理时间
        self.processing_times = self._initialize_processing_times()

        # 跳过概率与备选工序
        self.skip_probabilities = skip_probabilities or {1: 0.2, 3: 0.3}
        self.skip_probabilities = {k: v for k, v in self.skip_probabilities.items() if k < num_stages}

        self.alternative_operations = alternative_operations or {1: 4, 3: 4}
        self.alternative_operations = {k: v for k, v in self.alternative_operations.items()
                                       if k < num_stages and v < num_stages}

        # ---------------------------------------------------------------------
        # 2. 产品属性生成
        # ---------------------------------------------------------------------
        self.product_ages = np.random.uniform(1, 20, num_jobs)
        self.product_conditions = np.random.uniform(0.1, 1.0, num_jobs)
        self.product_structure_completeness = np.random.uniform(0.3, 1.0, num_jobs)
        self.critical_components = np.random.choice([0, 1], num_jobs, p=[0.7, 0.3])

        # 明确的目标权重：利润优先
        self.current_profit_weight = getattr(config, 'PROFIT_WEIGHT', 0.7) if config else 0.7
        self.current_makespan_weight = getattr(config, 'MAKESPAN_WEIGHT', 0.2) if config else 0.2
        self.current_balance_weight = getattr(config, 'BALANCE_WEIGHT', 0.1) if config else 0.1

        # 奖励统计
        self.reward_history = []
        self.profit_history = []
        self.makespan_history = []

        # 添加目标优先级记录
        self.objective_priority = "profit_first"  # profit_first, balanced, makespan_first
        self.dynamic_adjustment = True  # 是否动态调整权重

        # ---------------------------------------------------------------------
        # 3. 状态表示模块
        # ---------------------------------------------------------------------
        try:
            from EDFP_FFSP_DRL.environment.state_representation import StateRepresentation
            self.state_representation = StateRepresentation(num_jobs, num_stages, num_machines, config)
            print("✅ 状态表示模块初始化成功")
        except ImportError:
            print("⚠️ 未找到StateRepresentation模块，将使用简化状态。")
            self.state_representation = None
        except Exception as e:
            print(f"❌ 状态表示模块初始化失败: {e}")
            self.state_representation = None

        # ---------------------------------------------------------------------
        # 4. 运行时变量初始化
        # ---------------------------------------------------------------------
        self.total_profit = 0.0
        self.completed_jobs = set()
        self.reset()
        print(f"✅ 环境初始化完成，目标优先级: 利润优先")
        print(f"   奖励权重: 利润={self.current_profit_weight:.2f}, "
              f"时间={self.current_makespan_weight:.2f}, 平衡={self.current_balance_weight:.2f}")

    def _initialize_processing_times(self):
        """初始化处理时间矩阵"""
        # 基础处理时间配置 (工序 -> 基础时间)
        base_times = {
            0: 8,  # 拆卸外壳
            1: 15,  # 分离电路板
            2: 12,  # 提取贵金属
            3: 20,  # 精细分类
            4: 6  # 基础分类
        }

        # 扩展基础时间到所有工序
        processing_times = {}
        for stage in range(self.num_stages):
            base_time = base_times.get(stage, 10)  # 默认10

            # 为每台机器生成略有不同的处理时间
            machine_times = []
            for machine in range(self.num_machines):
                # 根据机器效率调整处理时间
                efficiency_factor = np.random.uniform(0.8, 1.2)
                machine_time = base_time * efficiency_factor
                machine_times.append(machine_time)

            processing_times[stage] = machine_times

        return processing_times

    def reset(self) -> torch.Tensor:
        """重置环境到初始状态"""
        # 生成动态参数
        self.disassembly_costs = self._generate_disassembly_costs()
        self.disassembly_profits = self._generate_disassembly_profits()

        # 重新随机化产品属性 (增加训练多样性)
        self.product_structure_completeness = np.random.uniform(0.3, 1.0, self.num_jobs)
        self.critical_components = np.random.choice([0, 1], self.num_jobs, p=[0.7, 0.3])
        self.product_ages = np.random.uniform(1, 20, self.num_jobs)

        # 初始化时间与状态
        self.current_time = 0
        self.completed_jobs = set()

        # 状态矩阵
        self.job_stages = np.zeros(self.num_jobs, dtype=int)
        self.job_arrival_times = np.zeros(self.num_jobs)
        self.job_completion_times = np.zeros((self.num_jobs, self.num_stages))
        self.job_current_operations = [0] * self.num_jobs

        # 机器状态
        self.machine_busy_until = [0.0] * self.num_machines
        self.machine_schedule = [[] for _ in range(self.num_machines)]

        # 利润统计
        self.total_profit = 0.0
        self.job_profits = np.zeros(self.num_jobs)

        # 动作历史记录
        self.action_history = []
        self.job_action_counts = np.zeros(self.num_jobs)
        self.machine_action_counts = np.zeros(self.num_machines)

        # 生成到达时间
        self._generate_dynamic_arrivals()

        return self._get_state_representation()

    def step(self, job_action: int, machine_action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        执行拆解调度动作 - 利润优先优化版
        核心逻辑：
        1. 利润是首要优化目标
        2. 完工时间是次要优化目标
        3. 奖励函数明确反映这一优先级
        """
        reward = 0.0
        done = False
        info = {}

        # 记录动作历史
        self.action_history.append((job_action, machine_action))
        if len(self.action_history) > 100:  # 保持最近100个动作
            self.action_history = self.action_history[-100:]

        self.job_action_counts[job_action] += 1
        self.machine_action_counts[machine_action] += 1

        # ---------------------------------------------------------
        # 1. 强力惩罚：越界检查
        # ---------------------------------------------------------
        if (job_action < 0 or job_action >= self.num_jobs or
                machine_action < 0 or machine_action >= self.num_machines):
            reward = -2.0  # 重罚越界动作
            info['action_out_of_bounds'] = True
            next_state = self._get_state_representation()
            return next_state, reward, done, info

        # 获取当前可用动作
        available_jobs, available_machines = self.get_available_actions()

        # ---------------------------------------------------------
        # 2. 环境自动推进时间（如果是无可奈何的等待）
        # ---------------------------------------------------------
        if not available_jobs or not available_machines:
            # 如果是因为真的没法动（都在忙），推进时间，给轻微惩罚
            old_time = self.current_time
            self.current_time = self._find_next_decision_point()
            reward = -0.2  # 增加等待惩罚
            info['waiting'] = True
            next_state = self._get_state_representation()
            return next_state, reward, done, info

        # ---------------------------------------------------------
        # 3. 强力惩罚：无效选择（Masking Failure Penalty）
        # ---------------------------------------------------------
        # 如果有机器可用，但智能体选了忙碌的；或有工件可用，选了没到的/做完的
        if job_action not in available_jobs or machine_action not in available_machines:
            reward = -1.5  # 重罚无效动作
            info['invalid_action'] = True
            next_state = self._get_state_representation()
            return next_state, reward, done, info

        # ---------------------------------------------------------
        # 4. 执行有效动作
        # ---------------------------------------------------------
        try:
            current_operation = self.job_current_operations[job_action]

            # 获取处理时间（基于工序和机器）
            processing_time = self._get_processing_time(current_operation, machine_action, job_action)

            # 计算利润与成本
            cost = self.disassembly_costs[job_action, current_operation, machine_action]
            profit = self.disassembly_profits[job_action, current_operation, machine_action]
            net_profit = profit - cost

            # 计算时间窗口
            start_time = max(self.current_time,
                             self.machine_busy_until[machine_action],
                             self.job_arrival_times[job_action])
            end_time = start_time + processing_time

            # 更新机器占用
            self.machine_busy_until[machine_action] = end_time
            self.machine_schedule[machine_action].append(
                (job_action, start_time, end_time, current_operation, net_profit)
            )

            # 更新工件记录
            if current_operation < self.num_stages:
                self.job_completion_times[job_action, current_operation] = end_time

            # 决定下一道工序（包含跳过逻辑）
            next_operation = self._get_next_operation(job_action, current_operation)
            self.job_current_operations[job_action] = next_operation
            self.job_stages[job_action] = next_operation

            # 更新累积利润
            self.total_profit += net_profit
            self.job_profits[job_action] += net_profit

            # 检查该工件是否彻底完成
            job_completed = (next_operation >= self.num_stages)
            if job_completed:
                self.completed_jobs.add(job_action)

            # ---------------------------------------------------------
            # 5. 计算利润优先的奖励
            # ---------------------------------------------------------
            reward = self._calculate_efficiency_reward(net_profit, processing_time, job_completed)

            # 记录奖励统计
            self.reward_history.append(reward)
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]

            # 推进环境时间
            self.current_time = self._find_next_decision_point()

            # ---------------------------------------------------------
            # 6. 检查Episode是否结束
            # ---------------------------------------------------------
            done = len(self.completed_jobs) == self.num_jobs
            if done:
                # 终极大奖：利润主导的综合评估
                makespan = self._calculate_makespan()
                total_profit = self.total_profit

                # 计算全局利润率 (防止除以零)
                global_profit_rate = total_profit / max(1.0, makespan)

                # 终极奖励直接与全局利润率挂钩
                # 系数 100 是为了让数值匹配神经网络的偏好范围 (-1 到 10 左右)
                episode_bonus = global_profit_rate * 100.0

                reward += episode_bonus

                info['episode_completed'] = True
                info['final_profit_rate'] = global_profit_rate  # 记录指

                                # 记录最终指标
                self.makespan_history.append(makespan)
                self.profit_history.append(total_profit)

            # 返回新状态
            next_state = self._get_state_representation()

            # 记录调试信息
            info.update({
                'net_profit': net_profit,
                'total_profit': self.total_profit,
                'job_completed': job_completed,
                'processing_time': processing_time,
                'job_action': job_action,
                'machine_action': machine_action,
                'reward_components': {
                    'profit': net_profit * 0.1 * self.current_profit_weight,
                    'time': -processing_time * 0.01 * self.current_makespan_weight,
                    'balance': (1.0 - len(self.machine_schedule[machine_action]) / (
                            self.num_jobs * 0.5)) * 0.5 * self.current_balance_weight
                }
            })

            return next_state, reward, done, info

        except Exception as e:
            print(f"❌ 执行异常: {e}")
            traceback.print_exc()
            return self._get_state_representation(), -2.0, False, {'error': str(e)}

    def _calculate_efficiency_reward(self, net_profit, processing_time, job_completed):
        """
        效率优先奖励函数：
        目标是最大化单位时间的利润 (Profit Rate)
        """
        reward = 0.0

        # 1. 瞬时效率奖励 (核心)
        # 如果处理时间极短，避免数值爆炸，设置最小分母
        safe_time = max(1.0, processing_time)
        step_efficiency = net_profit / safe_time

        # 缩放系数，防止 reward 过大 (假设 efficiency 约在 5-20 之间)
        reward += step_efficiency * 0.5

        # 2. 完工激励 (可选，但建议保留一点，防止为了高瞬时效率而故意不做最后一步)
        if job_completed:
            reward += 1.0

        return reward

    def _get_processing_time(self, operation: int, machine: int, job_id: int) -> float:
        """获取处理时间"""
        if operation in self.processing_times and machine < len(self.processing_times[operation]):
            base_time = self.processing_times[operation][machine]

            # 根据产品状态调整处理时间
            # 结构完整性差的产品需要更长时间
            structure_factor = 1.0 - self.product_structure_completeness[job_id]
            adjusted_time = base_time * (1.0 + structure_factor * 0.5)

            # 产品年限影响：老产品更难拆解
            age_factor = self.product_ages[job_id] / 20.0
            adjusted_time = adjusted_time * (1.0 + age_factor * 0.3)

            return adjusted_time

        # 默认处理时间
        return 10.0

    def _calculate_profit_focused_reward(self, net_profit, job_completed, job_id, machine_id,
                                         start_time, end_time, processing_time):
        """
        利润优先的奖励函数 - 明确优先级：利润最大化 > 完工时间最小化
        """
        reward = 0.0

        # 基础奖励：净利润
        base_reward = net_profit * 0.01  # 适当缩放

        # 完工奖励
        if job_completed:
            base_reward += 5.0

        # 时间惩罚（轻微）
        time_penalty = processing_time * 0.001

        # 最终奖励
        final_reward = base_reward - time_penalty

        return final_reward
    def _get_next_operation(self, job_id: int, current_operation: int) -> int:
        """获取下一个工序（包含跳过逻辑）"""
        product_type = self.product_types[job_id]
        operation_sequence = self.operation_sequences[product_type]

        # 异常处理：当前工序不在序列中
        if current_operation not in operation_sequence:
            # 尝试找下一个比当前大的有效工序
            for op in operation_sequence:
                if op > current_operation:
                    return op
            return self.num_stages

        current_idx = operation_sequence.index(current_operation)

        # 如果是最后一个工序
        if current_idx >= len(operation_sequence) - 1:
            return self.num_stages

        # 默认下一个
        next_operation = operation_sequence[current_idx + 1]

        # ---------------------------------------------------------
        # 跳过逻辑检测
        # ---------------------------------------------------------
        if self._should_skip_operation(job_id, current_operation):
            # 尝试使用备选工序
            if current_operation in self.alternative_operations:
                alt = self.alternative_operations[current_operation]
                if alt in operation_sequence and alt > current_operation:
                    return alt

            # 如果没备选，直接找再下一个有效工序
            for op in operation_sequence[current_idx + 1:]:
                if op > current_operation:
                    return op

            return self.num_stages

        return next_operation

    def _should_skip_operation(self, job_id: int, operation: int) -> bool:
        """
        判断是否跳过当前工序
        基于：结构完整性、关键部件缺失、概率
        """
        # 2. 关键部件缺失 -> 跳过提取工序
        if self.critical_components[job_id] == 0:
            if operation == 2:  # 假设2是贵金属提取
                return True

        # 3. 概率跳过 (模拟不可预见的损坏)
        if operation in self.skip_probabilities:
            # 年限越久，跳过概率越高
            age_factor = self.product_ages[job_id] / 20.0
            base_prob = self.skip_probabilities[operation]
            real_prob = min(0.9, base_prob * (1 + age_factor))
            return np.random.random() < real_prob

        return False

    def get_available_actions(self) -> Tuple[List[int], List[int]]:
        """获取当前可用的 Job 和 Machine 列表"""
        available_jobs = []
        available_machines = []

        # 找空闲机器
        for m_id in range(self.num_machines):
            if self.machine_busy_until[m_id] <= self.current_time + 1e-5:
                available_machines.append(m_id)

        # 找可用工件
        for j_id in range(self.num_jobs):
            if j_id in self.completed_jobs:
                continue
            if self.job_arrival_times[j_id] > self.current_time + 1e-5:
                continue
            if self.job_current_operations[j_id] >= self.num_stages:
                continue
            available_jobs.append(j_id)

        return available_jobs, available_machines

    def _find_next_decision_point(self) -> float:
        """寻找下一个决策时间点"""
        if len(self.completed_jobs) == self.num_jobs:
            return self.current_time

        # 1. 未来最早的机器释放时间
        future_machine_times = [t for t in self.machine_busy_until if t > self.current_time + 1e-6]
        next_machine_time = min(future_machine_times) if future_machine_times else float('inf')

        # 2. 未来最早的工件到达时间
        next_arrival_time = float('inf')
        for j_id in range(self.num_jobs):
            if j_id not in self.completed_jobs and self.job_arrival_times[j_id] > self.current_time + 1e-6:
                next_arrival_time = min(next_arrival_time, self.job_arrival_times[j_id])

        next_time = min(next_machine_time, next_arrival_time)

        if next_time == float('inf') or next_time <= self.current_time:
            # 防死锁：如果找不到下一个点，强制推进一小步
            return self.current_time + 1.0

        return next_time

    def _generate_dynamic_arrivals(self):
        """生成动态到达时间"""
        for j_id in range(self.num_jobs):
            if np.random.random() < 0.3:
                self.job_arrival_times[j_id] = np.random.uniform(10, 50)
            else:
                self.job_arrival_times[j_id] = 0

    def _generate_disassembly_costs(self):
        """生成成本矩阵 - 修改：成本 = 基础成本 * 处理时间"""
        costs = np.zeros((self.num_jobs, self.num_stages, self.num_machines))

        # 单位时间成本率
        cost_rates = {0: 1.0, 1: 2.5, 2: 1.5, 3: 3.0, 4: 0.5}

        for j in range(self.num_jobs):
            for s in range(self.num_stages):
                cost_rate = cost_rates.get(s, 1.0)

                for m in range(self.num_machines):
                    # 获取处理时间
                    processing_time = self._get_processing_time(s, m, j)

                    # 基础成本 = 成本率 * 处理时间
                    base_cost = cost_rate * processing_time

                    # 根据产品状态调整成本
                    # 结构完整性差的产品成本更高
                    structure_factor = 1.0 - self.product_structure_completeness[j]
                    cost = base_cost * (1.0 + structure_factor * 0.5)

                    # 关键部件缺失可能降低成本（跳过某些工序）
                    if self.critical_components[j] == 0 and s == 2:
                        cost *= 0.8  # 贵金属提取成本降低

                    costs[j, s, m] = cost

        return costs

    def _generate_disassembly_profits(self):
        """生成利润矩阵 - 修改：稳定利润范围，移除随机乘数"""
        profits = np.zeros((self.num_jobs, self.num_stages, self.num_machines))

        # 单位时间利润率（稳定，不随机变化）
        profit_rates = {0: 3.0, 1: 8.0, 2: 15.0, 3: 12.0, 4: 4.0}

        # 产品类型利润系数
        type_factors = {
            'electronics': 1.5,
            'mechanical': 1.0,
            'mixed': 1.2
        }

        for j in range(self.num_jobs):
            product_type = self.product_types[j]
            type_factor = type_factors.get(product_type, 1.0)

            for s in range(self.num_stages):
                profit_rate = profit_rates.get(s, 5.0)

                for m in range(self.num_machines):
                    # 获取处理时间
                    processing_time = self._get_processing_time(s, m, j)

                    # 基础利润 = 利润率 * 处理时间 * 产品类型系数
                    base_profit = profit_rate * processing_time * type_factor

                    # 根据产品状态微调利润
                    # 结构完整性好的产品利润更高
                    structure_factor = self.product_structure_completeness[j]
                    profit = base_profit * (0.8 + structure_factor * 0.4)

                    # 关键部件存在增加利润
                    if self.critical_components[j] == 1 and s == 2:
                        profit *= 1.5  # 贵金属提取利润增加

                    profits[j, s, m] = profit

        return profits

    def _get_state_representation(self) -> torch.Tensor:
        """获取状态向量"""
        if self.state_representation:
            raw = self._get_raw_state()
            return self.state_representation.get_combined_state(raw)
        else:
            # 简单的备用状态生成
            dim = self.config.STATE_DIM if self.config else 64
            return torch.zeros(dim)

    def _get_raw_state(self) -> Dict[str, Any]:
        """打包原始状态供StateRepresentation使用"""
        return {
            'job_stages': self.job_stages,
            'job_current_operations': self.job_current_operations,
            'completed_jobs': self.completed_jobs,
            'disassembly_costs': self.disassembly_costs,
            'disassembly_profits': self.disassembly_profits,
            'machine_schedule': self.machine_schedule,
            'job_completion_times': self.job_completion_times,
            'current_time': self.current_time,
            'machine_busy_until': self.machine_busy_until,
            'job_arrival_times': self.job_arrival_times,
            'total_profit': self.total_profit,
            'job_profits': self.job_profits,
            'product_ages': self.product_ages,
            'product_conditions': self.product_conditions,
            'product_types': self.product_types,
            'product_structure_completeness': self.product_structure_completeness,
            'critical_components': self.critical_components,
            'job_action_counts': self.job_action_counts,
            'machine_action_counts': self.machine_action_counts,
            'processing_times': self.processing_times
        }

    def _calculate_makespan(self) -> float:
        """计算完工时间"""
        if not self.completed_jobs:
            return 0.0
        makespan = 0.0
        for j in self.completed_jobs:
            job_makespan = np.max(self.job_completion_times[j])
            makespan = max(makespan, job_makespan)
        return makespan

    def get_action_diversity_stats(self):
        """获取动作多样性统计"""
        if len(self.action_history) == 0:
            return {"job_diversity": 0.0, "machine_diversity": 0.0}

        unique_jobs = len(set([a[0] for a in self.action_history]))
        unique_machines = len(set([a[1] for a in self.action_history]))
        total_actions = len(self.action_history)

        return {
            "job_diversity": unique_jobs / total_actions,
            "machine_diversity": unique_machines / total_actions,
            "unique_jobs": unique_jobs,
            "unique_machines": unique_machines,
            "total_actions": total_actions
        }

    def update_reward_weights(self, profit_weight=None, makespan_weight=None, balance_weight=None):
        """动态更新奖励权重"""
        if profit_weight is not None:
            self.current_profit_weight = profit_weight
        if makespan_weight is not None:
            self.current_makespan_weight = makespan_weight
        if balance_weight is not None:
            self.current_balance_weight = balance_weight

        print(f"⚖️ 更新奖励权重: 利润={self.current_profit_weight:.2f}, "
              f"时间={self.current_makespan_weight:.2f}, 平衡={self.current_balance_weight:.2f}")