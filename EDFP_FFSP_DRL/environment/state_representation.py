# environment/state_representation.py
import numpy as np
import torch
from typing import Dict, List, Tuple, Any


class StateRepresentation:
    """
    状态表示模块 - 退役产品拆解车间专用版本（改进版）
    """

    def __init__(self, num_jobs: int, num_stages: int, num_machines: int, config=None):
        self.num_jobs = num_jobs
        self.num_stages = num_stages
        self.num_machines = num_machines
        self.config = config
        print(f"✅ 退役产品拆解车间状态表示模块初始化: {num_jobs}产品, {num_stages}工序, {num_machines}设备")

    def get_combined_state(self, env_state: Dict) -> torch.Tensor:
        """获取组合状态表示 - 退役产品拆解车间专用（包含结构信息）"""
        try:
            # 提取环境状态
            job_stages = env_state['job_stages']
            job_current_operations = env_state['job_current_operations']
            completed_jobs = env_state['completed_jobs']
            disassembly_costs = env_state['disassembly_costs']
            disassembly_profits = env_state['disassembly_profits']
            machine_busy_until = env_state['machine_busy_until']
            current_time = env_state['current_time']
            job_arrival_times = env_state['job_arrival_times']
            machine_schedule = env_state['machine_schedule']
            total_profit = env_state['total_profit']
            job_profits = env_state['job_profits']
            product_ages = env_state['product_ages']
            product_conditions = env_state['product_conditions']
            product_types = env_state['product_types']
            product_structure_completeness = env_state['product_structure_completeness']
            critical_components = env_state['critical_components']

            state_features = []

            # 1. 产品状态特征（增强）
            for job_id in range(self.num_jobs):
                # 基础状态
                is_completed = 1.0 if job_id in completed_jobs else 0.0
                current_operation = job_current_operations[job_id]
                current_stage = current_operation / self.num_stages
                has_arrived = 1.0 if job_arrival_times[job_id] <= current_time else 0.0

                # 利润相关特征
                accumulated_profit = job_profits[job_id] / 100.0  # 归一化

                # 产品属性特征
                age_normalized = product_ages[job_id] / 20.0
                condition_normalized = product_conditions[job_id]

                # 当前工序的成本效益特征
                if current_operation < self.num_stages:
                    current_cost = np.mean(disassembly_costs[job_id, current_operation])
                    current_profit = np.mean(disassembly_profits[job_id, current_operation])
                    cost_profit_ratio = current_cost / max(1, current_profit)
                else:
                    current_cost = 0.0
                    current_profit = 0.0
                    cost_profit_ratio = 0.0

                # 产品类型编码
                type_encoding = self._encode_product_type(product_types[job_id])

                # 产品结构特征 - 新增
                structure_completeness = product_structure_completeness[job_id]
                critical_component = float(critical_components[job_id])
                age_impact = product_ages[job_id] / 20.0  # 年限影响
                condition_impact = 1.0 - product_conditions[job_id]  # 工况影响

                state_features.extend([
                    is_completed, current_stage, has_arrived,
                    accumulated_profit, age_normalized, condition_normalized,
                    current_cost / 50.0, current_profit / 150.0, cost_profit_ratio,
                    structure_completeness, critical_component, age_impact, condition_impact,
                    *type_encoding
                ])

            # 2. 机器状态特征
            for machine_id in range(self.num_machines):
                is_idle = 1.0 if machine_busy_until[machine_id] <= current_time else 0.0
                remaining_time = max(0, machine_busy_until[machine_id] - current_time) / 50.0

                # 机器负载和利润贡献
                machine_load = len(machine_schedule[machine_id]) / (self.num_jobs * 0.5)
                machine_profit = sum(job[4] for job in machine_schedule[machine_id]) / 100.0  # 累计利润

                state_features.extend([
                    is_idle, remaining_time, machine_load, machine_profit
                ])

            # 3. 全局特征（增强）
            completion_rate = len(completed_jobs) / self.num_jobs
            current_time_normalized = current_time / 200.0
            total_profit_normalized = total_profit / (self.num_jobs * 100)

            # 系统效率指标
            avg_job_stage = np.mean(job_current_operations) / self.num_stages
            profit_efficiency = total_profit / max(1, current_time)

            # 结构完整性统计
            avg_structure_completeness = np.mean(product_structure_completeness)
            critical_component_ratio = np.mean(critical_components)

            state_features.extend([
                completion_rate, avg_job_stage, current_time_normalized,
                total_profit_normalized, profit_efficiency,
                avg_structure_completeness, critical_component_ratio
            ])

            state_array = np.array(state_features, dtype=np.float32)

            # 归一化
            if self.config and hasattr(self.config, 'STATE_NORMALIZATION') and self.config.STATE_NORMALIZATION:
                state_array = self._normalize_state(state_array)

            return torch.FloatTensor(state_array)

        except Exception as e:
            print(f"❌ 状态表示错误: {e}")
            # 返回默认状态
            default_dim = self.num_jobs * 17 + self.num_machines * 4 + 7  # 调整维度
            return torch.zeros(default_dim)

    def _encode_product_type(self, product_type: str) -> List[float]:
        """产品类型编码"""
        encoding = [0.0, 0.0, 0.0]  # electronics, mechanical, mixed
        if product_type == 'electronics':
            encoding[0] = 1.0
        elif product_type == 'mechanical':
            encoding[1] = 1.0
        elif product_type == 'mixed':
            encoding[2] = 1.0
        return encoding

    def _normalize_state(self, state_array):
        """状态归一化"""
        state_array = np.clip(state_array, -1, 1)
        return state_array