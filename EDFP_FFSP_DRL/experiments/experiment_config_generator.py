# experiments/experiment_config_generator.py
import itertools
import json
import os
import shutil
import torch
from typing import Dict, List, Tuple
import numpy as np


class ExperimentConfigGenerator:
    """实验配置生成器"""

    def __init__(self, base_config_path: str = None):
        self.base_config = self._load_base_config(base_config_path)

        # 定义实验变量
        self.experiment_variables = {
            'NUM_JOBS': [5, 10, 20, 30],
            'NUM_MACHINES': [4, 8, 10],
            'PROFIT_WEIGHT': [0.1, 0.5, 0.7],
            'MAKESPAN_WEIGHT': [0.01, 0.05, 0.1],
            'TRAINER_TYPE': ['Integrated', 'BasePPO']  # 只保留这两种
        }

        # 实验分组设计
        self.experiment_groups = [
            {
                'name': '环境规模实验',
                'variables': ['NUM_JOBS', 'NUM_MACHINES'],
                'fixed': {'PROFIT_WEIGHT': 0.5, 'MAKESPAN_WEIGHT': 0.05, 'TRAINER_TYPE': 'BalancedPPO'}
            },
            {
                'name': '奖励权重实验',
                'variables': ['PROFIT_WEIGHT', 'MAKESPAN_WEIGHT'],
                'fixed': {'NUM_JOBS': 10, 'NUM_MACHINES': 8, 'TRAINER_TYPE': 'BalancedPPO'}
            },
            {
                'name': '算法对比实验',
                'variables': ['TRAINER_TYPE'],
                'fixed': {'NUM_JOBS': 10, 'NUM_MACHINES': 8, 'PROFIT_WEIGHT': 0.5, 'MAKESPAN_WEIGHT': 0.05}
            },
            {
                'name': '全面实验',
                # 变量列表：只变动 Trainer 和 Job
                'variables': ['TRAINER_TYPE', 'NUM_JOBS'],
                # 固定参数：机器数固定为8，权重固定
                'fixed': {
                    'NUM_MACHINES': 8,
                    'PROFIT_WEIGHT': 0.5,
                    'MAKESPAN_WEIGHT': 0.05,
                    'NUM_EPISODES': 200,
                    'SAVE_INTERVAL': 50
                },
            },
            {
                'name': '消融实验',
                'variables': ['TRAINER_TYPE', 'NUM_JOBS'],
                'fixed': {
                    'NUM_MACHINES': 8,
                    'PROFIT_WEIGHT': 0.5,
                    'MAKESPAN_WEIGHT': 0.05,
                    'NUM_EPISODES': 5000},
            }
        ]

    def _load_base_config(self, config_path: str) -> Dict:
        """加载基础配置"""
        base_config = {
            'NUM_EPISODES': 5000,
            'MAX_STEPS': 150,
            'ROLLOUT_LENGTH': 20,
            'BATCH_SIZE': 16,
            'PPO_EPOCHS': 5,
            'LEARNING_RATE': 3e-4,
            'GAMMA': 0.99,
            'LAM': 0.95,
            'PPO_CLIP_EPS': 0.2,
            'ENTROPY_COEF': 0.05,
            'VALUE_COEF': 0.5,
            'WEIGHT_DECAY': 1e-5,
            'ADAM_EPS': 1e-8,
            'GRAD_CLIP': 0.5,
            'STATE_NORMALIZATION': True,
            'SAVE_INTERVAL': 50,
            'MONITOR_INTERVAL': 50,
            'PLOT_INTERVAL': 1000,
            'NUM_STAGES': 5,
            'STATE_DIM': 128,
            'HIDDEN_DIM': 128,
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                base_config.update(user_config)
            except:
                print(f"无法加载配置文件 {config_path}，使用默认配置")

        return base_config

    def generate_experiment_configs(self, group_idx: int = None) -> List[Dict]:
        all_configs = []
        if group_idx is None:
            for group in self.experiment_groups:
                configs = self._generate_group_configs(group)
                all_configs.extend(configs)
        else:
            group = self.experiment_groups[group_idx]
            configs = self._generate_group_configs(group)
            all_configs.extend(configs)
        return all_configs

    def generate_configs_by_group_name(self, group_name: str) -> List[Dict]:
        """根据组名生成配置（新增方法）"""
        target_group = None
        for group in self.experiment_groups:
            if group['name'] == group_name:
                target_group = group
                break

        if target_group is None:
            print(f"⚠️ 未找到名为 '{group_name}' 的实验组")
            return []

        return self._generate_group_configs(target_group)

    def _generate_group_configs(self, group: Dict) -> List[Dict]:
        configs = []
        variable_names = group['variables']
        variable_values = [self.experiment_variables[name] for name in variable_names]

        for combination in itertools.product(*variable_values):
            config = self.base_config.copy()
            config.update(group['fixed'])
            for var_name, var_value in zip(variable_names, combination):
                config[var_name] = var_value

            config['JOB_ACTION_DIM'] = config['NUM_JOBS']
            config['MACHINE_ACTION_DIM'] = config['NUM_MACHINES']
            config['JOB_STATE_DIM'] = config['STATE_DIM']
            config['MACHINE_STATE_DIM'] = config['STATE_DIM']
            config['EXPERIMENT_GROUP'] = group['name']
            config['EXPERIMENT_ID'] = self._generate_experiment_id(config)
            configs.append(config)
        return configs

    def _generate_experiment_id(self, config: Dict) -> str:
        parts = []
        if config['EXPERIMENT_GROUP'] == '环境规模实验':
            parts.append(f"J{config['NUM_JOBS']}_M{config['NUM_MACHINES']}")
        elif config['EXPERIMENT_GROUP'] == '奖励权重实验':
            parts.append(f"P{config['PROFIT_WEIGHT']}_M{config['MAKESPAN_WEIGHT']}")
        elif config['EXPERIMENT_GROUP'] == '算法对比实验':
            parts.append(f"{config['TRAINER_TYPE']}")
        else:
            parts.append(f"J{config['NUM_JOBS']}_M{config['NUM_MACHINES']}")
            parts.append(f"P{config['PROFIT_WEIGHT']}")
            parts.append(f"{config['TRAINER_TYPE']}")
        return "_".join(parts)

    def save_configs(self, configs: List[Dict], output_dir: str = "experiment_configs", clear_output_dir: bool = False):
        """保存实验配置"""

        # 只有在明确要求时才清空整个目录
        if clear_output_dir and os.path.exists(output_dir):
            try:
                print(f"🧹 正在清理配置根目录: {output_dir} ...")
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"⚠️ 清理目录失败: {e}")

        os.makedirs(output_dir, exist_ok=True)

        for config in configs:
            exp_id = config['EXPERIMENT_ID']
            group = config['EXPERIMENT_GROUP']

            # 按组保存
            group_dir = os.path.join(output_dir, group.replace(' ', '_'))
            os.makedirs(group_dir, exist_ok=True)

            config_path = os.path.join(group_dir, f"{exp_id}.json")

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"保存配置: {config_path}")