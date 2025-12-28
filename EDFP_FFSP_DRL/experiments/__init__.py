"""
退役产品拆解车间DRL训练模块
"""

"""
退役产品拆解车间DRL实验模块

这个模块包含：
1. 实验配置生成器
2. 批量实验运行器
3. 结果分析器
4. Pareto前沿分析器
5. 综合实验运行脚本
"""

from .experiment_config_generator import ExperimentConfigGenerator
from .batch_experiment_runner import BatchExperimentRunner
from .analyze_results import ExperimentAnalyzer
from .pareto_analysis import ParetoAnalyzer

__version__ = "1.0.0"
__author__ = "Your Name"

# 提供快捷导入
__all__ = [
    'ExperimentConfigGenerator',
    'BatchExperimentRunner',
    'ExperimentAnalyzer',
    'ParetoAnalyzer',
    'run_experiments',
    'analyze_results'
]

# 注意：由于 run_experiments.py 中的 main 函数不是可导入的类，
# 我们不在 __init__.py 中直接导入它，但提供了运行函数

def run_experiments():
    """运行所有实验"""
    from EDFP_FFSP_DRL.experiments.run_experiments import main
    main()

def analyze_results():
    """分析实验结果"""
    from .analyze_results import main
    main()

def generate_pareto_analysis():
    """生成Pareto分析报告"""
    from .pareto_analysis import main
    main()