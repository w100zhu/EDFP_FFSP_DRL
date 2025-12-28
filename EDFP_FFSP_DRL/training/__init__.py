"""
退役产品拆解车间DRL训练模块
"""

from .base_ppo_trainer import BasePPOTrainer
from .balanced_ppo_trainer import BalancedPPOTrainer
from .curriculum_trainer import CurriculumTrainer
from .multihead_trainer import MultiHeadTrainer

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    'BasePPOTrainer',
    'BalancedPPOTrainer',
    'CurriculumTrainer',
    'MultiHeadTrainer'
]