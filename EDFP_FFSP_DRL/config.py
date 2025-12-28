# config.py (完整版本)
import torch
import os
import numpy as np

class Config:
    def __init__(self):
        # 设备配置
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("🎯 Disassembly Workshop Config Initialization...")

        # 简化环境参数 - 减少复杂度
        self.NUM_JOBS = 10  # 减少产品数量
        self.NUM_STAGES = 5  # 减少工序数
        self.NUM_MACHINES = 8  # 减少设备数量

        # 拆解车间特定参数
        self.STRUCTURE_COMPLETENESS_THRESHOLD = 0.4  # 结构完整性阈值
        self.CRITICAL_COMPONENT_REQUIRED = [2, 3]  # 需要关键部件的工序
        self.MAX_PRODUCT_AGE = 20.0  # 最大产品年限
        self.MIN_STRUCTURE_COMPLETENESS = 0.3  # 最小结构完整性

        # 简化状态维度计算
        # 使用固定的较小状态维度
        self.STATE_DIM = 64  # 增加状态维度以容纳新特征

        print(f"📊 固定状态维度: {self.STATE_DIM}")

        # 动作维度
        self.JOB_ACTION_DIM = self.NUM_JOBS
        self.MACHINE_ACTION_DIM = self.NUM_MACHINES
        self.JOB_STATE_DIM = self.STATE_DIM
        self.MACHINE_STATE_DIM = self.STATE_DIM

        # 简化网络参数
        self.HIDDEN_DIM = 64  # 增加隐藏层以适应更复杂的状态

        # 简化训练参数
        self.NUM_EPISODES = 20000  # 减少回合数
        self.MAX_STEPS = 500
        self.ROLLOUT_LENGTH = 20
        self.BATCH_SIZE = 16
        self.PPO_EPOCHS = 5
        self.LEARNING_RATE = 1e-4  # 更稳定的学习率

        # PPO参数
        self.GAMMA = 0.99
        self.LAM = 0.95
        self.PPO_CLIP_EPS = 0.15
        self.ENTROPY_COEF = 0.05
        self.VALUE_COEF = 0.3

        # 优化器参数
        self.WEIGHT_DECAY = 2e-5
        self.ADAM_EPS = 1e-7
        self.GRAD_CLIP = 0.3

        # 关闭状态归一化以避免问题
        self.STATE_NORMALIZATION = True

        # 模型保存和监控
        self.CHECKPOINT_DIR = "checkpoints_improved"
        self.SAVE_INTERVAL = 100
        self.SAVE_BEST = True
        self.MONITOR_INTERVAL = 50

        # 可视化参数
        self.PLOT_INTERVAL = 100  # 绘图间隔
        self.RESULTS_DIR = "training_results"  # 结果保存目录

        # 确保目录存在
        self._ensure_directories()

        print("✅ 改进配置初始化完成")

    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        print(f"📁 检查点目录: {self.CHECKPOINT_DIR}")
        print(f"📁 结果目录: {self.RESULTS_DIR}")

    def print_device_info(self):
        """打印设备信息"""
        print(f"🎯 使用设备: {self.DEVICE}")
        if torch.cuda.is_available():
            print(f"🎯 GPU型号: {torch.cuda.get_device_name()}")

    def print_config(self):
        """打印完整配置信息"""
        print("\n📋 改进配置信息:")
        print(f"   环境参数: {self.NUM_JOBS}作业, {self.NUM_STAGES}阶段, {self.NUM_MACHINES}机器")
        print(f"   状态维度: {self.STATE_DIM}")
        print(f"   动作维度: 作业={self.JOB_ACTION_DIM}, 机器={self.MACHINE_ACTION_DIM}")
        print(f"   网络参数: 隐藏层={self.HIDDEN_DIM}")
        print(f"   训练参数: {self.NUM_EPISODES}回合")
        print(f"   学习率: {self.LEARNING_RATE}")
        print(f"   拆解车间参数:")
        print(f"     - 结构完整性阈值: {self.STRUCTURE_COMPLETENESS_THRESHOLD}")
        print(f"     - 需要关键部件的工序: {self.CRITICAL_COMPONENT_REQUIRED}")
        print(f"     - 最大产品年限: {self.MAX_PRODUCT_AGE}")