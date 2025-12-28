import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict


def compute_advantages(rewards: List[float], values: List[float],
                       dones: List[bool], gamma: float = 0.99,
                       lam: float = 0.95) -> List[float]:
    """
    计算广义优势估计 (GAE)
    """
    advantages = []
    gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_not_done = 1.0 - dones[t]
            next_value = values[t] * next_not_done
        else:
            next_not_done = 1.0 - dones[t]
            next_value = values[t + 1] * next_not_done

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * next_not_done * gae
        advantages.insert(0, gae)

    return advantages


def normalize_advantages(advantages: List[float]) -> np.ndarray:
    """标准化优势函数"""
    advantages = np.array(advantages)
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def plot_training_curves(rewards: List[float], losses: List[float],
                         window_size: int = 100) -> None:
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 奖励曲线
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        ax1.plot(moving_avg)
        ax1.set_title(f'平均奖励 (移动平均, 窗口大小={window_size})')
    else:
        ax1.plot(rewards)
        ax1.set_title('训练奖励')

    ax1.set_ylabel('奖励')
    ax1.grid(True)

    # 损失曲线
    if losses:
        ax2.plot(losses)
        ax2.set_title('训练损失')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('损失')
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


def save_checkpoint(agent, optimizer, episode: int, filepath: str) -> None:
    """保存训练检查点"""
    torch.save({
        'episode': episode,
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(agent, optimizer, filepath: str) -> int:
    """加载训练检查点"""
    checkpoint = torch.load(filepath)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode']