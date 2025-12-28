import matplotlib.pyplot as plt
import numpy as np

# 加载训练历史数据
with open('training_results/training_history.txt', 'r') as f:
    lines = f.readlines()

# 解析利润数据
profits = []
for line in lines:
    if 'total_profit:' in line:
        values = line.split(':')[1].strip().split(',')
        profits = [float(v.strip()) for v in values if v.strip()]
        break

if profits:
    # 绘制利润曲线
    plt.figure(figsize=(12, 6))

    # 原始曲线
    plt.subplot(1, 2, 1)
    plt.plot(profits, alpha=0.5, label='Raw Profit')

    # 计算100回合移动平均
    window = min(100, len(profits))
    ma = np.convolve(profits, np.ones(window) / window, mode='valid')
    plt.plot(range(window - 1, len(profits)), ma, 'r-', linewidth=2,
             label=f'{window}-episode MA')

    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.title('Profit Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最后1000回合的详细视图
    plt.subplot(1, 2, 2)
    last_n = min(1000, len(profits))
    plt.plot(range(len(profits) - last_n, len(profits)),
             profits[-last_n:], 'b-', alpha=0.7)
    plt.xlabel('Last 1000 Episodes')
    plt.ylabel('Profit')
    plt.title(f'Last {last_n} Episodes (Final Convergence)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 计算收敛指标
    last_quarter = profits[-len(profits) // 4:]  # 最后1/4
    first_quarter = profits[:len(profits) // 4]  # 前1/4

    improvement = (np.mean(last_quarter) - np.mean(first_quarter)) / abs(np.mean(first_quarter)) * 100
    stability = np.std(last_quarter) / np.mean(last_quarter) * 100  # 变异系数

    print(f"📊 收敛分析报告:")
    print(f"   训练回合数: {len(profits)}")
    print(f"   平均利润: {np.mean(profits):.2f}")
    print(f"   最终1/4平均利润: {np.mean(last_quarter):.2f}")
    print(f"   初期1/4平均利润: {np.mean(first_quarter):.2f}")
    print(f"   利润提升: {improvement:.1f}%")
    print(f"   后期稳定性(变异系数): {stability:.1f}%")
    print(f"\n   收敛判断:")
    if stability < 5 and improvement > 10:
        print("   ✅ 训练已良好收敛")
    elif stability < 10 and improvement > 0:
        print("   ⚠️  训练基本收敛，但可继续优化")
    else:
        print("   ❌ 训练未收敛，需要调整")
else:
    print("未找到利润数据")