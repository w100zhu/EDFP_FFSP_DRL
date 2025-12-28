# main.py (最终完整版 - 支持算法切换、自动路径与维度适配)
import torch
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from collections import deque
import traceback

# 尝试导入pandas，如果失败则使用替代方案
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ pandas库未安装，将使用简化数据保存功能")
    PANDAS_AVAILABLE = False


    # 定义简单的替代函数
    class SimpleDataSaver:
        def __init__(self):
            self.data = {}

        def save_data(self, filename, data_dict):
            """简化版数据保存"""
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    for key, values in data_dict.items():
                        f.write(f"{key}: {','.join(map(str, values))}\n")
                print(f"💾 数据已保存: {filename}")
            except Exception as e:
                print(f"❌ 保存数据失败: {e}")

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# 确保能导入 EDFP_FFSP_DRL 包 (如果你的项目结构需要)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def debug_imports():
    """调试所有导入"""
    print("🔍 调试模块导入...")
    try:
        import config
        print("✅ config模块导入成功")
    except ImportError:
        # 尝试从子包导入
        try:
            from EDFP_FFSP_DRL import config
            print("✅ EDFP_FFSP_DRL.config模块导入成功")
        except Exception as e:
            print(f"❌ config模块导入失败: {e}")
            return False
    except Exception as e:
        print(f"❌ config模块导入失败: {e}")
        return False
    return True


def create_simple_config():
    """创建简化配置用于调试"""

    class SimpleConfig:
        def __init__(self):
            self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.NUM_JOBS = 5
            self.NUM_STAGES = 5
            self.NUM_MACHINES = 8
            self.STATE_DIM = 64  # 将被自动覆盖
            self.JOB_ACTION_DIM = self.NUM_JOBS
            self.MACHINE_ACTION_DIM = self.NUM_MACHINES
            self.JOB_STATE_DIM = self.STATE_DIM
            self.MACHINE_STATE_DIM = self.STATE_DIM
            self.HIDDEN_DIM = 128

            # 强化训练参数
            self.NUM_EPISODES = 20000
            self.MAX_STEPS = 150
            self.ROLLOUT_LENGTH = 20
            self.BATCH_SIZE = 64
            self.PPO_EPOCHS = 10
            self.LEARNING_RATE = 3e-4
            self.GAMMA = 0.99
            self.LAM = 0.95
            self.PPO_CLIP_EPS = 0.2
            self.ENTROPY_COEF = 0.02
            self.VALUE_COEF = 0.5
            self.WEIGHT_DECAY = 1e-5
            self.ADAM_EPS = 1e-8
            self.GRAD_CLIP = 0.5
            self.STATE_NORMALIZATION = True

            self.CHECKPOINT_DIR = "checkpoints_debug"
            self.SAVE_INTERVAL = 500
            self.SAVE_BEST = True
            self.MONITOR_INTERVAL = 50
            self.PLOT_INTERVAL = 1000
            self.RESULTS_DIR = "training_results"

            os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(self.RESULTS_DIR, exist_ok=True)

        def print_device_info(self):
            print(f"🎯 使用设备: {self.DEVICE}")

        def print_config(self):
            print("简化配置已加载...")

    return SimpleConfig()


# ======================================================================================
# 【核心功能】通用模型保存函数 (适配 run_comparison.py 的读取逻辑)
# ======================================================================================
def save_checkpoint(job_agent, machine_agent, episode, config, algo_type="Integrated", is_best=False):
    """保存模型检查点，支持 Integrated(单文件) 和 BasePPO(双文件)"""
    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === 模式 A: Integrated (单文件) ===
    if algo_type == "Integrated":
        checkpoint_data = {
            'episode': episode,
            # 保存 Actor/Critic 状态字典
            'job_actor_state': job_agent.actor.state_dict(),
            'job_critic_state': job_agent.critic.state_dict(),
            'job_optimizer_state': job_agent.optimizer.state_dict(),

            'machine_actor_state': machine_agent.actor.state_dict(),
            'machine_critic_state': machine_agent.critic.state_dict(),
            'machine_optimizer_state': machine_agent.optimizer.state_dict(),

            'config': str(config)
        }

        # 命名规则：best_checkpoint.pth 或 checkpoint_XXX.pth
        filename = 'best_checkpoint.pth' if is_best else f'checkpoint_{episode}.pth'
        filepath = os.path.join(checkpoint_dir, filename)

        try:
            torch.save(checkpoint_data, filepath)
            if is_best:
                print(f"💾 [Integrated] 已保存最佳模型: {filepath}")
            else:
                print(f"💾 [Integrated] 已保存检查点: {filename}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

    # === 模式 B: BasePPO (双文件) ===
    elif algo_type == "BasePPO":
        # 命名规则必须匹配 run_comparison.py: job_agent_base_XXX.pth
        prefix_job = 'best_job_agent_base' if is_best else f'job_agent_base_{episode}'
        prefix_machine = 'best_machine_agent_base' if is_best else f'machine_agent_base_{episode}'

        path_job = os.path.join(checkpoint_dir, f'{prefix_job}.pth')
        path_machine = os.path.join(checkpoint_dir, f'{prefix_machine}.pth')

        try:
            # 保存 Job Agent
            torch.save({
                'actor_state_dict': job_agent.actor.state_dict(),
                'critic_state_dict': job_agent.critic.state_dict(),
                'optimizer_state_dict': job_agent.optimizer.state_dict()
            }, path_job)

            # 保存 Machine Agent
            torch.save({
                'actor_state_dict': machine_agent.actor.state_dict(),
                'critic_state_dict': machine_agent.critic.state_dict(),
                'optimizer_state_dict': machine_agent.optimizer.state_dict()
            }, path_machine)

            print(f"💾 [BasePPO] 已保存双文件: {prefix_job}.pth & {prefix_machine}.pth")
        except Exception as e:
            print(f"❌ 保存失败: {e}")


def check_and_adjust_state_dimension(env, config):
    """
    运行一次环境重置，探测真实状态维度，并更新配置。
    防止维度不匹配错误。
    """
    print("\n📊 正在探测真实状态维度...")
    state = env.reset()

    # 兼容 Tensor 和 dict 返回
    if isinstance(state, dict):
        # 假设我们用 combined_state 或者第一个 value
        if 'combined_state' in state:
            state_data = state['combined_state']
        else:
            state_data = list(state.values())[0]
        actual_state_dim = state_data.shape[0] if hasattr(state_data, 'shape') else len(state_data)
    elif hasattr(state, 'shape'):
        actual_state_dim = state.shape[0]
    else:
        actual_state_dim = len(state)

    print(f"   Config预设维度: {config.STATE_DIM}")
    print(f"   环境真实维度: {actual_state_dim}")

    if actual_state_dim != config.STATE_DIM:
        print(f"🔄 维度不匹配，正在动态更新配置 ({config.STATE_DIM} -> {actual_state_dim})...")
        config.STATE_DIM = actual_state_dim
        config.JOB_STATE_DIM = actual_state_dim
        config.MACHINE_STATE_DIM = actual_state_dim
        return True

    return False


def plot_training_progress(training_history, config, episode):
    """绘制简单的训练曲线"""
    if episode % config.PLOT_INTERVAL != 0 and episode != config.NUM_EPISODES - 1:
        return
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(training_history['rewards'], label='Reward')
        if training_history['profits']:
            # 归一化显示以便同框
            profits = np.array(training_history['profits'])
            plt.plot(profits / (np.max(profits) + 1e-6) * np.max(training_history['rewards']),
                     label='Profit (Scaled)', alpha=0.5)

        plt.title(f'Episode {episode} Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(config.RESULTS_DIR, f'training_progress_episode_{episode}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 训练曲线已保存: {plot_path}")
    except Exception as e:
        print(f"❌ 绘图失败: {e}")


def save_training_history(training_history, config):
    """保存训练历史数据 CSV"""
    try:
        if PANDAS_AVAILABLE:
            # 确保所有列长度一致
            min_len = min(len(v) for k, v in training_history.items() if isinstance(v, list))
            data = {k: v[:min_len] for k, v in training_history.items() if isinstance(v, list)}

            df = pd.DataFrame(data)
            csv_path = os.path.join(config.RESULTS_DIR, 'training_history.csv')
            df.to_csv(csv_path, index=False)
            print(f"💾 训练历史CSV已保存: {csv_path}")
        else:
            print("⚠️ Pandas不可用，跳过CSV保存")
    except Exception as e:
        print(f"❌ 保存历史失败: {e}")


# ======================================================================================
# 主程序入口
# ======================================================================================
def main():
    print("🚀 启动退役产品拆解车间调度系统 (DFFSP-DRL)...")
    print(f"📁 工作目录: {os.getcwd()}")
    print(f"🔥 PyTorch版本: {torch.__version__} (CUDA: {torch.cuda.is_available()})")

    # =================【步骤1：在此处设置算法类型】=================
    # 选项: "Integrated" (推荐) 或 "BasePPO"
    ALGO_TYPE = "Integrated"
    # =============================================================

    print(f"\n🎯 当前选定算法: {ALGO_TYPE}")

    # 初始化数据记录
    training_history = {
        'rewards': [], 'profits': [], 'makespans': [], 'steps': [],
        'job_losses': [], 'machine_losses': [],
        # 'total_time': 0 # 单独处理
    }
    stats_50_history = {
        'episode_checkpoints': [], 'avg_profits': [], 'avg_makespans': []
    }

    try:
        # 1. 加载配置
        try:
            # 尝试导入项目中的 config
            try:
                from EDFP_FFSP_DRL.config import Config
            except ImportError:
                from config import Config
            config_instance = Config()
            print("✅ 已加载正式配置")
        except ImportError:
            print("⚠️ 无法加载正式配置，使用简化配置")
            config_instance = create_simple_config()

        # 2. 增强配置（为了更好的训练效果）
        if hasattr(config_instance, 'NUM_EPISODES'):
            # 如果配置里的太小，强制增大，防止训练不足
            if config_instance.NUM_EPISODES < 5000:
                print(f"⚠️ 检测到训练回合数过少 ({config_instance.NUM_EPISODES})，强制提升至 20000")
                config_instance.NUM_EPISODES = 20000

        # =================【步骤2：自动生成对比脚本所需的路径】=================
        # 路径格式: experiments/checkpoints/全面实验/J{Jobs}_M{Machines}_P{Profit}_{Algo}
        group_name = "全面实验"
        profit_weight = 0.5  # 假设默认利润权重为 0.5

        folder_name = f"J{config_instance.NUM_JOBS}_M{config_instance.NUM_MACHINES}_P{profit_weight}_{ALGO_TYPE}"

        # 寻找项目根目录 (假设 main.py 在项目根目录或其子目录下)
        # 这里我们假设 main.py 在项目根目录，或者在 EDFP_FFSP_DRL 内部
        # 我们向上找直到找到 experiments 文件夹或者在当前目录创建
        project_root = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(project_root) == 'EDFP_FFSP_DRL':  # 如果在包内
            project_root = os.path.dirname(project_root)

        checkpoint_root = os.path.join(project_root, "EDFP_FFSP_DRL", "experiments", "checkpoints") \
            if os.path.exists(os.path.join(project_root, "EDFP_FFSP_DRL")) else os.path.join(project_root,
                                                                                             "experiments",
                                                                                             "checkpoints")

        # 最终保存路径
        new_checkpoint_dir = os.path.join(checkpoint_root, group_name, folder_name)

        # 结果目录 (分开存放以免混淆)
        results_root = os.path.join(project_root, "training_results")
        new_results_dir = os.path.join(results_root, folder_name)

        # 覆盖配置
        config_instance.CHECKPOINT_DIR = new_checkpoint_dir
        config_instance.RESULTS_DIR = new_results_dir

        os.makedirs(config_instance.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config_instance.RESULTS_DIR, exist_ok=True)

        print(f"\n📂 模型保存路径: {config_instance.CHECKPOINT_DIR}")
        print(f"📂 结果保存路径: {config_instance.RESULTS_DIR}")
        print(f"ℹ️  参数: Jobs={config_instance.NUM_JOBS}, Machines={config_instance.NUM_MACHINES}")
        # ====================================================================

        # 3. 创建环境
        print("\n🏭 初始化环境...")
        try:
            from EDFP_FFSP_DRL.environment.dffsp_env import DFFSPEnvironment
        except ImportError:
            from environment.dffsp_env import DFFSPEnvironment

        env = DFFSPEnvironment(
            num_jobs=config_instance.NUM_JOBS,
            num_stages=config_instance.NUM_STAGES,
            num_machines=config_instance.NUM_MACHINES,
            config=config_instance
        )

        # 4. 维度探测与配置更新
        dimension_changed = check_and_adjust_state_dimension(env, config_instance)

        # 5. 初始化智能体
        print("\n🤖 初始化智能体...")
        try:
            from EDFP_FFSP_DRL.agents.job_agent import JobAgent
            from EDFP_FFSP_DRL.agents.machine_agent import MachineAgent
        except ImportError:
            from agents.job_agent import JobAgent
            from agents.machine_agent import MachineAgent

        job_agent = JobAgent(config_instance)
        machine_agent = MachineAgent(config_instance)
        print("✅ 智能体初始化完成")

        # 6. 初始化训练器 (根据 ALGO_TYPE)
        print(f"\n🎯 初始化 {ALGO_TYPE} 训练器...")
        if ALGO_TYPE == "Integrated":
            try:
                from EDFP_FFSP_DRL.training.integrated_trainer import IntegratedTrainer
            except ImportError:
                from training.integrated_trainer import IntegratedTrainer
            trainer = IntegratedTrainer(config_instance, job_agent, machine_agent, env)

        elif ALGO_TYPE == "BasePPO":
            try:
                from EDFP_FFSP_DRL.training.ippo_trainer import IPPOTrainer
            except ImportError:
                from training.ippo_trainer import IPPOTrainer
            trainer = IPPOTrainer(config_instance, job_agent, machine_agent, env)

        else:
            raise ValueError(f"未知的算法类型: {ALGO_TYPE}")
        print(f"✅ {ALGO_TYPE} 训练器就绪")

        # 7. 开始训练循环
        print(f"\n🚀 开始训练循环 (目标: {config_instance.NUM_EPISODES} 回合)...")
        best_reward = -float('inf')
        start_time = time.time()

        recent_rewards = deque(maxlen=50)
        recent_profits = deque(maxlen=50)
        recent_makespans = deque(maxlen=50)

        for episode in range(config_instance.NUM_EPISODES):
            try:
                # 更新课程学习进度 (如果环境支持)
                if hasattr(env, 'update_training_progress'):
                    env.update_training_progress(episode, config_instance.NUM_EPISODES)

                # 收集经验
                result = trainer.collect_rollout()

                # 处理异常返回
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    # print(f"⚠️ Ep {episode}: Rollout收集失败，跳过")
                    continue

                batch, episode_reward, steps, _ = result

                # 更新网络
                if batch:
                    job_loss, machine_loss = trainer.update_agents(batch)
                else:
                    job_loss, machine_loss = 0.0, 0.0

                # 记录数据
                training_history['rewards'].append(episode_reward)
                training_history['steps'].append(steps)
                training_history['job_losses'].append(job_loss)
                training_history['machine_losses'].append(machine_loss)

                # 获取环境指标
                current_profit = env.total_profit if hasattr(env, 'total_profit') else 0
                current_makespan = env._calculate_makespan() if hasattr(env, '_calculate_makespan') else 0

                training_history['profits'].append(current_profit)
                training_history['makespans'].append(current_makespan)

                # 更新最佳模型
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    if config_instance.SAVE_BEST:
                        save_checkpoint(job_agent, machine_agent, episode, config_instance,
                                        algo_type=ALGO_TYPE, is_best=True)

                # 监控日志
                recent_rewards.append(episode_reward)
                recent_profits.append(current_profit)
                recent_makespans.append(current_makespan)

                if episode % config_instance.MONITOR_INTERVAL == 0:
                    avg_rew = np.mean(recent_rewards)
                    avg_prof = np.mean(recent_profits)
                    avg_mksp = np.mean(recent_makespans)

                    elapsed = time.time() - start_time
                    fps = (episode + 1) / elapsed

                    print(f"📈 Ep {episode}/{config_instance.NUM_EPISODES} | "
                          f"Rew: {avg_rew:.2f} | Prof: {avg_prof:.1f} | Mksp: {avg_mksp:.1f} | "
                          f"Best: {best_reward:.2f} | FPS: {fps:.1f}")

                    # 记录统计
                    stats_50_history['episode_checkpoints'].append(episode)
                    stats_50_history['avg_profits'].append(avg_prof)
                    stats_50_history['avg_makespans'].append(avg_mksp)

                # 定期保存检查点
                if episode % config_instance.SAVE_INTERVAL == 0 and episode > 0:
                    save_checkpoint(job_agent, machine_agent, episode, config_instance,
                                    algo_type=ALGO_TYPE, is_best=False)

                # 绘图
                if episode % config_instance.PLOT_INTERVAL == 0 and episode > 0:
                    plot_training_progress(training_history, config_instance, episode)

            except KeyboardInterrupt:
                print("\n🛑 用户中断训练")
                break
            except Exception as e:
                print(f"❌ Ep {episode} 异常: {e}")
                traceback.print_exc()
                continue

        # 8. 训练结束处理
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成! 总耗时: {total_time:.1f}s")

        # 保存最终模型
        save_checkpoint(job_agent, machine_agent, config_instance.NUM_EPISODES - 1,
                        config_instance, algo_type=ALGO_TYPE, is_best=False)

        # 保存历史数据
        save_training_history(training_history, config_instance)

        # 保存统计摘要
        try:
            stats_path = os.path.join(config_instance.RESULTS_DIR, 'stats_50_episodes.csv')
            pd.DataFrame(stats_50_history).to_csv(stats_path, index=False)
            print(f"💾 统计摘要已保存: {stats_path}")
        except:
            pass

        return 0

    except Exception as e:
        print(f"❌ 主程序崩溃: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())