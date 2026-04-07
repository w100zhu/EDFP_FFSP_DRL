# experiments/pareto_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import os
import json
import glob
import re
from scipy.spatial import ConvexHull
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class ParetoAnalyzer:
    """
    Pareto前沿分析器
    用于分析多目标优化问题的Pareto最优解
    支持从训练曲线文件加载数据
    """

    def __init__(self,
                 data_source: str = None,
                 results_dir: str = "experiment_results",
                 data_file: str = None):
        """
        初始化分析器

        参数:
            data_source: 数据源，可选:
                - 'training_curves': 训练曲线CSV文件
                - 'experiment_summary': 实验摘要CSV文件
                - None: 自动检测
            results_dir: 结果目录
            data_file: 单个数据文件路径（如果提供）
        """
        self.results_dir = results_dir
        self.data_source = data_source
        self.data_file = data_file
        self.df = self._load_experiment_data()
        # 存储各组Pareto点的字典
        self.group_pareto_points = {}

    def _load_experiment_data(self) -> pd.DataFrame:
        """
        加载实验数据
        """
        print(f"📂 加载实验数据...")

        # 如果有指定的数据文件，直接加载
        if self.data_file and os.path.exists(self.data_file):
            print(f"  从指定文件加载: {self.data_file}")
            return self._load_single_file(self.data_file)

        # 检查数据源类型
        if self.data_source == 'training_curves' or self._detect_training_curves():
            print(f"🔍 检测到训练曲线数据")
            return self._load_from_training_curves()
        elif self.data_source == 'experiment_summary' or self._detect_experiment_summary():
            print(f"🔍 检测到实验摘要数据")
            return self._load_from_experiment_summary()
        else:
            print(f"⚠️  无法自动检测数据源，尝试从目录加载所有CSV文件")
            return self._load_all_csv_files()

    def _detect_training_curves(self) -> bool:
        """检测是否有训练曲线数据"""
        # 检查是否有training_curves文件
        curve_files = glob.glob(os.path.join(self.results_dir, "**/*training_curves*.csv"), recursive=True)
        return len(curve_files) > 0

    def _detect_experiment_summary(self) -> bool:
        """检测是否有实验摘要数据"""
        # 检查是否有experiment_summary文件
        summary_files = glob.glob(os.path.join(self.results_dir, "**/*experiment_summary*.csv"), recursive=True)
        return len(summary_files) > 0

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """加载单个文件"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ 成功加载文件: {file_path}")
            print(f"   数据形状: {df.shape}")
            print(f"   列名: {list(df.columns)}")

            # 标准化列名
            df = self._standardize_column_names(df)

            # 数据验证
            df = self._validate_and_filter_data(df)

            return df
        except Exception as e:
            print(f"❌ 加载文件失败: {file_path}")
            print(f"   错误: {e}")
            return pd.DataFrame()

    def _load_from_training_curves(self) -> pd.DataFrame:
        """
        从训练曲线文件中加载数据
        提取每个实验的最终性能指标
        """
        print(f"🔍 搜索训练曲线文件...")

        # 查找所有训练曲线文件
        curve_files = glob.glob(os.path.join(self.results_dir, "**/*training_curves*.csv"), recursive=True)
        curve_files.extend(glob.glob(os.path.join(self.results_dir, "**/*train_curve*.csv"), recursive=True))
        curve_files.extend(glob.glob(os.path.join(self.results_dir, "**/*curve*.csv"), recursive=True))

        if not curve_files:
            print(f"❌ 未找到训练曲线文件")
            return pd.DataFrame()

        print(f"✅ 找到 {len(curve_files)} 个训练曲线文件")

        all_experiments = []

        for file_path in curve_files:
            try:
                # 从文件名提取实验信息
                file_name = os.path.basename(file_path)
                dir_name = os.path.basename(os.path.dirname(file_path))

                print(f"  处理文件: {file_name}")

                # 加载数据
                curve_df = pd.read_csv(file_path)

                if curve_df.empty:
                    print(f"  ⚠️  文件为空，跳过: {file_name}")
                    continue

                # 提取最终性能指标（最后N个episode的平均值）
                final_performance = self._extract_final_performance(curve_df, file_name, dir_name)

                if final_performance is not None:
                    all_experiments.append(final_performance)
                    print(f"  ✅ 提取成功: {len(curve_df)} 个数据点")

            except Exception as e:
                print(f"  ❌ 处理文件失败: {file_name}")
                print(f"     错误: {e}")
                continue

        if not all_experiments:
            print(f"❌ 未能从任何文件中提取数据")
            return pd.DataFrame()

        # 合并所有实验数据
        combined_df = pd.DataFrame(all_experiments)

        print(f"\n📊 合并数据统计:")
        print(f"   实验数量: {len(combined_df)}")
        print(f"   数据列: {list(combined_df.columns)}")

        return combined_df

    def _extract_final_performance(self, curve_df: pd.DataFrame, file_name: str, dir_name: str) -> Optional[Dict]:
        """
        从训练曲线中提取最终性能指标

        参数:
            curve_df: 训练曲线DataFrame
            file_name: 文件名
            dir_name: 目录名

        返回:
            包含最终性能指标的字典
        """
        if curve_df.empty:
            return None

        # 确定列名（处理不同的列名格式）
        profit_col = None
        makespan_col = None
        profit_rate_col = None

        # 查找利润列
        for col in ['profit', 'avg_test_profit', 'test_profit', 'avg_profit']:
            if col in curve_df.columns:
                profit_col = col
                break

        # 查找完工时长列
        for col in ['makespan', 'avg_test_makespan', 'test_makespan', 'avg_makespan']:
            if col in curve_df.columns:
                makespan_col = col
                break

        # 查找利润率列
        for col in ['profit_rate', 'profit_ratio', 'profit_margin']:
            if col in curve_df.columns:
                profit_rate_col = col
                break

        if not profit_col or not makespan_col:
            print(f"  ⚠️  缺少关键列: profit={profit_col}, makespan={makespan_col}")
            print(f"     可用列: {list(curve_df.columns)}")
            return None

        # 计算最终性能指标（最后100个episode或最后10%的数据点）
        n_last = min(100, len(curve_df) // 10)
        if n_last < 10:
            n_last = len(curve_df)  # 如果数据点太少，使用所有点

        # 获取最后n_last个数据点
        last_data = curve_df.tail(n_last)

        # 计算平均性能
        avg_profit = last_data[profit_col].mean()
        avg_makespan = last_data[makespan_col].mean()

        # 如果完工时长<=0，跳过这个实验
        if avg_makespan <= 0:
            print(f"  ⚠️  无效的完工时长: {avg_makespan:.2f}，跳过")
            return None

        # 计算单位时间利润
        profit_per_time = avg_profit / avg_makespan

        # 如果有利润率列，使用它
        if profit_rate_col and profit_rate_col in last_data.columns:
            avg_profit_rate = last_data[profit_rate_col].mean()
        else:
            avg_profit_rate = profit_per_time

        # 构建实验数据
        experiment_data = {
            'experiment_id': file_name.replace('.csv', ''),
            'experiment_group': dir_name,
            'avg_test_profit': avg_profit,
            'avg_test_makespan': avg_makespan,
            'profit_per_time': profit_per_time,
            'profit_rate': avg_profit_rate,
            'source_file': file_name,
            'total_episodes': len(curve_df),
            'last_n_episodes': n_last,
            'min_profit': curve_df[profit_col].min(),
            'max_profit': curve_df[profit_col].max(),
            'min_makespan': curve_df[makespan_col].min(),
            'max_makespan': curve_df[makespan_col].max(),
        }

        # 从文件名中提取可能的配置信息
        self._extract_config_from_filename(file_name, experiment_data)

        return experiment_data

    def _extract_config_from_filename(self, file_name: str, experiment_data: Dict):
        """从文件名中提取配置信息"""
        # 尝试提取权重信息（例如：P0.1_M0.05）
        weight_pattern = r'[Pp](\d+\.?\d*)[_\-]?[Mm](\d+\.?\d*)'
        match = re.search(weight_pattern, file_name)

        if match:
            experiment_data['profit_weight'] = float(match.group(1))
            experiment_data['makespan_weight'] = float(match.group(2))

        # 尝试提取算法类型
        algorithm_keywords = ['ppo', 'a2c', 'dqn', 'td3', 'sac', 'integrated']
        for keyword in algorithm_keywords:
            if keyword.lower() in file_name.lower():
                experiment_data['trainer_type'] = keyword.upper()
                break

        # 尝试提取环境规模
        job_pattern = r'[Jj](\d+)'
        machine_pattern = r'[Mm](\d+)'

        job_match = re.search(job_pattern, file_name)
        machine_match = re.search(machine_pattern, file_name)

        if job_match:
            experiment_data['num_jobs'] = int(job_match.group(1))
        if machine_match and 'makespan_weight' not in experiment_data:  # 避免重复
            experiment_data['num_machines'] = int(machine_match.group(1))

    def _load_from_experiment_summary(self) -> pd.DataFrame:
        """从实验摘要文件加载数据"""
        summary_files = glob.glob(os.path.join(self.results_dir, "**/*experiment_summary*.csv"), recursive=True)

        if not summary_files:
            print(f"❌ 未找到实验摘要文件")
            return pd.DataFrame()

        print(f"✅ 找到 {len(summary_files)} 个实验摘要文件")

        dfs = []
        for file_path in summary_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    dfs.append(df)
                    print(f"  加载: {os.path.basename(file_path)} ({len(df)} 行)")
            except Exception as e:
                print(f"  ❌ 加载失败: {os.path.basename(file_path)}")
                continue

        if not dfs:
            return pd.DataFrame()

        # 合并所有DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # 标准化列名
        combined_df = self._standardize_column_names(combined_df)

        # 数据验证和过滤
        combined_df = self._validate_and_filter_data(combined_df)

        return combined_df

    def _load_all_csv_files(self) -> pd.DataFrame:
        """加载目录中的所有CSV文件"""
        csv_files = glob.glob(os.path.join(self.results_dir, "**/*.csv"), recursive=True)

        if not csv_files:
            print(f"❌ 未找到任何CSV文件")
            return pd.DataFrame()

        print(f"✅ 找到 {len(csv_files)} 个CSV文件")

        # 尝试识别并加载数据
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            print(f"  尝试: {file_name}")

            try:
                df = pd.read_csv(file_path)

                if df.empty:
                    continue

                # 检查是否包含必要的列
                has_profit = any(col in df.columns for col in ['profit', 'avg_test_profit'])
                has_makespan = any(col in df.columns for col in ['makespan', 'avg_test_makespan'])

                if has_profit and has_makespan:
                    print(f"  🔍 找到有效数据文件: {file_name}")

                    # 如果是训练曲线文件
                    if 'episode' in df.columns:
                        print(f"  📈 识别为训练曲线文件")
                        # 从单个文件创建实验数据
                        final_performance = self._extract_final_performance(df, file_name, os.path.basename(
                            os.path.dirname(file_path)))

                        if final_performance:
                            result_df = pd.DataFrame([final_performance])
                            result_df = self._validate_and_filter_data(result_df)
                            return result_df
                    else:
                        # 实验摘要文件
                        print(f"  📊 识别为实验摘要文件")
                        df = self._standardize_column_names(df)
                        df = self._validate_and_filter_data(df)
                        return df

            except Exception as e:
                continue

        print(f"❌ 未能从任何CSV文件中提取有效数据")
        return pd.DataFrame()

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        df = df.copy()

        # 重命名列以保持一致性
        column_mapping = {
            'group': 'experiment_group',
            'profit': 'avg_test_profit',
            'makespan': 'avg_test_makespan',
            'test_profit': 'avg_test_profit',
            'test_makespan': 'avg_test_makespan',
            'avg_profit': 'avg_test_profit',
            'avg_makespan': 'avg_test_makespan'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # 确保有experiment_group列
        if 'experiment_group' not in df.columns:
            if 'experiment_id' in df.columns:
                df['experiment_group'] = df['experiment_id'].apply(
                    lambda x: str(x).split('_')[0] if '_' in str(x) else 'default')
            else:
                df['experiment_group'] = 'default_group'

        return df

    def _validate_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证和过滤数据"""
        if df.empty:
            return df

        print(f"\n🔍 数据验证:")
        print(f"   原始数据行数: {len(df)}")

        # 检查关键列是否存在
        required_cols = ['avg_test_profit', 'avg_test_makespan']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"❌ 缺失关键列: {missing_cols}")
            print(f"   可用列: {list(df.columns)}")

            # 尝试查找替代列名
            for col in ['profit', 'test_profit', 'avg_profit']:
                if col in df.columns and 'avg_test_profit' not in df.columns:
                    df['avg_test_profit'] = df[col]
                    print(f"  ✅ 使用 '{col}' 作为利润列")
                    break

            for col in ['makespan', 'test_makespan', 'avg_makespan']:
                if col in df.columns and 'avg_test_makespan' not in df.columns:
                    df['avg_test_makespan'] = df[col]
                    print(f"  ✅ 使用 '{col}' 作为完工时长列")
                    break

        # 重新检查关键列
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 仍然缺失关键列: {missing_cols}")
            return pd.DataFrame()

        # 统计初始数据
        initial_count = len(df)
        profit_zero_count = (df['avg_test_profit'] == 0).sum()
        makespan_zero_count = (df['avg_test_makespan'] == 0).sum()
        makespan_negative_count = (df['avg_test_makespan'] < 0).sum()

        print(f"   利润为0的数据点: {profit_zero_count}/{initial_count}")
        print(f"   完工时长为0的数据点: {makespan_zero_count}/{initial_count}")
        print(f"   完工时长为负的数据点: {makespan_negative_count}/{initial_count}")

        # 过滤掉完工时长为0或负的数据点
        valid_mask = df['avg_test_makespan'] > 0
        filtered_df = df[valid_mask].copy()

        removed_count = initial_count - len(filtered_df)
        print(f"\n✅ 数据过滤完成:")
        print(f"   移除无效数据点（完工时长<=0）: {removed_count}")
        print(f"   剩余有效数据点: {len(filtered_df)}")

        if len(filtered_df) == 0:
            print(f"❌ 警告：过滤后没有有效数据点")
            return pd.DataFrame()

        # 计算单位时间利润
        filtered_df['profit_per_time'] = filtered_df['avg_test_profit'] / filtered_df['avg_test_makespan']

        # 显示有效数据的统计信息
        print(f"\n📊 有效数据统计:")
        print(f"   利润范围: [{filtered_df['avg_test_profit'].min():.2f}, {filtered_df['avg_test_profit'].max():.2f}]")
        print(
            f"   完工时长范围: [{filtered_df['avg_test_makespan'].min():.2f}, {filtered_df['avg_test_makespan'].max():.2f}]")
        print(
            f"   单位时间利润范围: [{filtered_df['profit_per_time'].min():.4f}, {filtered_df['profit_per_time'].max():.4f}]")

        return filtered_df

    def extract_pareto_points(self,
                              objective1: str = 'avg_test_profit',
                              objective2: str = 'avg_test_makespan',
                              minimize_objectives: List[bool] = [False, True]) -> pd.DataFrame:
        """
        从实验数据中提取Pareto最优解

        参数:
            objective1: 第一个目标（默认：利润，最大化）
            objective2: 第二个目标（默认：完工时长，最小化）
            minimize_objectives: 每个目标的优化方向 [obj1_minimize, obj2_minimize]
        """
        if self.df.empty:
            print("❌ 没有实验数据")
            return pd.DataFrame()

        # 检查目标列是否存在
        if objective1 not in self.df.columns:
            print(f"❌ 列 '{objective1}' 不存在")
            print(f"   可用列: {list(self.df.columns)}")
            return pd.DataFrame()

        # 确保所需列存在
        required_cols = [objective1, objective2]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"❌ 缺失列: {missing_cols}")
            return pd.DataFrame()

        # 提取关键数据
        points = self.df[required_cols].copy()
        points = points.dropna()

        if len(points) == 0:
            print("❌ 没有有效的数据点")
            return pd.DataFrame()

        if len(points) < 2:
            print(f"⚠️  只有 {len(points)} 个有效数据点，无法进行Pareto分析")
            # 只有一个点，直接返回
            return points

        print(f"📊 有效数据点统计:")
        print(f"   数据点总数: {len(points)}")
        print(f"   目标1 ({objective1}): 范围 [{points[objective1].min():.2f}, {points[objective1].max():.2f}]")
        print(f"   目标2 ({objective2}): 范围 [{points[objective2].min():.2f}, {points[objective2].max():.2f}]")

        # 转换为numpy数组
        X = points[[objective1, objective2]].values

        # 根据优化方向调整符号（使所有目标都变为最小化问题）
        if not minimize_objectives[0]:  # 最大化目标1 → 取负转换为最小化
            X[:, 0] = -X[:, 0]
        if not minimize_objectives[1]:  # 最大化目标2 → 取负转换为最小化
            X[:, 1] = -X[:, 1]

        # 计算Pareto前沿
        pareto_mask = self._is_pareto_efficient(X)

        # 提取Pareto点
        pareto_points = points[pareto_mask].copy()

        # 恢复原始值
        if not minimize_objectives[0]:
            pareto_points[objective1] = -pareto_points[objective1]
        if not minimize_objectives[1]:
            pareto_points[objective2] = -pareto_points[objective2]

        # 排序（按objective2升序）
        pareto_points = pareto_points.sort_values(by=objective2)

        print(f"✅ 找到 {len(pareto_points)} 个Pareto最优点（共 {len(points)} 个有效点）")

        return pareto_points

    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """
        计算Pareto前沿（最小化所有目标）

        参数:
            costs: (n_points, n_objectives) 数组，所有目标都应最小化
        """
        n_points = costs.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_efficient[i]:
                # 找到所有被i点支配的点
                dominates = np.all(costs[i] <= costs, axis=1) & np.any(costs[i] < costs, axis=1)
                is_efficient[dominates] = False

        return is_efficient

    def extract_pareto_points_from_df(self, df: pd.DataFrame,
                                      objective1: str, objective2: str,
                                      minimize_objectives: List[bool]) -> pd.DataFrame:
        """从指定DataFrame中提取Pareto点"""
        points = df[[objective1, objective2]].copy().dropna()

        if len(points) == 0:
            return pd.DataFrame()

        if len(points) < 2:
            # 只有一个点，直接返回
            return points

        X = points[[objective1, objective2]].values

        if not minimize_objectives[0]:
            X[:, 0] = -X[:, 0]
        if not minimize_objectives[1]:
            X[:, 1] = -X[:, 1]

        pareto_mask = self._is_pareto_efficient(X)
        pareto_points = points[pareto_mask].copy()

        if not minimize_objectives[0]:
            pareto_points[objective1] = -pareto_points[objective1]
        if not minimize_objectives[1]:
            pareto_points[objective2] = -pareto_points[objective2]

        return pareto_points.sort_values(by=objective2)

    def extract_pareto_by_group(self, group_column: str = 'experiment_group',
                                objective1: str = 'avg_test_profit',
                                objective2: str = 'avg_test_makespan',
                                minimize_objectives: List[bool] = [False, True]) -> Dict[str, pd.DataFrame]:
        """
        按照实验组提取Pareto点

        参数:
            group_column: 分组列名
            objective1: 第一个目标（利润）
            objective2: 第二个目标（完工时长）
            minimize_objectives: 优化方向

        返回:
            字典，键为组名，值为该组的Pareto点DataFrame
        """
        if self.df.empty:
            print("❌ 没有数据可分析")
            return {}

        if group_column not in self.df.columns:
            print(f"❌ 列 '{group_column}' 不存在")
            print(f"   可用列: {list(self.df.columns)}")
            return {}

        # 检查目标列是否存在
        if objective1 not in self.df.columns:
            print(f"❌ 目标列 '{objective1}' 不存在")
            return {}
        if objective2 not in self.df.columns:
            print(f"❌ 目标列 '{objective2}' 不存在")
            return {}

        groups = self.df[group_column].unique()
        group_pareto_dict = {}

        print(f"\n🔍 开始按组提取Pareto点，共 {len(groups)} 个组...")

        for group in groups:
            group_df = self.df[self.df[group_column] == group]

            if len(group_df) < 2:
                print(f"⚠️  组 '{group}' 数据点不足，跳过")
                continue

            # 提取该组的Pareto点
            pareto_points = self.extract_pareto_points_from_df(
                group_df, objective1, objective2, minimize_objectives
            )

            if not pareto_points.empty:
                # 添加组信息
                pareto_points_with_group = pareto_points.copy()
                pareto_points_with_group[group_column] = group

                # 添加其他有用信息
                for col in ['trainer_type', 'num_jobs', 'num_machines',
                            'profit_weight', 'makespan_weight', 'avg_test_profit',
                            'avg_test_makespan', 'profit_per_time', 'profit_rate',
                            'source_file', 'total_episodes']:
                    if col in group_df.columns and col not in pareto_points_with_group.columns:
                        try:
                            # 创建掩码匹配Pareto点
                            mask = (group_df[objective1] == pareto_points.iloc[0][objective1]) & \
                                   (group_df[objective2] == pareto_points.iloc[0][objective2])

                            if mask.any():
                                pareto_points_with_group[col] = group_df.loc[mask, col].iloc[0]
                        except:
                            pass

                group_pareto_dict[group] = pareto_points_with_group
                print(f"✅ 组 '{group}': 找到 {len(pareto_points)} 个Pareto点")
            else:
                print(f"⚠️  组 '{group}': 未找到Pareto点")

        self.group_pareto_points = group_pareto_dict
        return group_pareto_dict

    def save_all_pareto_points(self, output_dir: str = "pareto_output",
                               group_column: str = 'experiment_group',
                               objective1: str = 'avg_test_profit',
                               objective2: str = 'avg_test_makespan') -> Dict[str, str]:
        """
        保存所有Pareto点，包括按组保存和整体保存

        参数:
            output_dir: 输出目录
            group_column: 分组列名
            objective1: 第一个目标（利润）
            objective2: 第二个目标（完工时长）

        返回:
            字典，包含保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        print(f"\n💾 开始保存Pareto点数据...")

        if self.df.empty:
            print("❌ 没有数据可保存")
            return saved_files

        # 检查目标列是否存在
        if objective1 not in self.df.columns:
            print(f"❌ 目标列 '{objective1}' 不存在")
            print(f"   可用列: {list(self.df.columns)}")
            return saved_files

        # 检查分组列是否存在
        if group_column not in self.df.columns:
            print(f"❌ 分组列 '{group_column}' 不存在")
            print(f"   可用列: {list(self.df.columns)}")
            return saved_files

        # 1. 按组提取并保存Pareto点
        group_pareto_dict = self.extract_pareto_by_group(
            group_column=group_column,
            objective1=objective1,
            objective2=objective2,
            minimize_objectives=[False, True]
        )

        # 保存每个组的Pareto点
        for group_name, group_df in group_pareto_dict.items():
            # 清理组名，用于文件名
            safe_group_name = "".join([c if c.isalnum() else "_" for c in str(group_name)])
            file_path = os.path.join(output_dir, f"pareto_points_{safe_group_name}.csv")

            # 排序并保存
            group_df_sorted = group_df.sort_values(by=objective2)
            group_df_sorted.to_csv(file_path, index=False, encoding='utf-8-sig')

            saved_files[f"group_{safe_group_name}"] = file_path
            print(f"✅ 保存组 '{group_name}' 的 {len(group_df_sorted)} 个Pareto点到: {file_path}")

        # 2. 合并所有组的Pareto点
        if group_pareto_dict:
            all_pareto_points = pd.concat(group_pareto_dict.values(), ignore_index=True)

            # 排序
            all_pareto_points = all_pareto_points.sort_values(by=[group_column, objective2])

            # 保存合并文件
            merged_file = os.path.join(output_dir, "all_pareto_points_merged.csv")
            all_pareto_points.to_csv(merged_file, index=False, encoding='utf-8-sig')

            saved_files["all_merged"] = merged_file
            print(f"✅ 保存合并的 {len(all_pareto_points)} 个Pareto点到: {merged_file}")

            # 3. 生成分组统计信息
            self._generate_group_statistics(group_pareto_dict, output_dir, objective1, objective2)
        else:
            print("⚠️  没有找到任何Pareto点")

        # 4. 保存所有过滤后的数据点
        if not self.df.empty:
            df_all_points = self.df.copy()
            df_all_points['is_pareto'] = False

            # 标记Pareto点
            if group_pareto_dict:
                for group_name, group_df in group_pareto_dict.items():
                    for idx, row in group_df.iterrows():
                        mask = (df_all_points[group_column] == group_name)

                        # 尝试匹配目标1
                        if objective1 in row and objective1 in df_all_points.columns:
                            mask = mask & (df_all_points[objective1] == row[objective1])

                        # 尝试匹配目标2
                        if objective2 in row and objective2 in df_all_points.columns:
                            mask = mask & (df_all_points[objective2] == row[objective2])

                        if mask.any():
                            df_all_points.loc[mask, 'is_pareto'] = True

            # 保存所有点
            all_points_file = os.path.join(output_dir, "all_experiment_points.csv")
            df_all_points.to_csv(all_points_file, index=False, encoding='utf-8-sig')

            saved_files["all_points"] = all_points_file
            print(f"✅ 保存所有 {len(df_all_points)} 个实验点到: {all_points_file}")
            print(f"   其中 {df_all_points['is_pareto'].sum()} 个是Pareto最优点")

        # 5. 生成分析报告
        self._generate_pareto_analysis_report(group_pareto_dict, output_dir, objective1, objective2, group_column)

        return saved_files

    def _generate_group_statistics(self, group_pareto_dict: Dict[str, pd.DataFrame],
                                   output_dir: str, objective1: str, objective2: str):
        """生成分组统计信息"""
        stats_data = []

        for group_name, group_df in group_pareto_dict.items():
            if not group_df.empty:
                stats = {
                    'group_name': group_name,
                    'pareto_points_count': len(group_df),
                    f'min_{objective1}': group_df[objective1].min(),
                    f'max_{objective1}': group_df[objective1].max(),
                    f'avg_{objective1}': group_df[objective1].mean(),
                    f'min_{objective2}': group_df[objective2].min(),
                    f'max_{objective2}': group_df[objective2].max(),
                    f'avg_{objective2}': group_df[objective2].mean(),
                }

                # 计算目标范围
                stats[f'{objective1}_range'] = stats[f'max_{objective1}'] - stats[f'min_{objective1}']
                stats[f'{objective2}_range'] = stats[f'max_{objective2}'] - stats[f'min_{objective2}']

                # 如果有单位时间利润数据，也统计
                if 'profit_per_time' in group_df.columns:
                    stats['min_profit_per_time'] = group_df['profit_per_time'].min()
                    stats['max_profit_per_time'] = group_df['profit_per_time'].max()
                    stats['avg_profit_per_time'] = group_df['profit_per_time'].mean()

                # 如果有利润率数据，也统计
                if 'profit_rate' in group_df.columns:
                    stats['min_profit_rate'] = group_df['profit_rate'].min()
                    stats['max_profit_rate'] = group_df['profit_rate'].max()
                    stats['avg_profit_rate'] = group_df['profit_rate'].mean()

                stats_data.append(stats)

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_file = os.path.join(output_dir, "group_statistics.csv")
            stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
            print(f"✅ 保存分组统计信息到: {stats_file}")

    def _generate_pareto_analysis_report(self, group_pareto_dict: Dict[str, pd.DataFrame],
                                         output_dir: str, objective1: str, objective2: str,
                                         group_column: str):
        """生成Pareto分析报告"""
        report_path = os.path.join(output_dir, "pareto_analysis_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("           Pareto前沿分析报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据源: {self.data_source or '自动检测'}\n")
            f.write(f"目标1: {objective1} (最大化)\n")
            f.write(f"目标2: {objective2} (最小化)\n")
            f.write(f"分组列: {group_column}\n")
            f.write(f"有效数据点数: {len(self.df)}\n\n")

            if not group_pareto_dict:
                f.write("❌ 警告：未找到任何Pareto点\n")
                f.write("\n可能原因:\n")
                f.write("1. 数据点太少\n")
                f.write("2. 所有数据点都相同\n")
                f.write("3. 只有一个实验组\n")
                return

            f.write("-" * 70 + "\n")
            f.write("1. 各组Pareto点统计\n")
            f.write("-" * 70 + "\n")

            total_pareto_points = 0
            for group_name, group_df in group_pareto_dict.items():
                f.write(f"\n组名: {group_name}\n")
                f.write(f"  Pareto点数: {len(group_df)}\n")
                if not group_df.empty:
                    f.write(
                        f"  {objective1}范围: [{group_df[objective1].min():.2f}, {group_df[objective1].max():.2f}]\n")
                    f.write(
                        f"  {objective2}范围: [{group_df[objective2].min():.2f}, {group_df[objective2].max():.2f}]\n")

                    total_pareto_points += len(group_df)

            f.write(f"\n总计: {len(group_pareto_dict)} 个组, {total_pareto_points} 个Pareto点\n\n")

            # 配置建议
            if total_pareto_points > 0:
                all_points = pd.concat(group_pareto_dict.values(), ignore_index=True)

                f.write("-" * 70 + "\n")
                f.write("2. 配置建议\n")
                f.write("-" * 70 + "\n")

                # 最大利润配置
                max_profit_idx = all_points[objective1].idxmax()
                max_profit_config = all_points.loc[max_profit_idx]

                f.write(f"\nA. 最大利润配置:\n")
                f.write(f"   组: {max_profit_config.get(group_column, 'N/A')}\n")
                f.write(f"   利润: {max_profit_config[objective1]:.2f}\n")
                f.write(f"   完工时长: {max_profit_config[objective2]:.2f}\n")
                if 'profit_per_time' in max_profit_config:
                    f.write(f"   单位时间利润: {max_profit_config['profit_per_time']:.4f}\n")
                if 'profit_rate' in max_profit_config:
                    f.write(f"   利润率: {max_profit_config['profit_rate']:.4f}\n")

                # 最小完工时长配置
                min_makespan_idx = all_points[objective2].idxmin()
                min_makespan_config = all_points.loc[min_makespan_idx]

                f.write(f"\nB. 最小完工时长配置:\n")
                f.write(f"   组: {min_makespan_config.get(group_column, 'N/A')}\n")
                f.write(f"   完工时长: {min_makespan_config[objective2]:.2f}\n")
                f.write(f"   利润: {min_makespan_config[objective1]:.2f}\n")
                if 'profit_per_time' in min_makespan_config:
                    f.write(f"   单位时间利润: {min_makespan_config['profit_per_time']:.4f}\n")
                if 'profit_rate' in min_makespan_config:
                    f.write(f"   利润率: {min_makespan_config['profit_rate']:.4f}\n")

                # 平衡配置（取Pareto前沿的中点）
                if len(all_points) >= 3:
                    sorted_points = all_points.sort_values(by=objective2)
                    mid_idx = len(sorted_points) // 2
                    balanced_config = sorted_points.iloc[mid_idx]

                    f.write(f"\nC. 平衡配置 (Pareto前沿中点):\n")
                    f.write(f"   组: {balanced_config.get(group_column, 'N/A')}\n")
                    f.write(f"   利润: {balanced_config[objective1]:.2f}\n")
                    f.write(f"   完工时长: {balanced_config[objective2]:.2f}\n")
                    if 'profit_per_time' in balanced_config:
                        f.write(f"   单位时间利润: {balanced_config['profit_per_time']:.4f}\n")
                    if 'profit_rate' in balanced_config:
                        f.write(f"   利润率: {balanced_config['profit_rate']:.4f}\n")

        print(f"✅ 保存分析报告到: {report_path}")


def main():
    """主函数"""
    print("📈 Pareto前沿分析程序")
    print("=" * 50)

    # 用户可以选择不同的运行方式
    print("\n请选择运行方式:")
    print("1. 分析单个训练曲线文件")
    print("2. 分析目录中的所有实验")
    print("3. 分析指定的数据文件")

    choice = input("\n请输入选择 (1-3, 默认2): ").strip()

    if choice == "1":
        # 分析单个训练曲线文件
        file_path = input("请输入训练曲线文件路径: ").strip()
        if not file_path:
            print("❌ 未提供文件路径")
            return

        analyzer = ParetoAnalyzer(
            data_source='training_curves',
            data_file=file_path
        )

    elif choice == "3":
        # 分析指定的数据文件
        file_path = input("请输入数据文件路径: ").strip()
        if not file_path:
            print("❌ 未提供文件路径")
            return

        analyzer = ParetoAnalyzer(data_file=file_path)

    else:
        # 分析目录中的所有实验
        results_dir = input(f"请输入实验目录路径 (默认: experiment_results): ").strip()
        if not results_dir:
            results_dir = "experiment_results"

        analyzer = ParetoAnalyzer(results_dir=results_dir)

    if analyzer.df.empty:
        print("❌ 没有加载到有效数据")
        return

    print(f"\n📊 数据加载成功:")
    print(f"   数据行数: {len(analyzer.df)}")
    print(f"   数据列: {list(analyzer.df.columns)}")

    # 选择分析目标
    print("\n🔧 选择分析目标:")
    print("1. 利润 vs 完工时长")
    print("2. 利润率 vs 完工时长")
    print("3. 单位时间利润 vs 完工时长")

    target_choice = input("请输入选择 (1-3, 默认1): ").strip()

    if target_choice == "2" and 'profit_rate' in analyzer.df.columns:
        objective1 = 'profit_rate'
        print("✅ 使用利润率作为目标1")
    elif target_choice == "3":
        objective1 = 'profit_per_time'
        print("✅ 使用单位时间利润作为目标1")
    else:
        objective1 = 'avg_test_profit'
        print("✅ 使用利润作为目标1")

    objective2 = 'avg_test_makespan'

    # 保存所有Pareto点
    saved_files = analyzer.save_all_pareto_points(
        output_dir="pareto_output",
        group_column='experiment_group',
        objective1=objective1,
        objective2=objective2
    )

    print(f"\n🎉 Pareto前沿分析完成!")
    print(f"   输出文件保存在: pareto_output/")
    if saved_files:
        print(f"   共生成 {len(saved_files)} 个输出文件")


if __name__ == "__main__":
    main()