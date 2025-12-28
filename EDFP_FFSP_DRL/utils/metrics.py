import numpy as np
from typing import List, Tuple, Dict


def compute_makespan(schedule: List[List[Tuple]]) -> float:
    """
    计算最大完成时间
    """
    if not schedule:
        return 0.0

    makespan = 0.0
    for machine_schedule in schedule:
        if machine_schedule:
            last_end_time = max(job_info[2] for job_info in machine_schedule)
            makespan = max(makespan, last_end_time)

    return makespan


def compute_utilization(schedule: List[List[Tuple]], num_machines: int,
                        total_time: float) -> float:
    """
    计算机器利用率
    """
    if total_time <= 0:
        return 0.0

    total_busy_time = 0.0
    for machine_schedule in schedule:
        machine_busy_time = sum(job_info[2] - job_info[1] for job_info in machine_schedule)
        total_busy_time += machine_busy_time

    return total_busy_time / (num_machines * total_time)


def compute_tardiness(schedule: List[List[Tuple]], due_dates: Dict[int, float]) -> float:
    """
    计算总延迟时间
    """
    # 计算每个作业的完成时间
    job_completion_times = {}
    for machine_schedule in schedule:
        for job_info in machine_schedule:
            job_id = job_info[0]
            completion_time = job_info[2]
            if job_id not in job_completion_times or completion_time > job_completion_times[job_id]:
                job_completion_times[job_id] = completion_time

    # 计算总延迟
    total_tardiness = 0.0
    for job_id, completion_time in job_completion_times.items():
        if job_id in due_dates:
            tardiness = max(0, completion_time - due_dates[job_id])
            total_tardiness += tardiness

    return total_tardiness


def compute_flow_time(schedule: List[List[Tuple]], arrival_times: Dict[int, float]) -> float:
    """
    计算总流时间（作业在系统中的总时间）
    """
    # 计算每个作业的完成时间
    job_completion_times = {}
    for machine_schedule in schedule:
        for job_info in machine_schedule:
            job_id = job_info[0]
            completion_time = job_info[2]
            if job_id not in job_completion_times or completion_time > job_completion_times[job_id]:
                job_completion_times[job_id] = completion_time

    # 计算总流时间
    total_flow_time = 0.0
    for job_id, completion_time in job_completion_times.items():
        arrival_time = arrival_times.get(job_id, 0.0)
        flow_time = completion_time - arrival_time
        total_flow_time += flow_time

    return total_flow_time


def generate_due_dates(num_jobs: int, processing_times: np.ndarray,
                       tightness: float = 1.5) -> Dict[int, float]:
    """
    生成作业的交付日期
    """
    due_dates = {}

    for job_id in range(num_jobs):
        # 计算作业的总处理时间
        total_processing_time = np.sum(processing_times[job_id])

        # 基于处理时间和紧度生成交付日期
        due_date = total_processing_time * tightness
        due_dates[job_id] = due_date

    return due_dates