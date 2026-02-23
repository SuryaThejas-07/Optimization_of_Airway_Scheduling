from __future__ import annotations

import pandas as pd


def check_conflicts(schedule: pd.DataFrame, sep_matrix: list[list[int]]) -> int:
    if schedule.empty:
        return 0
    times = schedule["scheduled_time"].values
    conflicts = 0
    for i in range(len(times)):
        for j in range(i):
            if times[i] - times[j] < sep_matrix[i][j]:
                conflicts += 1
    return conflicts
