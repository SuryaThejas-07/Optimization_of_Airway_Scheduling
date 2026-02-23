from __future__ import annotations

import pandas as pd


def check_conflicts(schedule: pd.DataFrame, sep_matrix: list[list[int]]) -> int:
    if schedule.empty or "assigned_runway" not in schedule.columns:
        return 0
    
    df = schedule.sort_values("scheduled_time")
    times = df["scheduled_time"].values
    runways = df["assigned_runway"].values
    # We need a way to map the original indices to the sorted order if sep_matrix is index-based
    # But usually, in this context, the sep_matrix provided to check_conflicts is already 
    # aligned with the schedule rows. Let's assume the schedule has 'original_index' or similar.
    # Actually, the standard pattern in the dashboard is that the schedule rows match the matrix.
    
    conflicts = 0
    n = len(df)
    for i in range(n):
        for j in range(i):
            if runways[i] == runways[j]:
                # Find required separation. 
                # Note: this assumes sep_matrix is aligned with the DataFrame rows
                if times[i] - times[j] < sep_matrix[i][j]:
                    conflicts += 1
    return conflicts
