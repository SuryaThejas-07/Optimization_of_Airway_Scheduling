from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    schedule: pd.DataFrame, sep_matrix: list[list[int]] | None = None
) -> dict:
    if schedule.empty:
        return {}
    total_delay = schedule["delay"].sum()
    avg_delay = schedule["delay"].mean()
    makespan = schedule["scheduled_time"].max() - schedule["scheduled_time"].min()
    throughput = len(schedule) / max(makespan, 1.0)
    metrics = {
        "total_delay": float(total_delay),
        "avg_delay": float(avg_delay),
        "makespan": float(makespan),
        "throughput": float(throughput),
    }
    if sep_matrix is not None and "flight_index" in schedule.columns:
        slack_avg, slack_min = _safety_slack(schedule, sep_matrix)
        metrics["avg_safety_slack"] = float(slack_avg)
        metrics["min_safety_slack"] = float(slack_min)
    return metrics


def _safety_slack(schedule: pd.DataFrame, sep_matrix: list[list[int]]) -> tuple[float, float]:
    # Ensure flights are sorted by scheduled time for chronological slack checking
    df = schedule.sort_values("scheduled_time")
    indices = df["flight_index"].astype(int).values
    times = df["scheduled_time"].values
    runways = df["assigned_runway"].values
    sep = np.asarray(sep_matrix)
    n = len(indices)
    slacks = []
    
    for i in range(1, n):
        # Find the most recent flight ON THE SAME RUNWAY
        for j in range(i - 1, -1, -1):
            if runways[i] == runways[j]:
                required = sep[indices[i], indices[j]]
                slacks.append(times[i] - times[j] - required)
                break  # Only check separation against the head of the runway queue
    
    if not slacks:
        return 0.0, 0.0
    slacks = np.asarray(slacks, dtype=float)
    return float(slacks.mean()), float(slacks.min())
