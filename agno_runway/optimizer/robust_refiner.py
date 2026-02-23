from __future__ import annotations

import torch


def refine_schedule_with_runways(
    schedule_times: torch.Tensor, sep_matrix: torch.Tensor, runway_count: int
) -> tuple[torch.Tensor, list[int]]:
    times = schedule_times.clone()
    n = times.numel()
    
    # Track the scheduled time and the original index of the last flight on each runway
    last_runway_time = [0.0 for _ in range(runway_count)]
    last_runway_idx = [-1 for _ in range(runway_count)]
    runways = [0 for _ in range(n)]

    for i in range(n):
        # Choose the runway that becomes available earliest
        earliest_available_time = min(last_runway_time)
        runway_idx = last_runway_time.index(earliest_available_time)
        
        # Current flight cannot land before its ETA
        scheduled_time = times[i].item()
        
        # Enforce separation only against the PREVIOUS flight on THIS SAME runway
        prev_idx = last_runway_idx[runway_idx]
        if prev_idx != -1:
            required_sep = sep_matrix[i, prev_idx].item()
            prev_time = last_runway_time[runway_idx]
            scheduled_time = max(scheduled_time, prev_time + required_sep)
        
        # Also ensure it doesn't land before the runway is physically clear from its own previous operation
        scheduled_time = max(scheduled_time, earliest_available_time)
        
        times[i] = scheduled_time
        runways[i] = runway_idx
        last_runway_time[runway_idx] = scheduled_time
        last_runway_idx[runway_idx] = i

    return times, runways


def refine_schedule(
    schedule_times: torch.Tensor, sep_matrix: torch.Tensor, runway_count: int
) -> torch.Tensor:
    times, _ = refine_schedule_with_runways(schedule_times, sep_matrix, runway_count)
    return times
