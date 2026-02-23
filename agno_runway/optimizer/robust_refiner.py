from __future__ import annotations

import torch


def refine_schedule_with_runways(
    schedule_times: torch.Tensor, sep_matrix: torch.Tensor, runway_count: int
) -> tuple[torch.Tensor, list[int]]:
    times = schedule_times.clone()
    n = times.numel()
    runway_available = [0.0 for _ in range(runway_count)]
    runways = [0 for _ in range(n)]

    for i in range(n):
        earliest = min(runway_available)
        runway_idx = runway_available.index(earliest)
        times[i] = max(times[i], earliest)

        # Vectorized separation check: for each prior flight j, compute required time
        if i > 0:
            prior_times = times[:i]
            required_sep = sep_matrix[i, :i]
            # Compute minimum time this flight must slot at based on each prior conflict
            min_times = prior_times + required_sep
            times[i] = max(times[i], min_times.max())

        runway_available[runway_idx] = times[i].item() if isinstance(times[i], torch.Tensor) else times[i]
        runways[i] = runway_idx

    return times, runways


def refine_schedule(
    schedule_times: torch.Tensor, sep_matrix: torch.Tensor, runway_count: int
) -> torch.Tensor:
    times, _ = refine_schedule_with_runways(schedule_times, sep_matrix, runway_count)
    return times
