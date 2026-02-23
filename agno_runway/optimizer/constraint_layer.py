from __future__ import annotations

import torch


def apply_separation_penalty(
    schedule_times: torch.Tensor, sep_matrix: torch.Tensor, steps: int = 8, lr: float = 0.2
) -> torch.Tensor:
    times = schedule_times.clone()
    n = times.numel()
    for _ in range(steps):
        grads = torch.zeros_like(times)
        for i in range(n):
            for j in range(i):
                required = sep_matrix[i, j]
                if times[i] - times[j] < required:
                    grads[i] += (required - (times[i] - times[j]))
        times = times + lr * grads
    return times
