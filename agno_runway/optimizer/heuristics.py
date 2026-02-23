from __future__ import annotations

import random
import time
import torch
from .robust_refiner import refine_schedule_with_runways

def schedule_cost(
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    order: torch.Tensor,
    runway_count: int,
) -> float:
    """Calculates total delay for a given ordering."""
    ordered_eta = eta[order]
    sep_ordered = sep_matrix[order][:, order]
    schedule_times, _ = refine_schedule_with_runways(
        ordered_eta, sep_ordered, runway_count
    )
    return (schedule_times - ordered_eta).clamp(min=0).sum().item()

def separation_swap(
    order: torch.Tensor,
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    max_eta_gap: float = 600.0,
    passes: int = 2,
) -> torch.Tensor:
    """Local swap heuristic based on separation matrix efficiency."""
    order = order.clone()
    n = order.numel()
    for _ in range(passes):
        swapped = False
        for i in range(n - 1):
            a = order[i]
            b = order[i + 1]
            eta_gap = (eta[b] - eta[a]).item()
            if eta_gap < -max_eta_gap:
                continue
            sep_ab = sep_matrix[a, b].item()
            sep_ba = sep_matrix[b, a].item()
            if sep_ba + 1e-6 < sep_ab:
                order[i], order[i + 1] = b, a
                swapped = True
        if not swapped:
            break
    return order

def lns_refine(
    order: torch.Tensor,
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    runway_count: int,
    seconds: float,
) -> torch.Tensor:
    """Large Neighborhood Search to refine flight ordering."""
    start = time.perf_counter()
    order = order.clone()
    best_cost = schedule_cost(eta, sep_matrix, order, runway_count)
    n = order.numel()

    while (time.perf_counter() - start) < seconds:
        block_size = min(n, random.randint(6, 12))
        start_idx = random.randint(0, max(0, n - block_size))
        block = order[start_idx : start_idx + block_size]
        remaining = torch.cat([order[:start_idx], order[start_idx + block_size :]])

        for item in block:
            rem_len = remaining.numel()
            if rem_len == 0:
                remaining = item.unsqueeze(0)
                continue
            
            # Sample positions to test
            stride = max(1, rem_len // 20)
            positions = list(range(0, rem_len + 1, stride))
            if positions[-1] != rem_len:
                positions.append(rem_len)

            best_pos = 0
            best_local = None
            for pos in positions:
                candidate = torch.cat([remaining[:pos], item.unsqueeze(0), remaining[pos:]])
                cost = schedule_cost(eta, sep_matrix, candidate, runway_count)
                if best_local is None or cost < best_local:
                    best_local = cost
                    best_pos = pos

            remaining = torch.cat([remaining[:best_pos], item.unsqueeze(0), remaining[best_pos:]])

        candidate_cost = schedule_cost(eta, sep_matrix, remaining, runway_count)
        if candidate_cost < best_cost:
            order = remaining
            best_cost = candidate_cost

    return order
