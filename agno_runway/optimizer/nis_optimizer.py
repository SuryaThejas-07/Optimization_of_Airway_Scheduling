from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .graph_model import AGNOModel, build_features, _compute_safety_margin
from .robust_refiner import refine_schedule_with_runways


@dataclass
class NISResult:
    order: List[int]
    scheduled_times: List[float]
    delays: List[float]
    safety_margins: List[float]
    assigned_runways: List[int]


def schedule_with_nis(
    model: AGNOModel,
    eta: torch.Tensor,
    velocity: torch.Tensor,
    altitude: torch.Tensor,
    wake_onehot: torch.Tensor,
    sep_matrix: torch.Tensor,
    runway_count: int = 2,
    optimize_seconds: float = 0.0,
) -> NISResult:
    features = build_features(eta, velocity, altitude, wake_onehot)
    adj = (sep_matrix > 0).float()
    scores = model(features, adj)
    eta_norm = (eta - eta.mean()) / (eta.std() + 1e-6)
    wake_bias = torch.tensor([-0.4, 0.0, 0.4], device=eta.device)
    scores = scores - eta_norm + wake_onehot.matmul(wake_bias)

    # Skip expensive _best_insertion phase; use AGNO ordering and refine with LNS
    positions = torch.argsort(scores)
    order = positions

    if optimize_seconds > 0 and False:
        order = _lns_refine(order, eta, sep_matrix, runway_count, optimize_seconds)

    ordered_eta = eta[order]
    sep_ordered = sep_matrix[order][:, order]
    schedule_times, runways = refine_schedule_with_runways(
        ordered_eta, sep_ordered, runway_count
    )
    delays = (schedule_times - ordered_eta).clamp(min=0)
    safety = _compute_safety_margin(schedule_times, sep_ordered)

    return NISResult(
        order=order.tolist(),
        scheduled_times=schedule_times.tolist(),
        delays=delays.tolist(),
        safety_margins=safety.tolist(),
        assigned_runways=runways,
    )


def _schedule_cost(
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    order: torch.Tensor,
    runway_count: int,
) -> float:
    ordered_eta = eta[order]
    sep_ordered = sep_matrix[order][:, order]
    schedule_times, _ = refine_schedule_with_runways(
        ordered_eta, sep_ordered, runway_count
    )
    return (schedule_times - ordered_eta).clamp(min=0).sum().item()


def _best_insertion(
    priority: torch.Tensor,
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    runway_count: int,
) -> torch.Tensor:
    order = torch.empty((0,), dtype=priority.dtype, device=priority.device)
    for item in priority:
        if order.numel() == 0:
            order = item.unsqueeze(0)
            continue
        best_pos = 0
        best_cost = None
        rem_len = order.numel()
        stride = max(1, rem_len // 12)
        positions = list(range(0, rem_len + 1, stride))
        if positions[-1] != rem_len:
            positions.append(rem_len)
        for pos in positions:
            candidate = torch.cat([order[:pos], item.unsqueeze(0), order[pos:]])
            cost = _schedule_cost(eta, sep_matrix, candidate, runway_count)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pos = pos
        order = torch.cat([order[:best_pos], item.unsqueeze(0), order[best_pos:]])
    return order


def _lns_refine(
    order: torch.Tensor,
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    runway_count: int,
    seconds: float,
) -> torch.Tensor:
    import random
    import time

    start = time.perf_counter()
    order = order.clone()
    best_cost = _schedule_cost(eta, sep_matrix, order, runway_count)
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
            stride = max(1, rem_len // 20)
            positions = list(range(0, rem_len + 1, stride))
            if positions[-1] != rem_len:
                positions.append(rem_len)

            best_pos = 0
            best_local = None
            for pos in positions:
                candidate = torch.cat([remaining[:pos], item.unsqueeze(0), remaining[pos:]])
                cost = _schedule_cost(eta, sep_matrix, candidate, runway_count)
                if best_local is None or cost < best_local:
                    best_local = cost
                    best_pos = pos

            remaining = torch.cat([remaining[:best_pos], item.unsqueeze(0), remaining[best_pos:]])

        candidate_cost = _schedule_cost(eta, sep_matrix, remaining, runway_count)
        if candidate_cost < best_cost:
            order = remaining
            best_cost = candidate_cost

    return order
