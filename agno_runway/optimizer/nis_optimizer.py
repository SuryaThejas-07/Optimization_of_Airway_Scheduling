from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .graph_model import AGNOModel, build_features, _compute_safety_margin
from .robust_refiner import refine_schedule_with_runways
from .heuristics import lns_refine, schedule_cost


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

    if optimize_seconds > 0:
        order = lns_refine(order, eta, sep_matrix, runway_count, optimize_seconds)

    ordered_eta = eta[order]
    sep_ordered = sep_matrix[order][:, order]
    schedule_times, runways = refine_schedule_with_runways(
        ordered_eta, sep_ordered, runway_count
    )
    delays = (schedule_times - ordered_eta).clamp(min=0)
    safety = _compute_safety_margin(schedule_times, sep_ordered, runways)

    return NISResult(
        order=order.tolist(),
        scheduled_times=schedule_times.tolist(),
        delays=delays.tolist(),
        safety_margins=safety.tolist(),
        assigned_runways=runways,
    )


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
            cost = schedule_cost(eta, sep_matrix, candidate, runway_count)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pos = pos
        order = torch.cat([order[:best_pos], item.unsqueeze(0), order[best_pos:]])
    return order
