from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .differentiable_sort import soft_rank
from .robust_refiner import refine_schedule_with_runways


@dataclass
class ScheduleResult:
    order: List[int]
    scheduled_times: List[float]
    delays: List[float]
    safety_margins: List[float]
    assigned_runways: List[int]


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, steps: int = 2) -> None:
        super().__init__()
        self.steps = steps
        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.neigh_proj = nn.Linear(in_dim, hidden_dim)
        self.hidden_self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_neigh_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh = torch.matmul(adj, x) / deg
        h = F.relu(self.self_proj(x) + self.neigh_proj(neigh))
        for _ in range(max(self.steps - 1, 0)):
            neigh = torch.matmul(adj, h) / deg
            h = F.relu(self.hidden_self_proj(h) + self.hidden_neigh_proj(neigh))
        return self.out_proj(h)


class AGNOModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = GraphEncoder(feature_dim, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(features, adj)
        scores = self.score_head(h).squeeze(-1)
        return scores


def build_features(
    eta: torch.Tensor, velocity: torch.Tensor, altitude: torch.Tensor, wake_class: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        [
            eta.unsqueeze(-1),
            velocity.unsqueeze(-1),
            altitude.unsqueeze(-1),
            wake_class,
        ],
        dim=-1,
    )


def schedule_with_agnos(
    model: AGNOModel,
    eta: torch.Tensor,
    velocity: torch.Tensor,
    altitude: torch.Tensor,
    wake_onehot: torch.Tensor,
    sep_matrix: torch.Tensor,
    runway_count: int = 2,
    tau: float = 0.3,
    optimize_seconds: float = 0.0,
) -> ScheduleResult:
    features = build_features(eta, velocity, altitude, wake_onehot)
    adj = (sep_matrix > 0).float()

    scores = model(features, adj)
    eta_norm = (eta - eta.mean()) / (eta.std() + 1e-6)
    wake_bias = torch.tensor([-0.4, 0.0, 0.4], device=eta.device)
    scores = scores - eta_norm + wake_onehot.matmul(wake_bias)
    positions = soft_rank(scores, tau=tau)
    order = torch.argsort(positions)
    # Disabled: order = _separation_swap(order, eta, sep_matrix, passes=5)
    # Disabled: order = _window_refine(order, sep_matrix, window=4)
    if optimize_seconds > 0:
        order = _lns_refine(order, eta, sep_matrix, runway_count, optimize_seconds)

    ordered_eta = eta[order]
    sep_ordered = sep_matrix[order][:, order]

    schedule_times = ordered_eta.clone()
    schedule_times, runways = refine_schedule_with_runways(
        schedule_times, sep_ordered, runway_count
    )

    delays = (schedule_times - ordered_eta).clamp(min=0)
    safety = _compute_safety_margin(schedule_times, sep_ordered)

    return ScheduleResult(
        order=order.tolist(),
        scheduled_times=schedule_times.tolist(),
        delays=delays.tolist(),
        safety_margins=safety.tolist(),
        assigned_runways=runways,
    )


def _compute_safety_margin(times: torch.Tensor, sep: torch.Tensor) -> torch.Tensor:
    n = times.numel()
    margins = torch.full((n,), float("inf"), device=times.device)
    for i in range(n):
        diffs = times[i] - times
        required = sep[:, i]
        slack = diffs - required
        slack[i] = float("inf")
        margins[i] = torch.min(slack)
    return margins


def _separation_swap(
    order: torch.Tensor,
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    max_eta_gap: float = 600.0,
    passes: int = 2,
) -> torch.Tensor:
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


def _window_refine(
    order: torch.Tensor, sep_matrix: torch.Tensor, window: int = 4
) -> torch.Tensor:
    if window <= 2:
        return order
    order = order.clone()
    n = order.numel()
    for start in range(0, n - window + 1):
        block = order[start : start + window]
        best_block = block
        best_cost = _block_cost(block, sep_matrix)
        for perm in _permute_indices(window):
            perm_idx = torch.tensor(perm, device=block.device)
            candidate = block[perm_idx]
            cost = _block_cost(candidate, sep_matrix)
            if cost < best_cost:
                best_cost = cost
                best_block = candidate
        order[start : start + window] = best_block
    return order


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

        # Reinsert block items with limited candidate positions.
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


def _block_cost(block: torch.Tensor, sep_matrix: torch.Tensor) -> float:
    cost = 0.0
    for i in range(block.numel() - 1):
        cost += sep_matrix[block[i], block[i + 1]].item()
    return cost


def _permute_indices(n: int) -> list[list[int]]:
    if n == 3:
        return [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ]
    if n == 4:
        return [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ]
    return [list(range(n))]
