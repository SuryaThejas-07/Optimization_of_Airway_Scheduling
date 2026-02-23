from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .differentiable_sort import soft_rank
from .robust_refiner import refine_schedule_with_runways
from .heuristics import separation_swap, lns_refine, schedule_cost


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
    order = separation_swap(order, eta, sep_matrix, passes=5)
    # Disabled: order = _window_refine(order, sep_matrix, window=4)
    if optimize_seconds > 0:
        order = lns_refine(order, eta, sep_matrix, runway_count, optimize_seconds)

    ordered_eta = eta[order]
    sep_ordered = sep_matrix[order][:, order]

    schedule_times = ordered_eta.clone()
    schedule_times, runways = refine_schedule_with_runways(
        schedule_times, sep_ordered, runway_count
    )

    delays = (schedule_times - ordered_eta).clamp(min=0)
    safety = _compute_safety_margin(schedule_times, sep_ordered, runways)

    return ScheduleResult(
        order=order.tolist(),
        scheduled_times=schedule_times.tolist(),
        delays=delays.tolist(),
        safety_margins=safety.tolist(),
        assigned_runways=runways,
    )


def _compute_safety_margin(times: torch.Tensor, sep: torch.Tensor, runways: List[int]) -> torch.Tensor:
    n = times.numel()
    margins = torch.full((n,), float("inf"), device=times.device)
    for i in range(n):
        for j in range(i):
            if runways[i] == runways[j]:
                slack = (times[i] - times[j]) - sep[j, i]
                if slack < margins[i]:
                    margins[i] = slack
    return margins

