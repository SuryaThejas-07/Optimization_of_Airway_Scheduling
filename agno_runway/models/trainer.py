from __future__ import annotations

from typing import List

import torch
from torch import nn

from agno_runway.models.graph_builder import build_flight_graph
from agno_runway.models.gnn_priority import PriorityGNN
from agno_runway.utils.aviation_rules import AviationRulesEngine
from agno_runway.utils.schemas import FlightInput


def simulated_priority_targets(
    flights: List[FlightInput], emergency_bonus: float = 5.0
) -> torch.Tensor:
    if not flights:
        return torch.empty((0,), dtype=torch.float32)

    eta = torch.tensor([f.eta_seconds for f in flights], dtype=torch.float32)
    base = -(eta - eta.mean()) / (eta.std(unbiased=False) + 1e-6)
    emergency = torch.tensor(
        [emergency_bonus if f.emergency else 0.0 for f in flights], dtype=torch.float32
    )
    return base + emergency


def train_with_simulated_targets(
    model: PriorityGNN,
    flights: List[FlightInput],
    rules: AviationRulesEngine,
    time_proximity_seconds: float,
    epochs: int = 40,
    lr: float = 1e-3,
    device: str = "cpu",
) -> PriorityGNN:
    if not flights:
        return model

    graph = build_flight_graph(flights, rules, time_proximity_seconds).to(device)
    targets = simulated_priority_targets(flights).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        pred = model(graph.x, graph.edge_index, graph.edge_attr)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()

    return model
