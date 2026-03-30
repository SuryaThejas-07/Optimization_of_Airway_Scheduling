from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from agno_runway.models.graph_builder import build_flight_graph
from agno_runway.utils.aviation_rules import AviationRulesEngine
from agno_runway.utils.schemas import FlightInput


class PriorityGNN(nn.Module):
    def __init__(
        self, in_dim: int = 6, hidden_dim: int = 64, edge_dim: int = 2
    ) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=2, edge_dim=edge_dim, dropout=0.1
        )
        self.conv2 = GATv2Conv(
            hidden_dim * 2, hidden_dim, heads=1, edge_dim=edge_dim, dropout=0.1
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_attr)
        h = torch.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = torch.relu(h)
        return self.head(h).squeeze(-1)


class PriorityScoringService:
    def __init__(
        self,
        model: PriorityGNN,
        rules: AviationRulesEngine,
        time_proximity_seconds: float,
        emergency_priority_bonus: float,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.rules = rules
        self.time_proximity_seconds = time_proximity_seconds
        self.emergency_priority_bonus = emergency_priority_bonus
        self.device = device

    @torch.no_grad()
    def score_flights(self, flights: List[FlightInput]) -> List[float]:
        if not flights:
            return []

        graph = build_flight_graph(
            flights=flights,
            rules=self.rules,
            time_proximity_seconds=self.time_proximity_seconds,
        )

        graph = graph.to(self.device)
        self.model.eval()
        scores = self.model(graph.x, graph.edge_index, graph.edge_attr).detach().cpu()

        # Add deterministic bias terms so the service is useful even before formal training.
        eta = torch.tensor([f.eta_seconds for f in flights], dtype=torch.float32)
        eta_bias = -(eta - eta.mean()) / (eta.std(unbiased=False) + 1e-6)

        emergency = torch.tensor(
            [self.emergency_priority_bonus if f.emergency else 0.0 for f in flights],
            dtype=torch.float32,
        )
        adjusted = scores + eta_bias + emergency
        return adjusted.tolist()
