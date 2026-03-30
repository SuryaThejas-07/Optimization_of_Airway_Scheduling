from __future__ import annotations

from typing import List

import torch
from torch_geometric.data import Data

from agno_runway.utils.aviation_rules import AviationRulesEngine
from agno_runway.utils.schemas import FlightInput


_WAKE_TO_ONEHOT = {
    "H": [1.0, 0.0, 0.0],
    "M": [0.0, 1.0, 0.0],
    "L": [0.0, 0.0, 1.0],
}


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    v = torch.tensor(values, dtype=torch.float32)
    std = float(v.std(unbiased=False).item())
    if std < 1e-6:
        return [0.0 for _ in values]
    mean = float(v.mean().item())
    return [float((x - mean) / std) for x in values]


def build_flight_graph(
    flights: List[FlightInput],
    rules: AviationRulesEngine,
    time_proximity_seconds: float,
) -> Data:
    if not flights:
        return Data(
            x=torch.empty((0, 6), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    etas = [f.eta_seconds for f in flights]
    speeds = [f.velocity for f in flights]
    altitudes = [f.altitude for f in flights]

    eta_norm = _normalize(etas)
    speed_norm = _normalize(speeds)
    alt_norm = _normalize(altitudes)

    node_features: List[List[float]] = []
    for i, flight in enumerate(flights):
        onehot = _WAKE_TO_ONEHOT.get(flight.wake_class, _WAKE_TO_ONEHOT["M"])
        node_features.append([eta_norm[i], speed_norm[i], alt_norm[i], *onehot])

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_attr: List[List[float]] = []

    for i, leader in enumerate(flights):
        for j, follower in enumerate(flights):
            if i == j:
                continue

            eta_diff = abs(leader.eta_seconds - follower.eta_seconds)
            required_sep = rules.required_wake_gap(
                leader.wake_class, follower.wake_class
            )
            if eta_diff <= time_proximity_seconds or required_sep > 100.0:
                edge_src.append(i)
                edge_dst.append(j)
                edge_attr.append(
                    [required_sep / 300.0, eta_diff / max(time_proximity_seconds, 1.0)]
                )

    if not edge_src:
        for i in range(len(flights)):
            edge_src.append(i)
            edge_dst.append(i)
            edge_attr.append([0.0, 0.0])

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
