from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim

from agno_runway.data.event_extractor import EventConfig, detect_events
from agno_runway.data.loader import load_states
from agno_runway.data.separation_builder import build_separation_matrix
from agno_runway.optimizer.graph_model import AGNOModel, build_features


def _wake_onehot(series: pd.Series) -> torch.Tensor:
    mapping = {"H": 0, "M": 1, "L": 2}
    idx = series.map(mapping).fillna(1).astype(int).values
    onehot = torch.zeros((len(idx), 3))
    onehot[range(len(idx)), idx] = 1.0
    return onehot


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--radius-km", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = (
        Path(args.data)
        if args.data
        else root.parent / "states_2022-06-27-23.csv"
    )

    states, airport = load_states(data_path, radius_km=args.radius_km)
    if airport is None:
        raise RuntimeError("Could not infer airport location from data.")

    flights = detect_events(states, EventConfig(airport_center=airport, radius_km=args.radius_km))
    if flights.empty:
        raise RuntimeError("No events found.")

    device = _resolve_device(args.device)
    print(f"Using device: {device.type}")
    model = AGNOModel(feature_dim=6, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    eta = torch.tensor(flights["eta_seconds"].values, dtype=torch.float32, device=device)
    velocity = torch.tensor(flights["velocity"].values, dtype=torch.float32, device=device)
    altitude = torch.tensor(
        flights["geoaltitude"].fillna(0).values, dtype=torch.float32, device=device
    )
    wake_onehot = _wake_onehot(flights["wake_class"]).to(device)
    sep_matrix = torch.tensor(
        build_separation_matrix(flights["wake_class"].tolist()),
        dtype=torch.float32,
        device=device,
    )
    features = build_features(eta, velocity, altitude, wake_onehot)
    adj = (sep_matrix > 0).float()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        scores = model(features, adj)
        # Proxy objective: encourage earlier ETA flights to score higher priority.
        target = -eta
        loss = torch.mean((scores - target) ** 2)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1} loss={loss.item():.3f}")


if __name__ == "__main__":
    main()
