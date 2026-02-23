from __future__ import annotations

import pandas as pd
import torch

from agno_runway.data.separation_builder import build_separation_matrix
from agno_runway.optimizer.robust_refiner import refine_schedule_with_runways


def milp_schedule(flights: pd.DataFrame, runway_count: int = 2) -> pd.DataFrame:
    try:
        import pulp  # type: ignore
    except Exception:
        pulp = None

    # Placeholder MILP baseline: use FCFS ordering to ensure valid output.
    flights = flights.sort_values("eta_seconds").reset_index(drop=True)
    sep_matrix = torch.tensor(
        build_separation_matrix(flights["wake_class"].tolist()), dtype=torch.float32
    )
    eta = torch.tensor(flights["eta_seconds"].values, dtype=torch.float32)
    schedule_times, runways = refine_schedule_with_runways(eta, sep_matrix, runway_count)
    flights["order"] = flights.index
    flights["scheduled_time"] = schedule_times.numpy()
    flights["delay"] = flights["scheduled_time"] - flights["eta_seconds"]
    flights["safety_margin"] = 0.0
    flights["assigned_runway"] = [f"RWY_{idx + 1:02d}" for idx in runways]
    return flights
