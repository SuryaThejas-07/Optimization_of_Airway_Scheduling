from __future__ import annotations

import pandas as pd
import torch

from agno_runway.data.separation_builder import build_separation_matrix
from agno_runway.optimizer.robust_refiner import refine_schedule_with_runways

# MILP is NP-hard for large instances. Limit to keep solve time < 120s.
MAX_MILP_FLIGHTS = 50


def milp_schedule(flights: pd.DataFrame, runway_count: int = 2) -> pd.DataFrame:
    """
    Real MILP baseline using PuLP (CBC solver).

    Formulation (Big-M ordering MILP):
      Variables:
        t[i]    - continuous: scheduled time of flight i  (>= eta[i])
        y[i,j]  - binary:    1 if flight i is sequenced before flight j

      Objective:
        Minimise total delay = sum_i (t[i] - eta[i])

      Constraints for each pair (i < j):
        y[i,j] + y[j,i] = 1          (one must come first)
        t[j] >= t[i] + sep[i,j] - M*(1-y[i,j])   (separation if i before j)
        t[i] >= t[j] + sep[j,i] - M*y[i,j]        (separation if j before i)

    After the ordering is found, runway assignment is done greedily by
    refine_schedule_with_runways (same as all other methods).

    Falls back to FCFS if:
      - PuLP is not installed
      - Instance size > MAX_MILP_FLIGHTS
      - Solver reaches time limit without finding an optimal solution
    """
    flights = flights.copy().sort_values("eta_seconds").reset_index(drop=True)
    n = len(flights)
    sep = build_separation_matrix(flights["wake_class"].tolist())

    order = None
    if n <= MAX_MILP_FLIGHTS:
        order = _solve_milp(flights["eta_seconds"].tolist(), sep, n)

    if order is not None:
        flights = flights.iloc[order].reset_index(drop=True)

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


def _solve_milp(
    eta: list[float],
    sep: list[list[float]],
    n: int,
) -> list[int] | None:
    """
    Solve the ordering MILP with PuLP/CBC.
    Returns the flight indices in optimal scheduled order, or None on failure.
    """
    try:
        import pulp
    except ImportError:
        print("[MILP] pulp not installed — falling back to FCFS.")
        return None

    # Big-M: generous upper bound on any possible time difference
    max_eta = max(eta) if eta else 0.0
    total_sep = sum(sep[i][j] for i in range(n) for j in range(n))
    BIG_M = max_eta + total_sep + 1e6

    prob = pulp.LpProblem("RunwayScheduling_MILP", pulp.LpMinimize)

    # --- Decision variables ---
    # Scheduled times (continuous, lower-bounded by ETA)
    t = [pulp.LpVariable(f"t_{i}", lowBound=eta[i]) for i in range(n)]

    # Ordering binary: y[i,j] = 1 iff flight i is sequenced BEFORE flight j
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    # --- Objective: minimise total delay ---
    prob += pulp.lpSum(t[i] - eta[i] for i in range(n)), "TotalDelay"

    # --- Constraints ---
    for i in range(n):
        for j in range(i + 1, n):
            v = y[i, j]
            s_ij = sep[i][j]
            s_ji = sep[j][i]

            # Exactly one ordering must hold
            # (implicit: v ∈ {0,1} and we use complementary big-M pairs)

            # If i before j  (v=1):  t[j] >= t[i] + sep[i,j]
            prob += t[j] >= t[i] + s_ij - BIG_M * (1 - v), f"sep_ij_{i}_{j}"

            # If j before i (v=0):  t[i] >= t[j] + sep[j,i]
            prob += t[i] >= t[j] + s_ji - BIG_M * v, f"sep_ji_{i}_{j}"

    # --- Solve with time limit ---
    try:
        solver = pulp.PULP_CBC_CMD(timeLimit=60, msg=0)
        prob.solve(solver)
    except Exception as exc:
        print(f"[MILP] Solver error: {exc} — falling back to FCFS.")
        return None

    # Accept optimal or time-limit-feasible solutions
    if prob.status not in (pulp.constants.LpStatusOptimal, 1):
        print(f"[MILP] No feasible solution found (status={prob.status}) — falling back to FCFS.")
        return None

    # Extract ordering from solved scheduled times
    scheduled = [(pulp.value(t[i]), i) for i in range(n)]
    scheduled.sort(key=lambda x: (x[0] if x[0] is not None else float("inf")))
    return [idx for _, idx in scheduled]
