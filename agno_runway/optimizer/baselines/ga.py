from __future__ import annotations

import random
import pandas as pd
import torch

from agno_runway.data.separation_builder import build_separation_matrix
from agno_runway.optimizer.robust_refiner import refine_schedule_with_runways


def ga_schedule(
    flights: pd.DataFrame,
    generations: int = 30,
    pop_size: int = 20,
    runway_count: int = 2,
) -> pd.DataFrame:
    flights = flights.copy()
    indices = list(range(len(flights)))

    def fitness(order: list[int]) -> float:
        eta = flights.loc[order, "eta_seconds"].values
        delay = (eta.max() - eta.min())
        return -delay

    population = [random.sample(indices, len(indices)) for _ in range(pop_size)]
    for _ in range(generations):
        scored = sorted(population, key=fitness, reverse=True)
        elites = scored[: max(2, pop_size // 5)]
        population = elites[:]
        while len(population) < pop_size:
            parent = random.choice(elites)
            child = parent[:]
            i, j = random.sample(range(len(child)), 2)
            child[i], child[j] = child[j], child[i]
            population.append(child)

    best = max(population, key=fitness)
    flights = flights.loc[best].reset_index(drop=True)
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
