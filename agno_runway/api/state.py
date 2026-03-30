from __future__ import annotations

from dataclasses import dataclass

from agno_runway.config import AGNORSConfig, load_config_from_env
from agno_runway.models.gnn_priority import PriorityGNN, PriorityScoringService
from agno_runway.simulation.engine import RealTimeSimulationEngine
from agno_runway.simulation.scheduler import AGNORSPlusScheduler
from agno_runway.utils.aviation_rules import AviationRulesEngine


@dataclass
class RuntimeState:
    config: AGNORSConfig
    rules: AviationRulesEngine
    scorer: PriorityScoringService
    scheduler: AGNORSPlusScheduler
    simulation: RealTimeSimulationEngine


def build_runtime_state() -> RuntimeState:
    config = load_config_from_env()
    rules = AviationRulesEngine(
        min_time_gap_seconds=config.min_time_gap_seconds,
        runway_occupancy_seconds=config.runway_occupancy_seconds,
        safety_buffer_seconds=config.safety_buffer_seconds,
    )
    model = PriorityGNN(in_dim=6, hidden_dim=64, edge_dim=2)
    scorer = PriorityScoringService(
        model=model,
        rules=rules,
        time_proximity_seconds=config.time_proximity_seconds,
        emergency_priority_bonus=config.emergency_priority_bonus,
        device="cpu",
    )
    scheduler = AGNORSPlusScheduler(
        runway_count=config.runway_count,
        scorer=scorer,
        rules=rules,
    )
    simulation = RealTimeSimulationEngine(
        scheduler=scheduler,
        tick_seconds=config.simulation_tick_seconds,
        arrival_probability=config.simulation_arrival_probability,
        horizon_seconds=config.simulation_horizon_seconds,
        random_seed=config.random_seed,
    )
    return RuntimeState(
        config=config,
        rules=rules,
        scorer=scorer,
        scheduler=scheduler,
        simulation=simulation,
    )
