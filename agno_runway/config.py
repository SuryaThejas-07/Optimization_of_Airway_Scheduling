from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class AGNORSConfig:
    runway_count: int = 2
    min_time_gap_seconds: float = 30.0
    runway_occupancy_seconds: float = 50.0
    safety_buffer_seconds: float = 10.0
    time_proximity_seconds: float = 300.0
    simulation_tick_seconds: float = 15.0
    simulation_arrival_probability: float = 0.35
    simulation_horizon_seconds: float = 900.0
    emergency_priority_bonus: float = 1000.0
    random_seed: int = 42


def load_config_from_env() -> AGNORSConfig:
    return AGNORSConfig(
        runway_count=int(os.getenv("AGNORS_RUNWAY_COUNT", "2")),
        min_time_gap_seconds=float(os.getenv("AGNORS_MIN_TIME_GAP_SECONDS", "30")),
        runway_occupancy_seconds=float(
            os.getenv("AGNORS_RUNWAY_OCCUPANCY_SECONDS", "50")
        ),
        safety_buffer_seconds=float(os.getenv("AGNORS_SAFETY_BUFFER_SECONDS", "10")),
        time_proximity_seconds=float(os.getenv("AGNORS_TIME_PROXIMITY_SECONDS", "300")),
        simulation_tick_seconds=float(os.getenv("AGNORS_SIM_TICK_SECONDS", "15")),
        simulation_arrival_probability=float(
            os.getenv("AGNORS_SIM_ARRIVAL_PROBABILITY", "0.35")
        ),
        simulation_horizon_seconds=float(
            os.getenv("AGNORS_SIM_HORIZON_SECONDS", "900")
        ),
        emergency_priority_bonus=float(
            os.getenv("AGNORS_EMERGENCY_PRIORITY_BONUS", "1000")
        ),
        random_seed=int(os.getenv("AGNORS_RANDOM_SEED", "42")),
    )
