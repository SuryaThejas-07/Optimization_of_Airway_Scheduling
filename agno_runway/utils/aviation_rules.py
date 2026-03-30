from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class AviationRulesEngine:
    min_time_gap_seconds: float = 30.0
    runway_occupancy_seconds: float = 50.0
    safety_buffer_seconds: float = 10.0

    # Simplified ICAO-style wake separation matrix in seconds.
    separation_matrix: Dict[str, Dict[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.separation_matrix is None:
            self.separation_matrix = {
                "H": {"H": 120.0, "M": 150.0, "L": 180.0},
                "M": {"H": 90.0, "M": 120.0, "L": 150.0},
                "L": {"H": 90.0, "M": 90.0, "L": 90.0},
            }

    def required_wake_gap(self, leader_wake: str, follower_wake: str) -> float:
        leader = leader_wake if leader_wake in self.separation_matrix else "M"
        follower = (
            follower_wake if follower_wake in self.separation_matrix[leader] else "M"
        )
        return self.separation_matrix[leader][follower]

    def required_gap(self, leader_wake: str, follower_wake: str) -> float:
        wake_gap = self.required_wake_gap(leader_wake, follower_wake)
        occupancy_gap = self.runway_occupancy_seconds + self.safety_buffer_seconds
        return max(
            wake_gap + self.safety_buffer_seconds,
            occupancy_gap,
            self.min_time_gap_seconds,
        )

    def is_safe_interval(
        self,
        leader_time: float,
        follower_time: float,
        leader_wake: str,
        follower_wake: str,
    ) -> bool:
        return (follower_time - leader_time) >= self.required_gap(
            leader_wake, follower_wake
        )
