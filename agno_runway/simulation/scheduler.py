from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from agno_runway.utils.logger import get_logger
from agno_runway.utils.aviation_rules import AviationRulesEngine
from agno_runway.utils.schemas import FlightInput, MetricsResponse, ScheduledFlight

if TYPE_CHECKING:
    from agno_runway.models.gnn_priority import PriorityScoringService


class AGNORSPlusScheduler:
    def __init__(
        self,
        runway_count: int,
        scorer: PriorityScoringService,
        rules: AviationRulesEngine,
    ) -> None:
        self.runway_count = runway_count
        self.scorer = scorer
        self.rules = rules
        self._flights: List[FlightInput] = []
        self._schedule: List[ScheduledFlight] = []
        self._lock = threading.RLock()
        self._logger = get_logger("agno_rs_plus.scheduler")

    def add_flight(self, flight: FlightInput) -> ScheduledFlight:
        with self._lock:
            self._flights.append(flight)
            self._schedule = self._build_schedule_with_failsafe(self._flights)
            for item in self._schedule:
                if item.flight_id == flight.flight_id:
                    return item
        raise RuntimeError("Flight was added but no schedule entry was found.")

    def add_flights(self, flights: List[FlightInput]) -> List[ScheduledFlight]:
        if not flights:
            return []
        with self._lock:
            self._flights.extend(flights)
            self._schedule = self._build_schedule_with_failsafe(self._flights)
            added_ids = {f.flight_id for f in flights}
            return [s for s in self._schedule if s.flight_id in added_ids]

    def reschedule_all(self) -> List[ScheduledFlight]:
        with self._lock:
            self._schedule = self._build_schedule_with_failsafe(self._flights)
            return list(self._schedule)

    def reset(self) -> None:
        with self._lock:
            self._flights = []
            self._schedule = []

    def get_schedule(self) -> List[ScheduledFlight]:
        with self._lock:
            return list(self._schedule)

    def get_runway_schedule(self) -> Dict[str, List[ScheduledFlight]]:
        with self._lock:
            runways: Dict[str, List[ScheduledFlight]] = {
                f"RWY_{i+1:02d}": [] for i in range(self.runway_count)
            }
            for flight in self._schedule:
                runways[f"RWY_{flight.assigned_runway + 1:02d}"].append(flight)
            for key in runways:
                runways[key] = sorted(runways[key], key=lambda x: x.scheduled_time)
            return runways

    def get_metrics(self) -> MetricsResponse:
        with self._lock:
            total = len(self._schedule)
            if total == 0:
                return MetricsResponse(
                    total_flights=0,
                    total_delay_seconds=0.0,
                    average_delay_seconds=0.0,
                    throughput_flights_per_second=0.0,
                    safety_violations=0,
                )

            total_delay = sum(s.delay_seconds for s in self._schedule)
            avg_delay = total_delay / total
            t_min = min(s.scheduled_time for s in self._schedule)
            t_max = max(s.scheduled_time for s in self._schedule)
            horizon = max(1.0, t_max - t_min)
            throughput = total / horizon

            violations = 0
            runway_map: Dict[str, List[ScheduledFlight]] = {
                f"RWY_{i+1:02d}": [] for i in range(self.runway_count)
            }
            for flight in self._schedule:
                runway_map[f"RWY_{flight.assigned_runway + 1:02d}"].append(flight)
            for runway_flights in runway_map.values():
                runway_flights.sort(key=lambda x: x.scheduled_time)
                for i in range(1, len(runway_flights)):
                    leader = runway_flights[i - 1]
                    follower = runway_flights[i]
                    if not self.rules.is_safe_interval(
                        leader_time=leader.scheduled_time,
                        follower_time=follower.scheduled_time,
                        leader_wake=leader.wake_class,
                        follower_wake=follower.wake_class,
                    ):
                        violations += 1

            return MetricsResponse(
                total_flights=total,
                total_delay_seconds=total_delay,
                average_delay_seconds=avg_delay,
                throughput_flights_per_second=throughput,
                safety_violations=violations,
            )

    def _build_schedule_with_failsafe(
        self, flights: List[FlightInput]
    ) -> List[ScheduledFlight]:
        try:
            primary = self._build_schedule_policy(flights)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.exception(
                "Primary scheduling failed; switching to fail-safe policy: %s", exc
            )
            return self._build_schedule_failsafe(flights)

        violations = self._count_safety_violations(primary)
        if violations == 0 and len(primary) == len(flights):
            return primary

        self._logger.warning(
            "Primary schedule rejected (violations=%d, scheduled=%d, flights=%d); "
            "switching to fail-safe policy",
            violations,
            len(primary),
            len(flights),
        )
        return self._build_schedule_failsafe(flights)

    def _build_schedule_policy(
        self, flights: List[FlightInput]
    ) -> List[ScheduledFlight]:
        if not flights:
            return []

        scores = self.scorer.score_flights(flights)
        ranked = sorted(
            zip(flights, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        runway_last: Dict[int, Optional[ScheduledFlight]] = {
            i: None for i in range(self.runway_count)
        }
        scheduled: List[ScheduledFlight] = []

        for flight, score in ranked:
            runway, scheduled_time = self._best_runway_assignment(flight, runway_last)
            delay = max(0.0, scheduled_time - flight.eta_seconds)
            assigned = ScheduledFlight(
                flight_id=flight.flight_id,
                eta_seconds=flight.eta_seconds,
                scheduled_time=scheduled_time,
                delay_seconds=delay,
                assigned_runway=runway,
                wake_class=flight.wake_class,
                event_type=flight.event_type,
                priority_score=float(score),
                emergency=flight.emergency,
            )
            scheduled.append(assigned)
            runway_last[runway] = assigned

        return sorted(scheduled, key=lambda x: x.scheduled_time)

    def _build_schedule_failsafe(
        self, flights: List[FlightInput]
    ) -> List[ScheduledFlight]:
        if not flights:
            return []

        # Deterministic emergency-aware FCFS fallback when primary policy is unsafe.
        ranked = sorted(
            flights,
            key=lambda f: (
                0 if f.emergency else 1,
                f.eta_seconds,
                f.flight_id,
            ),
        )

        runway_last: Dict[int, Optional[ScheduledFlight]] = {
            i: None for i in range(self.runway_count)
        }
        scheduled: List[ScheduledFlight] = []

        for rank, flight in enumerate(ranked):
            runway, scheduled_time = self._best_runway_assignment(flight, runway_last)
            delay = max(0.0, scheduled_time - flight.eta_seconds)
            score = float(len(ranked) - rank)
            assigned = ScheduledFlight(
                flight_id=flight.flight_id,
                eta_seconds=flight.eta_seconds,
                scheduled_time=scheduled_time,
                delay_seconds=delay,
                assigned_runway=runway,
                wake_class=flight.wake_class,
                event_type=flight.event_type,
                priority_score=score,
                emergency=flight.emergency,
            )
            scheduled.append(assigned)
            runway_last[runway] = assigned

        return sorted(scheduled, key=lambda x: x.scheduled_time)

    def _count_safety_violations(self, schedule: List[ScheduledFlight]) -> int:
        if not schedule:
            return 0

        runway_map: Dict[str, List[ScheduledFlight]] = {
            f"RWY_{i+1:02d}": [] for i in range(self.runway_count)
        }
        for flight in schedule:
            runway_map[f"RWY_{flight.assigned_runway + 1:02d}"].append(flight)

        violations = 0
        for runway_flights in runway_map.values():
            runway_flights.sort(key=lambda x: x.scheduled_time)
            for i in range(1, len(runway_flights)):
                leader = runway_flights[i - 1]
                follower = runway_flights[i]
                if not self.rules.is_safe_interval(
                    leader_time=leader.scheduled_time,
                    follower_time=follower.scheduled_time,
                    leader_wake=leader.wake_class,
                    follower_wake=follower.wake_class,
                ):
                    violations += 1

        return violations

    def _best_runway_assignment(
        self,
        flight: FlightInput,
        runway_last: Dict[int, Optional[ScheduledFlight]],
    ) -> Tuple[int, float]:
        best_runway = 0
        best_time = float("inf")

        for runway_idx in range(self.runway_count):
            previous = runway_last[runway_idx]
            candidate = flight.eta_seconds

            if previous is not None:
                wake_gap = self.rules.required_gap(
                    previous.wake_class, flight.wake_class
                )
                occupancy_gap = (
                    self.rules.runway_occupancy_seconds
                    + self.rules.safety_buffer_seconds
                )
                candidate = max(
                    candidate,
                    previous.scheduled_time + wake_gap,
                    previous.scheduled_time + occupancy_gap,
                )

            if candidate < best_time:
                best_time = candidate
                best_runway = runway_idx

        return best_runway, best_time
