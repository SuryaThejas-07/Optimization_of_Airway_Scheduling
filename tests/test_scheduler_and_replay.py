from __future__ import annotations

from typing import List

from agno_runway.simulation.engine import RealTimeSimulationEngine
from agno_runway.simulation.scheduler import AGNORSPlusScheduler
from agno_runway.utils.aviation_rules import AviationRulesEngine
from agno_runway.utils.schemas import FlightInput, ScheduledFlight


class DummyScorer:
    def score_flights(self, flights: List[FlightInput]) -> List[float]:
        return [float(-f.eta_seconds) for f in flights]


def make_scheduler() -> AGNORSPlusScheduler:
    return AGNORSPlusScheduler(
        runway_count=1,
        scorer=DummyScorer(),
        rules=AviationRulesEngine(),
    )


def test_scheduler_failsafe_fallback(monkeypatch) -> None:
    scheduler = make_scheduler()

    flights = [
        FlightInput(
            flight_id="F1",
            eta_seconds=100.0,
            velocity=80.0,
            altitude=3000.0,
            wake_class="M",
            event_type="arrival",
            emergency=False,
        ),
        FlightInput(
            flight_id="F2",
            eta_seconds=101.0,
            velocity=80.0,
            altitude=3000.0,
            wake_class="L",
            event_type="arrival",
            emergency=True,
        ),
    ]

    unsafe_schedule = [
        ScheduledFlight(
            flight_id="F1",
            eta_seconds=100.0,
            scheduled_time=100.0,
            delay_seconds=0.0,
            assigned_runway=0,
            wake_class="M",
            event_type="arrival",
            priority_score=1.0,
            emergency=False,
        ),
        ScheduledFlight(
            flight_id="F2",
            eta_seconds=101.0,
            scheduled_time=101.0,
            delay_seconds=0.0,
            assigned_runway=0,
            wake_class="L",
            event_type="arrival",
            priority_score=1.0,
            emergency=True,
        ),
    ]

    monkeypatch.setattr(scheduler, "_build_schedule_policy", lambda _: unsafe_schedule)

    scheduler.add_flights(flights)
    final_schedule = scheduler.get_schedule()

    assert len(final_schedule) == 2
    assert final_schedule[0].flight_id == "F2"
    assert final_schedule[1].scheduled_time > final_schedule[0].scheduled_time


def test_engine_replay_is_deterministic() -> None:
    scheduler = make_scheduler()
    engine = RealTimeSimulationEngine(
        scheduler=scheduler,
        tick_seconds=15.0,
        arrival_probability=0.6,
        horizon_seconds=120.0,
        random_seed=7,
    )

    engine.run(duration_seconds=75.0)
    scenario = engine.export_scenario()
    baseline_decisions = engine.get_decision_log()

    scheduler2 = make_scheduler()
    replay_engine = RealTimeSimulationEngine(
        scheduler=scheduler2,
        tick_seconds=15.0,
        arrival_probability=0.1,
        horizon_seconds=60.0,
        random_seed=999,
    )
    replay_result = replay_engine.replay_scenario(scenario)

    assert replay_result["replayed_ticks"] == len(scenario.ticks)
    assert replay_engine.get_decision_log() == baseline_decisions
