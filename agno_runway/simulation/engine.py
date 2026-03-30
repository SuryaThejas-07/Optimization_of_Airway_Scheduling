from __future__ import annotations

import json
import random
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional

from agno_runway.simulation.scheduler import AGNORSPlusScheduler
from agno_runway.utils.logger import get_logger
from agno_runway.utils.schemas import (
    FlightInput,
    ScheduledFlight,
    SimulationScenario,
    SimulationScenarioTick,
)


class RealTimeSimulationEngine:
    def __init__(
        self,
        scheduler: AGNORSPlusScheduler,
        tick_seconds: float,
        arrival_probability: float,
        horizon_seconds: float,
        random_seed: int = 42,
    ) -> None:
        self.scheduler = scheduler
        self.tick_seconds = tick_seconds
        self.arrival_probability = arrival_probability
        self.horizon_seconds = horizon_seconds
        self.current_time = 0.0
        self._random_seed = random_seed
        self._rng = random.Random(self._random_seed)
        self._decision_log: List[Dict[str, float | str | int | bool]] = []
        self._event_log: List[Dict[str, object]] = []
        self._logger = get_logger("agno_rs_plus.simulation")
        self._lock = threading.RLock()
        self._live_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._live_interval_seconds = 1.0

    def step(
        self, injected_flights: Optional[List[FlightInput]] = None
    ) -> List[ScheduledFlight]:
        with self._lock:
            tick_time = self.current_time
            if injected_flights is None:
                new_flights = self._generate_flights_for_tick()
            else:
                new_flights = list(injected_flights)
            self._record_tick_event(tick_time=tick_time, flights=new_flights)
            scheduled = self.scheduler.add_flights(new_flights)

            for item in scheduled:
                self._decision_log.append(
                    {
                        "tick_time": tick_time,
                        "flight_id": item.flight_id,
                        "runway": item.assigned_runway,
                        "scheduled_time": item.scheduled_time,
                        "delay_seconds": item.delay_seconds,
                        "emergency": item.emergency,
                    }
                )

            self._logger.info(
                "Tick %.1f: generated %d flights, scheduled %d, total flights in system %d",
                tick_time,
                len(new_flights),
                len(scheduled),
                len(self.scheduler.get_schedule()),
            )
            self.current_time += self.tick_seconds
        return scheduled

    def run(self, duration_seconds: float) -> Dict[str, object]:
        ticks = int(max(0, duration_seconds) // max(1.0, self.tick_seconds))
        for _ in range(ticks):
            self.step()

        return {
            "ticks": ticks,
            "final_time": self.current_time,
            "metrics": self.scheduler.get_metrics().model_dump(),
            "decisions": self._decision_log,
        }

    def get_decision_log(self) -> List[Dict[str, float | str | int | bool]]:
        with self._lock:
            return list(self._decision_log)

    def export_scenario(self) -> SimulationScenario:
        with self._lock:
            ticks: List[SimulationScenarioTick] = []
            for item in self._event_log:
                tick_time = float(item["tick_time"])
                flights_payload = item["flights"]
                flights = [FlightInput.model_validate(f) for f in flights_payload]  # type: ignore[arg-type]
                ticks.append(
                    SimulationScenarioTick(tick_time=tick_time, flights=flights)
                )

            return SimulationScenario(
                random_seed=self._random_seed,
                tick_seconds=self.tick_seconds,
                arrival_probability=self.arrival_probability,
                horizon_seconds=self.horizon_seconds,
                start_time=0.0,
                ticks=ticks,
            )

    def save_scenario(self, file_path: str | Path) -> str:
        scenario = self.export_scenario()
        output = Path(file_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(scenario.model_dump(), indent=2),
            encoding="utf-8",
        )
        return str(output)

    def replay_scenario(
        self,
        scenario: SimulationScenario | dict,
        reset_state: bool = True,
    ) -> Dict[str, object]:
        data = (
            scenario
            if isinstance(scenario, SimulationScenario)
            else SimulationScenario.model_validate(scenario)
        )

        self.stop_live()

        with self._lock:
            if reset_state:
                self.scheduler.reset()
                self.current_time = data.start_time
                self._decision_log = []
                self._event_log = []

            self._random_seed = data.random_seed
            self.tick_seconds = data.tick_seconds
            self.arrival_probability = data.arrival_probability
            self.horizon_seconds = data.horizon_seconds
            self._rng = random.Random(self._random_seed)

        for tick in data.ticks:
            with self._lock:
                self.current_time = float(tick.tick_time)
            flights = [FlightInput.model_validate(f) for f in tick.flights]
            self.step(injected_flights=flights)

        return {
            "replayed_ticks": len(data.ticks),
            "final_time": self.current_time,
            "metrics": self.scheduler.get_metrics().model_dump(),
            "decisions": self.get_decision_log(),
        }

    def load_and_replay_scenario(
        self,
        file_path: str | Path,
        reset_state: bool = True,
    ) -> Dict[str, object]:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return self.replay_scenario(payload, reset_state=reset_state)

    def reset(self, clear_recorded_events: bool = True) -> None:
        self.stop_live()
        with self._lock:
            self.scheduler.reset()
            self.current_time = 0.0
            self._decision_log = []
            if clear_recorded_events:
                self._event_log = []

    def get_recent_decisions(
        self, limit: int = 20
    ) -> List[Dict[str, float | str | int | bool]]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._decision_log[-limit:])

    def start_live(
        self, loop_interval_seconds: float = 1.0, max_ticks: Optional[int] = None
    ) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._stop_event.clear()
            self._live_interval_seconds = max(0.05, float(loop_interval_seconds))

        def _runner() -> None:
            ticks = 0
            try:
                while not self._stop_event.is_set():
                    self.step()
                    ticks += 1
                    if max_ticks is not None and ticks >= max_ticks:
                        break
                    time.sleep(self._live_interval_seconds)
            finally:
                with self._lock:
                    self._running = False
                    self._stop_event.set()

        self._live_thread = threading.Thread(target=_runner, daemon=True)
        self._live_thread.start()
        self._logger.info(
            "Live simulation started (interval=%.2fs)", self._live_interval_seconds
        )
        return True

    def stop_live(self) -> bool:
        with self._lock:
            if not self._running:
                return False
            thread = self._live_thread
            self._stop_event.set()

        if thread is not None:
            thread.join(timeout=max(1.0, self._live_interval_seconds * 2))

        with self._lock:
            self._running = False
        self._logger.info("Live simulation stopped")
        return True

    def live_status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "running": self._running,
                "current_time": self.current_time,
                "random_seed": self._random_seed,
                "tick_seconds": self.tick_seconds,
                "loop_interval_seconds": self._live_interval_seconds,
                "decision_count": len(self._decision_log),
                "recorded_ticks": len(self._event_log),
                "scheduled_flights": len(self.scheduler.get_schedule()),
            }

    def _record_tick_event(self, tick_time: float, flights: List[FlightInput]) -> None:
        self._event_log.append(
            {
                "tick_time": float(tick_time),
                "flights": [f.model_dump() for f in flights],
            }
        )

    def _generate_flights_for_tick(self) -> List[FlightInput]:
        generated: List[FlightInput] = []
        arrivals = 0

        while self._rng.random() < self.arrival_probability and arrivals < 3:
            arrivals += 1
            wake = self._rng.choices(["H", "M", "L"], weights=[0.15, 0.45, 0.4], k=1)[0]
            event_type = self._rng.choice(["arrival", "departure"])
            eta = self.current_time + self._rng.uniform(30.0, self.horizon_seconds)
            velocity = self._rng.uniform(60.0, 95.0)
            altitude = (
                self._rng.uniform(0.0, 14000.0)
                if event_type == "arrival"
                else self._rng.uniform(0.0, 3500.0)
            )
            emergency = self._rng.random() < 0.03
            flight_id = f"SIM-{int(self.current_time):05d}-{arrivals}"

            generated.append(
                FlightInput(
                    flight_id=flight_id,
                    eta_seconds=eta,
                    velocity=velocity,
                    altitude=altitude,
                    wake_class=wake,
                    event_type=event_type,
                    emergency=emergency,
                )
            )

        return generated
