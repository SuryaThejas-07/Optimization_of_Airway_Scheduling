from __future__ import annotations

from fastapi import FastAPI, HTTPException

from agno_runway.api.state import build_runtime_state
from agno_runway.utils.logger import get_logger
from agno_runway.utils.schemas import (
    FlightInput,
    MetricsResponse,
    RunwayStatusResponse,
    ScheduleFlightResponse,
    SimulationScenario,
)


app = FastAPI(title="AI-Powered Runway Scheduling System (AGNO-RS+)", version="1.0.0")
runtime = build_runtime_state()
logger = get_logger("agno_rs_plus.api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "AGNO-RS+"}


@app.post("/schedule-flight", response_model=ScheduleFlightResponse)
def schedule_flight(payload: FlightInput) -> ScheduleFlightResponse:
    existing_ids = {f.flight_id for f in runtime.scheduler.get_schedule()}
    if payload.flight_id in existing_ids:
        raise HTTPException(
            status_code=409, detail=f"Flight {payload.flight_id} already exists"
        )

    scheduled = runtime.scheduler.add_flight(payload)
    logger.info(
        "Scheduled flight %s on runway %d at %.2f (delay %.2f)",
        scheduled.flight_id,
        scheduled.assigned_runway + 1,
        scheduled.scheduled_time,
        scheduled.delay_seconds,
    )
    return ScheduleFlightResponse(
        assigned_runway=scheduled.assigned_runway,
        scheduled_time=scheduled.scheduled_time,
        delay_seconds=scheduled.delay_seconds,
        priority_score=scheduled.priority_score,
    )


@app.get("/runway-status", response_model=RunwayStatusResponse)
def runway_status() -> RunwayStatusResponse:
    return RunwayStatusResponse(
        runway_count=runtime.scheduler.runway_count,
        schedules=runtime.scheduler.get_runway_schedule(),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return runtime.scheduler.get_metrics()


@app.post("/simulation/step")
def simulation_step() -> dict[str, object]:
    scheduled = runtime.simulation.step()
    return {
        "scheduled_count": len(scheduled),
        "current_time": runtime.simulation.current_time,
        "metrics": runtime.scheduler.get_metrics().model_dump(),
    }


@app.post("/simulation/run")
def simulation_run(duration_seconds: float = 120.0) -> dict[str, object]:
    if duration_seconds <= 0:
        raise HTTPException(status_code=400, detail="duration_seconds must be > 0")
    return runtime.simulation.run(duration_seconds)


@app.post("/simulation/live/start")
def simulation_live_start(
    loop_interval_seconds: float = 1.0, max_ticks: int | None = None
) -> dict[str, object]:
    if loop_interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="loop_interval_seconds must be > 0")
    if max_ticks is not None and max_ticks <= 0:
        raise HTTPException(status_code=400, detail="max_ticks must be > 0")

    started = runtime.simulation.start_live(
        loop_interval_seconds=loop_interval_seconds,
        max_ticks=max_ticks,
    )
    return {
        "started": started,
        "status": runtime.simulation.live_status(),
        "message": "live simulation started" if started else "already running",
    }


@app.post("/simulation/live/stop")
def simulation_live_stop() -> dict[str, object]:
    stopped = runtime.simulation.stop_live()
    return {
        "stopped": stopped,
        "status": runtime.simulation.live_status(),
        "message": "live simulation stopped" if stopped else "not running",
    }


@app.get("/simulation/live/status")
def simulation_live_status() -> dict[str, object]:
    return runtime.simulation.live_status()


@app.get("/simulation/live/decisions")
def simulation_live_decisions(limit: int = 20) -> dict[str, object]:
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")
    return {
        "limit": limit,
        "decisions": runtime.simulation.get_recent_decisions(limit=limit),
    }


@app.get("/simulation/scenario")
def simulation_scenario_export() -> dict[str, object]:
    scenario = runtime.simulation.export_scenario()
    return {
        "scenario": scenario.model_dump(),
        "recorded_ticks": len(scenario.ticks),
    }


@app.post("/simulation/scenario/replay")
def simulation_scenario_replay(
    scenario: SimulationScenario,
    reset_state: bool = True,
) -> dict[str, object]:
    return runtime.simulation.replay_scenario(scenario, reset_state=reset_state)


@app.post("/simulation/reset")
def simulation_reset(clear_recorded_events: bool = True) -> dict[str, object]:
    runtime.simulation.reset(clear_recorded_events=clear_recorded_events)
    return {
        "reset": True,
        "status": runtime.simulation.live_status(),
    }
