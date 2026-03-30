from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


WakeClass = Literal["H", "M", "L"]
EventType = Literal["arrival", "departure"]


class FlightInput(BaseModel):
    flight_id: str = Field(..., description="Unique flight identifier")
    eta_seconds: float = Field(..., ge=0)
    velocity: float = Field(..., ge=0)
    altitude: float = Field(..., ge=0)
    wake_class: WakeClass
    event_type: EventType = "arrival"
    emergency: bool = False


class ScheduledFlight(BaseModel):
    flight_id: str
    eta_seconds: float
    scheduled_time: float
    delay_seconds: float
    assigned_runway: int
    wake_class: WakeClass
    event_type: EventType
    priority_score: float
    emergency: bool = False


class ScheduleFlightResponse(BaseModel):
    assigned_runway: int
    scheduled_time: float
    delay_seconds: float
    priority_score: float


class RunwayStatusResponse(BaseModel):
    runway_count: int
    schedules: Dict[str, List[ScheduledFlight]]


class MetricsResponse(BaseModel):
    total_flights: int
    total_delay_seconds: float
    average_delay_seconds: float
    throughput_flights_per_second: float
    safety_violations: int


class SimulationScenarioTick(BaseModel):
    tick_time: float = Field(..., ge=0)
    flights: List[FlightInput]


class SimulationScenario(BaseModel):
    random_seed: int
    tick_seconds: float = Field(..., gt=0)
    arrival_probability: float = Field(..., ge=0, le=1)
    horizon_seconds: float = Field(..., gt=0)
    start_time: float = Field(0.0, ge=0)
    ticks: List[SimulationScenarioTick]
