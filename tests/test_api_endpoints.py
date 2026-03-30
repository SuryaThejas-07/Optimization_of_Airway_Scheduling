from __future__ import annotations

import pytest


def _get_client_and_runtime():
    pytest.importorskip("fastapi")
    pytest.importorskip("torch_geometric")
    from fastapi.testclient import TestClient
    from agno_runway.api.main import app, runtime

    return TestClient(app), runtime


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    _, runtime = _get_client_and_runtime()
    runtime.simulation.reset(clear_recorded_events=True)


def test_schedule_and_metrics() -> None:
    client, _ = _get_client_and_runtime()

    payload = {
        "flight_id": "API-001",
        "eta_seconds": 150.0,
        "velocity": 75.0,
        "altitude": 3500.0,
        "wake_class": "M",
        "event_type": "arrival",
        "emergency": False,
    }

    resp = client.post("/schedule-flight", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "assigned_runway" in body
    assert "scheduled_time" in body

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["total_flights"] >= 1


def test_simulation_scenario_export_and_replay() -> None:
    client, _ = _get_client_and_runtime()

    run_resp = client.post("/simulation/run", params={"duration_seconds": 60.0})
    assert run_resp.status_code == 200

    export_resp = client.get("/simulation/scenario")
    assert export_resp.status_code == 200
    scenario = export_resp.json()["scenario"]
    assert "random_seed" in scenario
    assert "ticks" in scenario

    reset_resp = client.post("/simulation/reset")
    assert reset_resp.status_code == 200

    replay_resp = client.post(
        "/simulation/scenario/replay",
        json=scenario,
        params={"reset_state": True},
    )
    assert replay_resp.status_code == 200
    assert replay_resp.json()["replayed_ticks"] == len(scenario["ticks"])
