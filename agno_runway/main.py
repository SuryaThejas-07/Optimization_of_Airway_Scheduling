from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from agno_runway.data.event_extractor import EventConfig, detect_events
from agno_runway.data.loader import load_states, save_metadata
from agno_runway.data.separation_builder import build_separation_matrix, save_separation
from agno_runway.optimizer.graph_model import AGNOModel, schedule_with_agnos, build_features
from agno_runway.optimizer.nis_optimizer import schedule_with_nis
from agno_runway.optimizer.baselines.fcfs import fcfs_schedule
from agno_runway.optimizer.baselines.milp import milp_schedule
from agno_runway.optimizer.baselines.ga import ga_schedule
from agno_runway.analytics.metrics import compute_metrics


def _wake_onehot(series: pd.Series) -> torch.Tensor:
    mapping = {"H": 0, "M": 1, "L": 2}
    idx = series.map(mapping).fillna(1).astype(int).values
    onehot = torch.zeros((len(idx), 3))
    onehot[range(len(idx)), idx] = 1.0
    return onehot


def _add_composite_scores(metrics: dict) -> dict:
    keys = ["total_delay", "avg_delay", "throughput", "avg_safety_slack", "min_safety_slack"]
    values = {key: [] for key in keys}
    for method, vals in metrics.items():
        for key in keys:
            values[key].append(vals.get(key, 0.0))

    def _norm(arr, higher_better: bool) -> list[float]:
        arr = pd.Series(arr, dtype=float)
        if arr.max() == arr.min():
            return [0.0 for _ in arr]
        scaled = (arr - arr.min()) / (arr.max() - arr.min())
        return scaled.tolist() if higher_better else (1.0 - scaled).tolist()

    delay_n = _norm(values["total_delay"], higher_better=False)
    avg_delay_n = _norm(values["avg_delay"], higher_better=False)
    throughput_n = _norm(values["throughput"], higher_better=True)
    safety_n = _norm(values["avg_safety_slack"], higher_better=True)
    min_safety_n = _norm(values["min_safety_slack"], higher_better=True)

    for i, method in enumerate(metrics.keys()):
        metrics[method]["composite_score"] = float(
            0.35 * delay_n[i]
            + 0.25 * avg_delay_n[i]
            + 0.2 * throughput_n[i]
            + 0.1 * safety_n[i]
            + 0.1 * min_safety_n[i]
        )
    return metrics


def _pick_best_method(metrics: dict) -> str:
    best = None
    best_score = None
    for method, vals in metrics.items():
        score = vals.get("composite_score", 0.0)
        if best is None or score > best_score:
            best = method
            best_score = score
    return best or "AGNO-RS"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--airport-lat", type=float, default=None)
    parser.add_argument("--airport-lon", type=float, default=None)
    parser.add_argument("--radius-km", type=float, default=30.0)
    parser.add_argument("--runways", type=int, default=2)
    parser.add_argument("--optimize-seconds", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = (
        Path(args.data)
        if args.data
        else root.parent / "states_2022-06-27-23.csv"
    )
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    states, airport = load_states(
        data_path,
        airport_lat=args.airport_lat,
        airport_lon=args.airport_lon,
        radius_km=args.radius_km,
    )

    if airport is None:
        raise RuntimeError("Could not infer airport location from data.")

    config = EventConfig(airport_center=airport, radius_km=args.radius_km)
    flights = detect_events(states, config)
    if flights.empty:
        raise RuntimeError("No landing or takeoff events detected.")

    flights = flights.reset_index(drop=True)
    flights["flight_index"] = flights.index

    flights_path = outputs / "flights.csv"
    flights.to_csv(flights_path, index=False)

    classes = flights["wake_class"].tolist()
    separation_path = outputs / "separation.json"
    save_separation(separation_path, classes)

    metadata = {
        "airport_center": airport,
        "records": len(states),
        "events": len(flights),
    }
    save_metadata(outputs / "metadata.json", metadata)

    device = _resolve_device(args.device)
    print(f"Using device: {device.type}")
    model = AGNOModel(feature_dim=6, hidden_dim=64).to(device)

    model_path = outputs / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        print(f"Loaded trained model weights from {model_path}")
    else:
        print("No trained weights found — using random initialisation. Run gpu_train.py to train the model.")

    eta = torch.tensor(flights["eta_seconds"].values, dtype=torch.float32, device=device)
    velocity = torch.tensor(flights["velocity"].values, dtype=torch.float32, device=device)
    altitude = torch.tensor(
        flights["geoaltitude"].fillna(0).values, dtype=torch.float32, device=device
    )
    wake_onehot = _wake_onehot(flights["wake_class"]).to(device)
    sep_matrix = torch.tensor(build_separation_matrix(classes), dtype=torch.float32, device=device)
    features = build_features(eta, velocity, altitude, wake_onehot)
    adj = (sep_matrix > 0).float()
    with torch.no_grad():
        scores = model(features, adj).detach().cpu().tolist()

    result = schedule_with_agnos(
        model,
        eta,
        velocity,
        altitude,
        wake_onehot,
        sep_matrix,
        runway_count=args.runways,
        optimize_seconds=0.0,
    )

    nis_result = schedule_with_nis(
        model,
        eta,
        velocity,
        altitude,
        wake_onehot,
        sep_matrix,
        runway_count=args.runways,
        optimize_seconds=args.optimize_seconds,
    )

    def _build_schedule(result_obj, label: str) -> list[dict]:
        schedule_rows = []
        for idx, order_idx in enumerate(result_obj.order):
            row = flights.iloc[order_idx]
            schedule_rows.append(
                {
                    "aircraft": row["icao24"],
                    "callsign": row["callsign"].strip()
                    if isinstance(row["callsign"], str)
                    else "",
                    "event_type": row["event_type"],
                    "assigned_runway": f"RWY_{result_obj.assigned_runways[idx] + 1:02d}",
                    "scheduled_time": float(result_obj.scheduled_times[idx]),
                    "order": int(idx),
                    "delay": float(result_obj.delays[idx]),
                    "safety_margin": float(result_obj.safety_margins[idx]),
                    "wake_class": row["wake_class"],
                    "priority_score": float(scores[order_idx]),
                    "eta_seconds": float(row["eta_seconds"]),
                    "flight_index": int(row["flight_index"]),
                    "method": label,
                }
            )
        return schedule_rows

    schedule = _build_schedule(result, "AGNO-RS")
    nis_schedule = _build_schedule(nis_result, "NIS-LNS")

    schedule_path = outputs / "schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2))
    (outputs / "nis_schedule.json").write_text(json.dumps(nis_schedule, indent=2))

    flights_for_baselines = flights.copy()
    fcfs = fcfs_schedule(flights_for_baselines, runway_count=args.runways)
    milp = milp_schedule(flights_for_baselines, runway_count=args.runways)
    ga = ga_schedule(flights_for_baselines, runway_count=args.runways)

    metrics = {
        "AGNO-RS": compute_metrics(pd.DataFrame(schedule), sep_matrix.tolist()),
        "NIS-LNS": compute_metrics(pd.DataFrame(nis_schedule), sep_matrix.tolist()),
        "FCFS": compute_metrics(fcfs, sep_matrix.tolist()),
        "MILP": compute_metrics(milp, sep_matrix.tolist()),
        "GA": compute_metrics(ga, sep_matrix.tolist()),
    }

    metrics = _add_composite_scores(metrics)
    (outputs / "baselines.json").write_text(json.dumps(metrics, indent=2))
    baseline_schedules = {
        "AGNO-RS": schedule,
        "NIS-LNS": nis_schedule,
        "FCFS": fcfs.to_dict(orient="records"),
        "MILP": milp.to_dict(orient="records"),
        "GA": ga.to_dict(orient="records"),
    }
    (outputs / "baseline_schedules.json").write_text(
        json.dumps(baseline_schedules, indent=2)
    )

    best_method = _pick_best_method(metrics)
    best_schedule = baseline_schedules[best_method]
    (outputs / "best_method.json").write_text(json.dumps({"method": best_method}, indent=2))
    (outputs / "best_schedule.json").write_text(json.dumps(best_schedule, indent=2))

    if not args.no_ui:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(root / "ui" / "app.py"),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
