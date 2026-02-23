from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .runway_geofence import haversine_km
from .wake_classifier import classify_wake


@dataclass
class EventConfig:
    airport_center: Tuple[float, float]
    radius_km: float = 30.0
    landing_altitude_m: float = 1000.0
    takeoff_altitude_m: float = 1000.0
    min_speed_mps: float = 50.0


def _eta_seconds(distance_km: float, velocity_mps: float) -> float:
    if velocity_mps <= 1:
        return 0.0
    return (distance_km * 1000.0) / velocity_mps


def detect_events(df: pd.DataFrame, config: EventConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["distance_km"] = df.apply(
        lambda row: haversine_km(
            row["lat"], row["lon"], config.airport_center[0], config.airport_center[1]
        ),
        axis=1,
    )
    df = df[df["distance_km"] <= config.radius_km]
    base_df = df.copy()
    df["event_type"] = "cruise"

    landing_mask = (
        (df["geoaltitude"].fillna(0) <= config.landing_altitude_m)
        & (df["vertrate"].fillna(0) <= -1.0)
        & (df["velocity"].fillna(0) >= config.min_speed_mps)
    )
    takeoff_mask = (
        (df["geoaltitude"].fillna(0) <= config.takeoff_altitude_m)
        & (df["vertrate"].fillna(0) >= 1.0)
        & (df["velocity"].fillna(0) >= config.min_speed_mps)
    )

    df.loc[landing_mask, "event_type"] = "landing"
    df.loc[takeoff_mask, "event_type"] = "takeoff"
    df = df[df["event_type"] != "cruise"]

    if df.empty:
        # Fallback: infer events from low-altitude or on-ground points.
        relaxed = base_df.copy()
        low_alt = relaxed["geoaltitude"].fillna(0) <= 3000.0
        min_speed = relaxed["velocity"].fillna(0) >= 20.0
        relaxed = relaxed[low_alt & min_speed]
        if relaxed.empty and "onground" in base_df.columns:
            relaxed = base_df[base_df["onground"] == True]
        if relaxed.empty:
            return pd.DataFrame()
        relaxed["event_type"] = "takeoff"
        relaxed.loc[relaxed["vertrate"].fillna(0) <= 0.0, "event_type"] = "landing"
        df = relaxed

    df["eta_seconds"] = df.apply(
        lambda row: _eta_seconds(row["distance_km"], row["velocity"]), axis=1
    )
    df["wake_class"] = classify_wake(df)

    df = df.sort_values(["event_type", "eta_seconds"]).drop_duplicates(
        ["icao24", "event_type"], keep="first"
    )

    df["flight_id"] = (
        df["icao24"].astype(str)
        + "-"
        + df["event_type"].astype(str)
        + "-"
        + df["time"].astype(int).astype(str)
    )

    columns = [
        "flight_id",
        "icao24",
        "callsign",
        "event_type",
        "time",
        "eta_seconds",
        "velocity",
        "geoaltitude",
        "vertrate",
        "wake_class",
        "lat",
        "lon",
    ]
    return df[columns].reset_index(drop=True)
