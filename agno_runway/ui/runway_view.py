from __future__ import annotations

import pandas as pd
import plotly.express as px


def _apply_time_axis(df: pd.DataFrame, base_time: float | None) -> pd.DataFrame:
    df = df.copy()
    if base_time is None:
        df["start"] = df["scheduled_time"]
        df["finish"] = df["scheduled_time"] + 1.0
        return df
    base = pd.to_datetime(base_time, unit="s")
    df["start"] = (base + pd.to_timedelta(df["scheduled_time"], unit="s")).dt.floor("s")
    df["finish"] = (base + pd.to_timedelta(df["scheduled_time"] + 1.0, unit="s")).dt.floor("s")
    return df


def _add_flight_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].fillna("").astype(str).str.strip()
    if "aircraft" in df.columns:
        df["aircraft"] = df["aircraft"].fillna("").astype(str).str.strip()
    if "icao24" in df.columns:
        df["icao24"] = df["icao24"].fillna("").astype(str).str.strip()
    callsign = df.get("callsign", pd.Series([""] * len(df)))
    aircraft = df.get("aircraft", pd.Series([""] * len(df)))
    icao24 = df.get("icao24", pd.Series([""] * len(df)))
    df["flight_label"] = callsign.where(callsign != "", aircraft)
    df["flight_label"] = df["flight_label"].where(df["flight_label"] != "", icao24)
    df["flight_label"] = df["flight_label"].where(df["flight_label"] != "", "UNKNOWN")
    return df


def runway_timeline(schedule: pd.DataFrame, base_time: float | None = None):
    if schedule.empty:
        return px.scatter()
    df = _add_flight_label(_apply_time_axis(schedule, base_time))
    hover_cols = [col for col in ["flight_label", "callsign", "aircraft", "icao24", "delay"] if col in df.columns]
    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="assigned_runway",
        color="wake_class",
        hover_data=hover_cols,
        title="Runway Utilization",
    )
    fig.update_traces(text=df["flight_label"], textposition="inside")
    fig.update_yaxes(autorange="reversed")
    return fig


def arrivals_departures_timeline(schedule: pd.DataFrame, base_time: float | None = None):
    if schedule.empty:
        return px.scatter()
    df = _add_flight_label(_apply_time_axis(schedule, base_time))
    hover_cols = [col for col in ["flight_label", "callsign", "aircraft", "icao24", "delay", "priority_score"] if col in df.columns]
    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="assigned_runway",
        color="event_type",
        hover_data=hover_cols,
        title="Arrivals vs Departures by Runway",
    )
    fig.update_traces(text=df["flight_label"], textposition="inside")
    fig.update_yaxes(autorange="reversed")
    return fig


def runway_free_intervals(schedule: pd.DataFrame, base_time: float | None = None):
    if schedule.empty:
        return px.scatter()
    rows = []
    for runway, group in schedule.groupby("assigned_runway"):
        ordered = group.sort_values("scheduled_time")
        times = ordered["scheduled_time"].values
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]
            rows.append({"runway": runway, "start": times[i - 1], "finish": times[i], "gap": gap})
    if not rows:
        return px.scatter()
    df = pd.DataFrame(rows)
    if base_time is not None:
        base = pd.to_datetime(base_time, unit="s")
        df["start"] = (base + pd.to_timedelta(df["start"], unit="s")).dt.floor("s")
        df["finish"] = (base + pd.to_timedelta(df["finish"], unit="s")).dt.floor("s")
    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="runway",
        color="gap",
        title="Runway Free-Time Intervals",
    )
    fig.update_yaxes(autorange="reversed")
    return fig
