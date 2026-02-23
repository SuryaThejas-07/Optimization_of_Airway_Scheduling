from __future__ import annotations

import pandas as pd
import plotly.express as px


def gantt_chart(schedule: pd.DataFrame):
    if schedule.empty:
        return px.scatter()
    df = schedule.copy()
    df["start"] = df["scheduled_time"]
    df["finish"] = df["scheduled_time"] + 1.0
    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="callsign",
        color="assigned_runway",
        title="Runway Schedule Gantt",
    )
    fig.update_yaxes(autorange="reversed")
    return fig
