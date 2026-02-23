from __future__ import annotations

import pandas as pd
import plotly.express as px


def delay_histogram(schedule: pd.DataFrame):
    if schedule.empty:
        return px.histogram()
    return px.histogram(
        schedule,
        x="delay",
        nbins=20,
        title="Delay Histogram",
    )
