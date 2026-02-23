from __future__ import annotations

import pandas as pd


def throughput_series(schedule: pd.DataFrame, window: float = 300.0) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    times = schedule["scheduled_time"].sort_values().values
    buckets = (times // window).astype(int)
    counts = pd.Series(buckets).value_counts().sort_index()
    return pd.DataFrame({"bucket": counts.index, "count": counts.values})
