from __future__ import annotations

import pandas as pd


def classify_wake(df: pd.DataFrame) -> pd.Series:
    # Heuristic: speed-driven wake approximation when type is unavailable.
    def _label(speed: float | None) -> str:
        if speed is None or pd.isna(speed):
            return "M"
        if speed >= 240:
            return "H"
        if speed >= 170:
            return "M"
        return "L"

    return df["velocity"].apply(_label)
