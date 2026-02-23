from __future__ import annotations

import pandas as pd


def classify_wake(df: pd.DataFrame) -> pd.Series:
    # Heuristic: speed-driven wake approximation for final approach/takeoff speeds (m/s)
    def _label(speed: float | None) -> str:
        if speed is None or pd.isna(speed):
            return "M"
        if speed >= 85:   # ~165 kts
            return "H"
        if speed >= 65:   # ~125 kts
            return "M"
        return "L"

    return df["velocity"].apply(_label)
