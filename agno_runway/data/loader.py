from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .runway_geofence import bbox_from_center
from .data_optimizer import ParquetConverter, DataCache

EXPECTED_COLUMNS = [
    "time",
    "icao24",
    "lat",
    "lon",
    "velocity",
    "heading",
    "vertrate",
    "callsign",
    "onground",
    "alert",
    "spi",
    "squawk",
    "baroaltitude",
    "geoaltitude",
    "lastposupdate",
    "lastcontact",
]


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "time",
        "lat",
        "lon",
        "velocity",
        "heading",
        "vertrate",
        "baroaltitude",
        "geoaltitude",
        "lastposupdate",
        "lastcontact",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "onground" in df.columns:
        df["onground"] = df["onground"].astype(bool, errors="ignore")
    return df


def infer_airport_from_onground(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    if "onground" not in df.columns:
        return None
    ground = df[df["onground"] == True]
    if ground.empty:
        return None
    ground = ground.dropna(subset=["lat", "lon"])
    if ground.empty:
        return None
    # Use coarse grid clustering to find the densest airport region.
    bin_size = 0.1
    ground["lat_bin"] = (ground["lat"] / bin_size).round() * bin_size
    ground["lon_bin"] = (ground["lon"] / bin_size).round() * bin_size
    counts = ground.groupby(["lat_bin", "lon_bin"]).size().sort_values(ascending=False)
    if counts.empty:
        return None
    top_lat, top_lon = counts.index[0]
    cluster = ground[(ground["lat_bin"] == top_lat) & (ground["lon_bin"] == top_lon)]
    return (cluster["lat"].mean(), cluster["lon"].mean())


def load_states(
    file_path: str | Path,
    airport_lat: Optional[float] = None,
    airport_lon: Optional[float] = None,
    radius_km: float = 30.0,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: str | Path = "cache",
    prefer_parquet: bool = True,
) -> tuple[pd.DataFrame, Optional[Tuple[float, float]]]:
    """
    Load ADS-B states with optional Parquet conversion and caching.

    Parameters
    ----------
    file_path : str | Path
        Path to ADS-B CSV or Parquet file
    airport_lat, airport_lon : float, optional
        Airport coordinates; auto-inferred if omitted
    radius_km : float
        Geofence radius in km
    bbox : Tuple[float, float, float, float], optional
        Bounding box (min_lat, max_lat, min_lon, max_lon)
    limit : int, optional
        Limit records loaded
    use_cache : bool
        Enable caching of processed data
    cache_dir : str | Path
        Directory for cache files
    prefer_parquet : bool
        Auto-convert CSV to Parquet on first load

    Returns
    -------
    Tuple[pd.DataFrame, Optional[Tuple[float, float]]]
        (States dataframe, Airport coordinates)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    cache = DataCache(cache_dir) if use_cache else None

    # Try cache first
    if cache is not None:
        cached_df = cache.get(file_path)
        if cached_df is not None:
            print(f"✓ Loaded from cache ({len(cached_df)} records)")
            airport = infer_airport_from_onground(cached_df)
            if airport_lat is not None and airport_lon is not None:
                airport = (airport_lat, airport_lon)
            return cached_df, airport

    # Try Parquet conversion
    parquet_path = file_path.with_suffix(".parquet")
    if prefer_parquet and file_path.suffix == ".csv" and not parquet_path.exists():
        print("Converting CSV to Parquet for faster loading...")
        ParquetConverter.csv_to_parquet(file_path, parquet_path)
        file_path = parquet_path
    elif parquet_path.exists():
        file_path = parquet_path

    # Load data
    if file_path.suffix == ".parquet":
        print(f"Loading Parquet: {file_path}")
        df = pd.read_parquet(file_path)
        df = _coerce_types(df)
    else:
        print(f"Loading CSV: {file_path}")
        chunks = []
        remaining = limit
        for chunk in pd.read_csv(file_path, chunksize=200_000):
            chunk = _coerce_types(chunk)
            if bbox is not None:
                min_lat, max_lat, min_lon, max_lon = bbox
                chunk = chunk[
                    (chunk["lat"].between(min_lat, max_lat))
                    & (chunk["lon"].between(min_lon, max_lon))
                ]
            chunks.append(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    airport = None
    if airport_lat is not None and airport_lon is not None:
        airport = (airport_lat, airport_lon)
    else:
        airport = infer_airport_from_onground(df)

    if airport is not None:
        min_lat, max_lat, min_lon, max_lon = bbox_from_center(airport, radius_km)
        df = df[
            (df["lat"].between(min_lat, max_lat))
            & (df["lon"].between(min_lon, max_lon))
        ]

    df = df.reset_index(drop=True)

    # Cache result
    if cache is not None:
        cache.set(file_path, df)

    return df, airport


def load_states_original(
    file_path: str | Path,
    airport_lat: Optional[float] = None,
    airport_lon: Optional[float] = None,
    radius_km: float = 30.0,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, Optional[Tuple[float, float]]]:
    """Original load_states function (kept for backward compatibility)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    chunks = []
    remaining = limit
    for chunk in pd.read_csv(file_path, chunksize=200_000):
        chunk = _coerce_types(chunk)
        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            chunk = chunk[
                (chunk["lat"].between(min_lat, max_lat))
                & (chunk["lon"].between(min_lon, max_lon))
            ]
        chunks.append(chunk)
        if remaining is not None:
            remaining -= len(chunk)
            if remaining <= 0:
                break
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    airport = None
    if airport_lat is not None and airport_lon is not None:
        airport = (airport_lat, airport_lon)
    else:
        airport = infer_airport_from_onground(df)

    if airport is not None:
        min_lat, max_lat, min_lon, max_lon = bbox_from_center(airport, radius_km)
        df = df[
            (df["lat"].between(min_lat, max_lat))
            & (df["lon"].between(min_lon, max_lon))
        ]

    return df.reset_index(drop=True), airport


def save_metadata(path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
