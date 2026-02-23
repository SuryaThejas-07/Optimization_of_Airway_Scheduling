from __future__ import annotations

import math
from typing import Iterable, Tuple

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def within_radius(
    lat: float, lon: float, center: Tuple[float, float], radius_km: float
) -> bool:
    return haversine_km(lat, lon, center[0], center[1]) <= radius_km


def bbox_from_center(
    center: Tuple[float, float], radius_km: float
) -> Tuple[float, float, float, float]:
    lat, lon = center
    delta_lat = radius_km / 110.574
    delta_lon = radius_km / (111.320 * math.cos(math.radians(lat)) + 1e-6)
    return (lat - delta_lat, lat + delta_lat, lon - delta_lon, lon + delta_lon)


def filter_points(
    points: Iterable[Tuple[float, float]],
    center: Tuple[float, float],
    radius_km: float,
) -> list[bool]:
    return [within_radius(lat, lon, center, radius_km) for lat, lon in points]
