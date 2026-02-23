from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


DEFAULT_SEPARATION = {
    "H": {"H": 90, "M": 120, "L": 150},
    "M": {"H": 60, "M": 90, "L": 120},
    "L": {"H": 60, "M": 60, "L": 60},
}


def build_separation_matrix(classes: List[str]) -> list[list[int]]:
    matrix = []
    for i in classes:
        row = []
        for j in classes:
            row.append(DEFAULT_SEPARATION[i][j])
        matrix.append(row)
    return matrix


def save_separation(path: str | Path, classes: List[str]) -> None:
    matrix = build_separation_matrix(classes)
    payload = {
        "classes": classes,
        "matrix": matrix,
        "units": "seconds",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
