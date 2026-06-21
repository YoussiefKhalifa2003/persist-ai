from __future__ import annotations

import numpy as np

from lumen.types import BBox


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def center_inside(inner: BBox, outer: BBox, margin: float = 0.0) -> bool:
    expanded = BBox(
        outer.x1 - margin,
        outer.y1 - margin,
        outer.x2 + margin,
        outer.y2 + margin,
    )
    return (
        expanded.x1 <= inner.cx <= expanded.x2
        and expanded.y1 <= inner.cy <= expanded.y2
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bbox_center(b: BBox) -> tuple[float, float]:
    return b.cx, b.cy


def l2_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
