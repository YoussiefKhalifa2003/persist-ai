from __future__ import annotations

from lumen.types import BBox


def _edge_zones(
    occluder: BBox, margin: float, weight: float
) -> list[tuple[BBox, float]]:
    w = occluder.w
    h = occluder.h
    left = BBox(
        occluder.x1 - margin,
        occluder.y1,
        occluder.x1 + margin,
        occluder.y2,
    )
    right = BBox(
        occluder.x2 - margin,
        occluder.y1,
        occluder.x2 + margin,
        occluder.y2,
    )
    bottom = BBox(
        occluder.x1,
        occluder.y2 - margin,
        occluder.x2,
        occluder.y2 + margin,
    )
    return [(left, weight * 0.8), (right, weight * 0.8), (bottom, weight * 0.6)]


def compute_exit_zones(
    last_center: tuple[float, float],
    velocity: tuple[float, float],
    occluder: BBox | None,
    margin: float = 25.0,
) -> list[tuple[BBox, float]]:
    """Return candidate re-entry zones on occluder edges."""
    vx, vy = velocity
    speed = (vx**2 + vy**2) ** 0.5

    if occluder is None or speed < 0.5:
        cx, cy = last_center
        fallback = BBox(cx - margin * 2, cy - margin, cx + margin * 2, cy + margin)
        return [(fallback, 0.5)]

    zones = _edge_zones(occluder, margin, 1.0)

    if vx > 0.5:
        zones.sort(key=lambda z: 0 if z[0].cx > occluder.cx else 1)
    elif vx < -0.5:
        zones.sort(key=lambda z: 0 if z[0].cx < occluder.cx else 1)
    if vy > 0.5:
        zones = sorted(zones, key=lambda z: 0 if z[0].cy > occluder.cy else 1)

    return zones[:3]


def point_in_zone(point: tuple[float, float], zone: BBox) -> bool:
    x, y = point
    return zone.x1 <= x <= zone.x2 and zone.y1 <= y <= zone.y2
