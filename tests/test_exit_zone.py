import pytest

from lumen.core.exit_zone import compute_exit_zones, point_in_zone
from lumen.types import BBox


def test_exit_zone_with_occluder():
    occluder = BBox(100, 100, 200, 200)
    zones = compute_exit_zones((90, 150), (2.0, 0.0), occluder, margin=20)
    assert len(zones) >= 1
    cx = zones[0][0].cx
    assert cx >= occluder.x2 - 30 or cx <= occluder.x1 + 30


def test_exit_zone_no_occluder():
    zones = compute_exit_zones((100, 100), (1.0, 0.0), None, margin=25)
    assert len(zones) == 1


def test_point_in_zone():
    z = BBox(0, 0, 50, 50)
    assert point_in_zone((25, 25), z)
    assert not point_in_zone((100, 100), z)


def test_moving_right_prefers_right_edge():
    occluder = BBox(100, 100, 200, 200)
    zones = compute_exit_zones((120, 150), (5.0, 0.0), occluder, margin=15)
    assert zones[0][0].cx >= occluder.cx


def test_moving_left_prefers_left_edge():
    occluder = BBox(100, 100, 200, 200)
    zones = compute_exit_zones((180, 150), (-5.0, 0.0), occluder, margin=15)
    assert zones[0][0].cx <= occluder.cx


def test_zero_velocity_fallback():
    occluder = BBox(100, 100, 200, 200)
    zones = compute_exit_zones((150, 150), (0.0, 0.0), occluder, margin=25)
    assert len(zones) >= 1


def test_zone_weight_positive():
    occluder = BBox(50, 50, 150, 150)
    zones = compute_exit_zones((40, 100), (3.0, 0.0), occluder, margin=20)
    for _, w in zones:
        assert w > 0


def test_multiple_zones_returned():
    occluder = BBox(100, 100, 200, 250)
    zones = compute_exit_zones((150, 120), (0.0, 4.0), occluder, margin=20)
    assert len(zones) >= 2
