"""Tests for global PERSIST-AI occlusion rules."""

from __future__ import annotations

from lumen.pipelines.persist_occlusion import (
    finalize_anchor_path,
    find_all_occlusion_windows,
    frame_is_persist_latent,
)
from lumen.types import BBox, Detection


def test_van_window_triggers_latent():
    path = {i: BBox(320, 160, 365, 255) for i in range(50)}
    car = Detection(BBox(280, 200, 420, 320), 0.9, 2)
    dets = [car]
    windows = [(37, 44)]
    assert frame_is_persist_latent(41, path[41], dets, windows) is True


def test_no_latent_after_subject_exit_gap():
    path = {40: BBox(320, 160, 365, 255)}
    windows = [(37, 44), (85, 107)]
    assert frame_is_persist_latent(50, path[40], [], windows) is False


def test_find_van_and_bus_windows():
    path = {i: BBox(300 + i, 160, 345 + i, 255) for i in range(120)}
    all_dets: dict[int, list[Detection]] = {}
    for i in range(37, 44):
        all_dets[i] = [
            Detection(BBox(280, 200, 420, 320), 0.9, 2),
        ]
    for i in range(85, 108):
        all_dets[i] = [Detection(BBox(10, 80, 540, 280), 0.9, 5)]
    windows = find_all_occlusion_windows(path, all_dets, 0, 120)
    assert any(s <= 41 < e for s, e in windows)
    assert any(s <= 90 < e for s, e in windows)


def test_finalize_ends_after_last_window():
    path = {i: BBox(400, 150, 445, 245) for i in range(76)}
    all_dets: dict[int, list[Detection]] = {i: [] for i in range(120)}
    for i in range(75):
        all_dets[i] = [Detection(BBox(395, 150, 440, 245), 0.9, 0)]
    windows = [(37, 44), (85, 107)]
    out, wins, clip_len = finalize_anchor_path(path, all_dets, 0, 544.0, windows, post_exit_frames=8)
    assert not any(s >= 85 for s, _ in wins)
    assert out.get(85) is None
    assert clip_len <= 95
