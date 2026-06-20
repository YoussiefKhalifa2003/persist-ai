"""Tests for pedestrian anchor termination and compositor layout."""

from __future__ import annotations

import numpy as np

from lumen.data.pedestrian_clip_finder import (
    build_leftmost_woman_path,
    snap_ghost_off_vehicles,
    terminate_anchor_path,
)
from lumen.pipelines.persist_occlusion import finalize_anchor_path
from lumen.types import BBox, Detection
from lumen.viz.crowd_compositor import compose_crowd_frame
from lumen.viz.real_compositor import RealBeat


def test_terminate_clears_anchor_after_occlusion_dropout():
    path = {i: BBox(400, 100, 440, 200) for i in range(80)}
    all_dets = {100 + i: [] for i in range(60, 80)}
    out = terminate_anchor_path(path, all_dets, clip_start=55, oc_end=60, frame_w=544.0)
    assert out[65] is None
    assert out[70] is None
    assert out[59] is not None


def test_terminate_hard_cut_after_oc_end():
    path = {i: BBox(300, 100, 340, 200) for i in range(100)}
    all_dets: dict[int, list[Detection]] = {}
    out = terminate_anchor_path(path, all_dets, clip_start=0, oc_end=50, frame_w=544.0, max_after_oc=10)
    assert out[49] is not None
    assert out[50] is None
    assert out[99] is None


def test_compositor_lumen_on_left_canvas_half():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    dets = [Detection(BBox(10, 10, 40, 80), 0.9, 0)]
    anchor = BBox(50, 20, 90, 100)
    canvas = compose_crowd_frame(
        frame,
        RealBeat.VISIBLE,
        crowd_dets=dets,
        raw_dets=dets,
        target_anchor=anchor,
        lumen_target_bb=anchor,
        lumen_ghost=False,
        layout="split",
    )
    h, w = canvas.shape[:2]
    mid = (w - 8) // 2
    left = canvas[50:100, 10:mid]
    right = canvas[50:100, mid + 8 : w - 10]
    assert left.mean() > right.mean() * 0.5
    assert canvas.shape[1] == 200 * 2 + 8


def test_snap_ghost_keeps_sidewalk_cy():
    path = {i: BBox(460, 150, 505, 245) for i in range(100)}
    bus = Detection(BBox(370, 80, 544, 280), 0.9, 5)
    all_dets = {55 + i: [bus] for i in range(90, 105)}
    out = snap_ghost_off_vehicles(path, all_dets, 55, [(90, 105)], 544.0)
    assert out[95] is not None
    assert 190 <= out[95].cy <= 210
    assert 430 <= out[95].cx <= 470


def test_leftmost_path_uses_sidewalk_cy_not_van_height():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(40):
        all_dets[i] = [Detection(BBox(200 + i * 3, 160, 240 + i * 3, 210), 0.9, 0)]
    for i in range(40, 45):
        all_dets[i] = [Detection(BBox(320, 165, 350, 215), 0.9, 0)]
    path = build_leftmost_woman_path(all_dets, 0, 50, extrapolate_until=50)
    assert path[44] is not None
    assert path[44].cy >= 185


def test_finalize_clears_post_exit():
    # Extrapolated boxes continue after last YOLO match at frame 74.
    path = {i: BBox(460, 150, 505, 245) for i in range(85)}
    all_dets: dict[int, list[Detection]] = {}
    for i in range(75):
        all_dets[i] = [Detection(BBox(455, 150, 500, 245), 0.9, 0)]
    out, wins, clip_len = finalize_anchor_path(
        path, all_dets, 0, 544.0, [(37, 44), (85, 107)], post_exit_frames=8
    )
    assert out.get(85) is None
    assert out.get(95) is None
    assert not any(s >= 85 for s, _ in wins)
    assert clip_len <= 95


def test_finalize_keeps_van_window_when_subject_still_present():
    path = {i: BBox(300 + i, 160, 345 + i, 255) for i in range(50)}
    all_dets: dict[int, list[Detection]] = {
        i: [Detection(BBox(295 + i, 160, 340 + i, 255), 0.9, 0)] for i in range(45)
    }
    out, wins, _ = finalize_anchor_path(path, all_dets, 0, 544.0, [(37, 44)], post_exit_frames=5)
    assert any(s <= 40 < e for s, e in wins)
    assert out.get(40) is not None
