"""Tests for global PERSIST-AI occlusion rules."""

from __future__ import annotations

from lumen.pipelines.persist_occlusion import (
    finalize_anchor_path,
    find_all_occlusion_windows,
    frame_is_persist_latent,
    target_visible_enough,
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


def test_same_class_pedestrian_occluder_triggers_latent_window():
    path = {i: BBox(100 + i * 2, 100, 140 + i * 2, 200) for i in range(24)}
    all_dets: dict[int, list[Detection]] = {}
    for i in range(10):
        all_dets[i] = [Detection(path[i], 0.9, 0)]
    for i in range(10, 15):
        bb = path[i]
        # Nearby foreground pedestrian covers the target's body; the target's own
        # full-body box is absent.
        all_dets[i] = [Detection(BBox(bb.x1 - 8, 92, bb.x2 + 18, 210), 0.85, 0)]
    for i in range(15, 24):
        all_dets[i] = [Detection(path[i], 0.9, 0)]

    windows = find_all_occlusion_windows(path, all_dets, 0, 24, target_class_id=0)
    assert any(s <= 12 < e for s, e in windows)
    assert frame_is_persist_latent(12, path[12], all_dets[12], windows, target_class_id=0)


def test_adjacent_pedestrian_dropout_can_hold_latent_briefly():
    path = {i: BBox(220 + i * 3, 100, 260 + i * 3, 200) for i in range(20)}
    all_dets: dict[int, list[Detection]] = {}
    for i in range(10):
        all_dets[i] = [Detection(path[i], 0.9, 0)]
    for i in range(10, 14):
        bb = path[i]
        all_dets[i] = [Detection(BBox(bb.x2 + 8, 101, bb.x2 + 44, 201), 0.8, 0)]
    for i in range(14, 20):
        all_dets[i] = []

    windows = find_all_occlusion_windows(path, all_dets, 0, 20, target_class_id=0)
    assert any(s <= 11 < e for s, e in windows)


def test_visible_person_not_misclassified_as_same_class_occlusion():
    anchor = BBox(100, 100, 140, 200)
    dets = [Detection(BBox(101, 101, 141, 201), 0.9, 0)]
    assert frame_is_persist_latent(5, anchor, dets, [(5, 8)], target_class_id=0) is False


def test_merged_side_by_side_person_box_counts_as_visible_not_ghost():
    anchor = BBox(100, 100, 140, 200)
    merged = Detection(BBox(92, 101, 164, 201), 0.9, 0)
    assert target_visible_enough([merged], anchor, 0)
    assert frame_is_persist_latent(5, anchor, [merged], [(5, 8)], target_class_id=0) is False


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


def test_finalize_caps_edge_exit_window():
    path = {i: BBox(480 + i * 4, 100, 520 + i * 4, 200) for i in range(40)}
    all_dets: dict[int, list[Detection]] = {i: [] for i in range(40)}
    for i in range(15):
        all_dets[i] = [Detection(path[i], 0.9, 0)]
    windows = [(15, 40)]
    out, wins, clip_len = finalize_anchor_path(
        path,
        all_dets,
        0,
        640.0,
        windows,
        post_exit_frames=4,
        target_class_id=0,
    )
    assert wins[0][1] <= 35
    assert out.get(36) is None
    assert clip_len <= 39


def test_finalize_uses_short_hold_after_edge_exit():
    path = {i: BBox(470 + i * 5, 100, 510 + i * 5, 200) for i in range(24)}
    all_dets: dict[int, list[Detection]] = {i: [] for i in range(24)}
    for i in range(10):
        all_dets[i] = [Detection(path[i], 0.9, 0)]

    out, wins, _ = finalize_anchor_path(
        path,
        all_dets,
        0,
        544.0,
        [(10, 24)],
        post_exit_frames=5,
        edge_exit_hold_frames=2,
        target_class_id=0,
    )
    assert wins[0][1] <= wins[0][0] + 4
    assert out.get(wins[0][1] + 2) is None
