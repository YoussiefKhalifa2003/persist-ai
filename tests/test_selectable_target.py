from __future__ import annotations

import pytest
import numpy as np

from lumen.data.selectable_target import (
    _appearance_vector,
    _pick_next,
    _remove_visible_frames_from_windows,
    build_candidates,
    build_selectable_target_clip,
)
from lumen.pipelines.persist_occlusion import frame_is_persist_latent, find_all_occlusion_windows
from lumen.types import BBox, Detection


def _person(x: float, frame: int) -> Detection:
    return Detection(BBox(x, 100, x + 35, 180), 0.9, 0)


def _car(x: float, frame: int) -> Detection:
    return Detection(BBox(x, 130, x + 90, 180), 0.9, 2)


def test_candidates_return_people_and_vehicles():
    dets = {
        10: [
            Detection(BBox(10, 10, 30, 35), 0.9, 0),
            Detection(BBox(50, 80, 90, 170), 0.9, 0),
            Detection(BBox(120, 120, 230, 180), 0.9, 2),
        ]
    }
    candidates = build_candidates(dets, 10, {0, 2, 5, 7})
    assert [c["class_name"] for c in candidates] == ["person", "person", "car"]
    assert candidates[0]["tracking_quality"] in {"degraded", "low"}


def test_selected_person_path_tracks_clicked_detection_through_vehicle_occlusion():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(30):
        if 12 <= i < 17:
            all_dets[i] = [Detection(BBox(120, 80, 230, 210), 0.9, 2)]
        else:
            all_dets[i] = [_person(60 + i * 3, i)]

    clip = build_selectable_target_clip(all_dets, 0, 30, 5, 0, 320)
    assert clip.class_id == 0
    assert clip.target_memory is not None
    assert clip.target_memory.target_id == "5:0"
    assert clip.target_memory.state_by_frame[14] in {"OCCLUDED", "PREDICTED"}
    assert clip.target_memory.stable_height >= 70
    assert any(s <= 14 < e for s, e in clip.occlusion_windows)
    assert clip.anchor_path[5] is not None
    assert clip.predicted_paths[14]
    assert clip.confidence_by_frame[14] < 1.0
    assert frame_is_persist_latent(
        14,
        clip.anchor_path[14],
        all_dets[14],
        clip.occlusion_windows,
        target_class_id=0,
    )


def test_predicted_path_projects_forward_motion_during_occlusion():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(36):
        if 15 <= i < 21:
            all_dets[i] = [Detection(BBox(120, 80, 230, 210), 0.9, 2)]
        else:
            all_dets[i] = [_person(50 + i * 4, i)]

    clip = build_selectable_target_clip(all_dets, 0, 36, 6, 0, 360)
    path = clip.predicted_paths[17]
    assert len(path) > 5
    assert path[-1][0] > path[0][0]
    assert 0.2 <= clip.confidence_by_frame[17] < 1.0
    assert clip.target_memory is not None
    assert clip.target_memory.state_by_frame[17] in {"OCCLUDED", "PREDICTED"}


def test_selected_person_uses_appearance_to_avoid_neighbor_switch():
    all_dets: dict[int, list[Detection]] = {}
    frames = []
    for i in range(30):
        frame = np.zeros((220, 320, 3), dtype=np.uint8)
        frame[:] = (80, 80, 80)
        tan = BBox(60 + i * 3, 90, 95 + i * 3, 180)
        dark = BBox(94 + i * 3, 90, 129 + i * 3, 180)
        if 12 <= i < 17:
            all_dets[i] = [Detection(BBox(80, 80, 190, 190), 0.9, 2)]
        else:
            all_dets[i] = [Detection(tan, 0.9, 0), Detection(dark, 0.9, 0)]
            cv2 = pytest.importorskip("cv2")
            cv2.rectangle(frame, (int(tan.x1), int(tan.y1)), (int(tan.x2), int(tan.y2)), (90, 150, 190), -1)
            cv2.rectangle(frame, (int(dark.x1), int(dark.y1)), (int(dark.x2), int(dark.y2)), (35, 35, 45), -1)
        frames.append(frame)

    def frame_provider(abs_i: int):
        return frames[abs_i]

    clip = build_selectable_target_clip(
        all_dets,
        0,
        30,
        5,
        0,
        320,
        frame_provider=frame_provider,
    )
    assert clip.anchor_path[23] is not None
    expected_tan_cx = all_dets[23][0].bbox.cx
    expected_dark_cx = all_dets[23][1].bbox.cx
    assert abs(clip.anchor_path[23].cx - expected_tan_cx) < abs(clip.anchor_path[23].cx - expected_dark_cx)


def test_near_continuous_partial_person_does_not_switch_to_better_looking_neighbor():
    cv2 = pytest.importorskip("cv2")
    all_dets: dict[int, list[Detection]] = {}
    frames = []
    for i in range(34):
        frame = np.zeros((220, 220, 3), dtype=np.uint8)
        frame[:] = (70, 70, 70)
        x = 54 + i * 0.45
        target_full = BBox(x, 82, x + 26, 158)
        neighbor = BBox(x + 21, 82, x + 47, 158)
        if 20 <= i <= 26:
            target = BBox(x + 2, 96, x + 22, 153)
            target_color = (85, 85, 95)
        else:
            target = target_full
            target_color = (35, 35, 45)
        all_dets[i] = [Detection(target, 0.82, 0), Detection(neighbor, 0.80, 0)]
        cv2.rectangle(frame, (int(target.x1), int(target.y1)), (int(target.x2), int(target.y2)), target_color, -1)
        cv2.rectangle(frame, (int(neighbor.x1), int(neighbor.y1)), (int(neighbor.x2), int(neighbor.y2)), (35, 35, 45), -1)
        frames.append(frame)

    clip = build_selectable_target_clip(
        all_dets,
        0,
        34,
        0,
        0,
        220,
        frame_provider=lambda i: frames[i],
    )
    assert clip.anchor_path[24] is not None
    expected_target_cx = all_dets[24][0].bbox.cx
    neighbor_cx = all_dets[24][1].bbox.cx
    assert abs(clip.anchor_path[24].cx - expected_target_cx) < abs(clip.anchor_path[24].cx - neighbor_cx)
    assert clip.target_memory is not None
    assert clip.target_memory.target_id == "0:0"


def test_fully_visible_target_is_accepted_without_fabricated_occlusion_windows():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(24):
        all_dets[i] = [Detection(BBox(50 + i, 80, 78 + i, 156), 0.9, 0)]

    clip = build_selectable_target_clip(all_dets, 0, 24, 0, 0, 220)
    assert clip.occlusion_windows == []
    assert clip.target_memory is not None
    assert clip.target_memory.state_by_frame[18] == "VISIBLE"


def test_same_appearance_ambiguous_neighbors_are_not_confidently_accepted():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((240, 220, 3), dtype=np.uint8)
    frame[:] = (70, 70, 70)
    selected = BBox(50, 80, 90, 190)
    cand_a = BBox(55, 80, 95, 190)
    cand_b = BBox(59, 80, 99, 190)
    for bb in (selected, cand_a, cand_b):
        cv2.rectangle(frame, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), (40, 180, 80), -1)

    template = _appearance_vector(frame, selected, 0)
    picked = _pick_next(
        selected,
        [Detection(cand_a, 0.9, 0), Detection(cand_b, 0.88, 0)],
        0,
        1,
        template,
        root_template=template,
        root_size=(selected.w, selected.h),
        frame=frame,
        velocity=(5.0, 0.0),
    )
    assert picked is None


def test_oversized_foreground_person_is_not_accepted_as_selected_body():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((260, 260, 3), dtype=np.uint8)
    frame[:] = (60, 60, 60)
    selected = BBox(80, 80, 120, 180)
    foreground = BBox(68, 62, 140, 218)
    cv2.rectangle(frame, (int(selected.x1), int(selected.y1)), (int(selected.x2), int(selected.y2)), (40, 180, 80), -1)
    cv2.rectangle(frame, (int(foreground.x1), int(foreground.y1)), (int(foreground.x2), int(foreground.y2)), (40, 180, 80), -1)

    template = _appearance_vector(frame, selected, 0)
    picked = _pick_next(
        selected,
        [Detection(foreground, 0.9, 0)],
        0,
        1,
        template,
        root_template=template,
        root_size=(selected.w, selected.h),
        frame=frame,
        velocity=(0.0, 0.0),
    )
    assert picked is None


def test_partial_person_detection_does_not_shrink_selected_anchor():
    all_dets: dict[int, list[Detection]] = {}
    frames = []
    cv2 = pytest.importorskip("cv2")
    for i in range(24):
        frame = np.zeros((220, 320, 3), dtype=np.uint8)
        full = BBox(70 + i * 3, 90, 110 + i * 3, 190)
        if 10 <= i < 14:
            partial = BBox(full.x1 + 6, full.y1 + 34, full.x2 - 6, full.y1 + 68)
            all_dets[i] = [Detection(partial, 0.75, 0), Detection(BBox(60, 120, 180, 200), 0.9, 2)]
            cv2.rectangle(frame, (int(partial.x1), int(partial.y1)), (int(partial.x2), int(partial.y2)), (90, 150, 190), -1)
        else:
            all_dets[i] = [Detection(full, 0.9, 0)]
            cv2.rectangle(frame, (int(full.x1), int(full.y1)), (int(full.x2), int(full.y2)), (90, 150, 190), -1)
        frames.append(frame)

    clip = build_selectable_target_clip(
        all_dets,
        0,
        24,
        4,
        0,
        320,
        frame_provider=lambda i: frames[i],
    )
    assert clip.anchor_path[12] is not None
    assert clip.anchor_path[12].h >= 90
    assert frame_is_persist_latent(12, clip.anchor_path[12], all_dets[12], clip.occlusion_windows)


def test_partial_pole_window_holds_briefly_then_exits():
    path = {i: BBox(200 + i * 4, 90, 240 + i * 4, 190) for i in range(26)}
    all_dets: dict[int, list[Detection]] = {}
    for i in range(18):
        all_dets[i] = [Detection(path[i], 0.9, 0)]
    for i in range(18, 22):
        bb = path[i]
        all_dets[i] = [Detection(BBox(bb.cx - 9, bb.y1, bb.cx + 9, bb.y2), 0.6, 0)]
    for i in range(22, 26):
        all_dets[i] = []

    windows = find_all_occlusion_windows(path, all_dets, 0, 26, target_class_id=0)
    assert any(s <= 18 < e and e >= 22 for s, e in windows)


def test_selected_vehicle_path_uses_vehicle_visibility_not_person_visibility():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(32):
        if 14 <= i < 19:
            all_dets[i] = [Detection(BBox(125, 110, 250, 195), 0.9, 7)]
        else:
            all_dets[i] = [_car(40 + i * 2, i)]

    clip = build_selectable_target_clip(all_dets, 0, 32, 6, 0, 360)
    assert clip.class_id == 2
    assert any(s <= 16 < e for s, e in clip.occlusion_windows)
    assert frame_is_persist_latent(
        16,
        clip.anchor_path[16],
        all_dets[16],
        clip.occlusion_windows,
        target_class_id=2,
    )


def test_unstable_selected_target_renders_low_quality():
    all_dets = {0: [_person(50, 0)], 1: [], 2: []}
    clip = build_selectable_target_clip(all_dets, 0, 3, 0, 0, 320)
    assert clip.tracking_quality in {"degraded", "low"}
    assert clip.failure_mode is not None
    assert clip.anchor_path[0] is not None


def test_occlusion_windows_never_cover_selected_visible_frames():
    windows = _remove_visible_frames_from_windows([(4, 12), (18, 24)], {6, 7, 8, 20})
    assert windows == [(4, 6), (9, 12), (18, 20), (21, 24)]


def test_low_evidence_long_ghost_target_renders_with_uncertainty():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(80):
        if i < 20:
            all_dets[i] = [_person(50 + i * 2, i)]
        else:
            all_dets[i] = [Detection(BBox(70 + i * 2, 90, 160 + i * 2, 200), 0.9, 2)]

    clip = build_selectable_target_clip(all_dets, 0, 80, 5, 0, 320)
    assert clip.tracking_quality in {"degraded", "low"}
    assert clip.failure_mode is not None
    assert clip.predicted_paths


def test_stationary_target_prediction_collapses_instead_of_drifting():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(24):
        all_dets[i] = [Detection(BBox(80, 90, 118, 190), 0.9, 0)]

    clip = build_selectable_target_clip(all_dets, 0, 24, 4, 0, 320)
    path = clip.predicted_paths[12]
    assert clip.prediction_mode_by_frame is not None
    assert clip.prediction_mode_by_frame[12] == "stationary"
    assert max(abs(x - path[0][0]) for x, _y in path) < 1.0


def test_crowd_pass_by_does_not_swap_to_neighbor():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(24):
        all_dets[i] = [_person(40 + i * 2, i), _person(130 + i * 2, i)]

    clip = build_selectable_target_clip(all_dets, 0, 24, 6, 0, 320)
    start = clip.anchor_path.get(6)
    later = clip.anchor_path.get(18)
    assert start is not None and later is not None
    assert later.cx < 110


def test_stop_then_occlude_keeps_anchor_near_stop_point():
    all_dets: dict[int, list[Detection]] = {}
    for i in range(10):
        all_dets[i] = [_person(100, i)]
    for i in range(10, 18):
        all_dets[i] = [Detection(BBox(120, 80, 250, 210), 0.9, 2)]

    clip = build_selectable_target_clip(all_dets, 0, 18, 4, 0, 320)
    anchor_before_oc = clip.anchor_path.get(9)
    anchor_during_oc = clip.anchor_path.get(12)
    assert anchor_before_oc is not None
    assert anchor_during_oc is not None
    assert abs(anchor_during_oc.cx - anchor_before_oc.cx) < 35
