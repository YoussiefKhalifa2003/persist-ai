"""Global PERSIST-AI occlusion / latent-tracking rules (any clip)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from lumen.data.pedestrian_clip_finder import (
    _person_visible,
    _vehicle_near_anchor,
    _x_overlap,
)
from lumen.pipelines.comparison_pipeline import iou
from lumen.types import BBox, Detection

PERSON = 0
_DEBUG_LOG = Path("debug-6ac46f.log")


def _dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "6ac46f",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": data.get("runId", "pre-fix"),
    }
    with _DEBUG_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    # #endregion


def _target_visible(all_dets: list[Detection], anchor: BBox, target_class_id: int, thresh: float) -> bool:
    for d in all_dets:
        if d.class_id != target_class_id or iou(anchor, d.bbox) <= thresh:
            continue
        overlap = iou(anchor, d.bbox)
        # A narrow/short sliver overlapping the anchor is evidence of partial
        # occlusion, not proof that the full target is visible.
        if target_class_id == PERSON and (
            d.bbox.w < anchor.w * 0.80
            or d.bbox.h < anchor.h * 0.80
            or d.bbox.area < anchor.area * 0.72
            or d.bbox.w > anchor.w * 1.45
            or d.bbox.h > anchor.h * 1.35
            or d.bbox.area > anchor.area * 1.65
        ):
            merged_side_by_side = (
                d.bbox.w <= anchor.w * 1.85
                and anchor.h * 0.86 <= d.bbox.h <= anchor.h * 1.16
                and overlap >= 0.48
            )
            if target_class_id == PERSON and merged_side_by_side:
                return True
            continue
        return True
    return False


def target_visible_enough(
    all_dets: list[Detection],
    anchor: BBox,
    target_class_id: int,
    thresh: float = 0.14,
) -> bool:
    return _target_visible(all_dets, anchor, target_class_id, thresh)


def _partial_target_evidence(
    all_dets: list[Detection],
    anchor: BBox,
    target_class_id: int,
) -> bool:
    """A clipped target-like detection near the anchor means occlusion, not exit."""
    for d in all_dets:
        if d.class_id != target_class_id:
            continue
        if abs(d.bbox.cy - anchor.cy) > max(28.0, anchor.h * 0.45):
            continue
        if _x_overlap(anchor, d.bbox) < 0.10 and abs(d.bbox.cx - anchor.cx) > max(45.0, anchor.w * 1.5):
            continue
        clipped_width = d.bbox.w < anchor.w * 0.78
        clipped_height = d.bbox.h < anchor.h * 0.78
        clipped_area = d.bbox.area < anchor.area * 0.70
        if clipped_width or clipped_height or clipped_area:
            return True
    return False


def _same_class_occluder_near_anchor(
    all_dets: list[Detection],
    anchor: BBox,
    target_class_id: int,
) -> bool:
    """A larger overlapping same-class object can hide the selected target in crowds."""
    if target_class_id != PERSON:
        return False
    for d in all_dets:
        if d.class_id != target_class_id:
            continue
        overlap = iou(anchor, d.bbox)
        same_ground_band = abs(d.bbox.cy - anchor.cy) <= max(40.0, anchor.h * 0.42)
        adjacent_body = (
            same_ground_band
            and abs(d.bbox.cx - anchor.cx) <= max(58.0, anchor.w * 2.25)
            and anchor.h * 0.74 <= d.bbox.h <= anchor.h * 1.28
            and d.bbox.area >= anchor.area * 0.50
        )
        if adjacent_body:
            return True
        if overlap < 0.08:
            continue
        # If the detection is a good full-body match to the anchor, it is probably
        # the target, not an occluder. This function is only called after the
        # stricter target-visible check, but keep the distinction explicit.
        if (
            d.bbox.w >= anchor.w * 0.80
            and d.bbox.h >= anchor.h * 0.80
            and d.bbox.area >= anchor.area * 0.72
            and d.bbox.w <= anchor.w * 1.45
            and d.bbox.h <= anchor.h * 1.35
            and d.bbox.area <= anchor.area * 1.65
            and overlap >= 0.24
        ):
            continue
        covers_body = _x_overlap(anchor, d.bbox) >= 0.22 or overlap >= 0.16
        foreground_sized = d.bbox.area >= anchor.area * 0.55 or d.bbox.h >= anchor.h * 0.72
        if same_ground_band and covers_body and foreground_sized:
            return True
    return False

def frame_is_persist_latent(
    frame_idx: int,
    anchor: BBox | None,
    dets: list[Detection],
    occlusion_windows: list[tuple[int, int]],
    visible_thresh: float = 0.14,
    target_class_id: int = PERSON,
) -> bool:
    """True when the locked target is hidden but PERSIST-AI should hold ghost state."""
    if anchor is None:
        return False
    if _target_visible(dets, anchor, target_class_id, thresh=visible_thresh):
        return False
    if _vehicle_near_anchor(dets, anchor):
        return True
    if _partial_target_evidence(dets, anchor, target_class_id):
        return True
    if _same_class_occluder_near_anchor(dets, anchor, target_class_id):
        return True
    return any(start <= frame_idx < end for start, end in occlusion_windows)


def find_all_occlusion_windows(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    n: int,
    min_len: int = 2,
    merge_gap: int = 4,
    target_class_id: int = PERSON,
    tail_frames: int = 4,
) -> list[tuple[int, int]]:
    """Every interval where the anchor is hidden (vehicle block or YOLO dropout)."""
    flags: list[bool] = []
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            flags.append(False)
            continue
        abs_i = clip_start + i
        dets = all_dets.get(abs_i, [])
        if _target_visible(dets, anchor, target_class_id, thresh=0.14):
            flags.append(False)
            continue
        flags.append(
            _vehicle_near_anchor(dets, anchor)
            or _partial_target_evidence(dets, anchor, target_class_id)
            or _same_class_occluder_near_anchor(dets, anchor, target_class_id)
        )

    raw: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < n and flags[i]:
            i += 1
        if i - start >= min_len:
            raw.append((start, i))

    if not raw:
        return []

    extended: list[tuple[int, int]] = []
    for s, e in raw:
        end = e
        for j in range(e, min(n, e + tail_frames)):
            anchor = path.get(j)
            if anchor is None:
                break
            dets = all_dets.get(clip_start + j, [])
            if _target_visible(dets, anchor, target_class_id, thresh=0.14):
                break
            end = j + 1
        extended.append((s, end))

    capped = [(s, min(e, s + (45 if target_class_id == PERSON else 32))) for s, e in extended]
    merged: list[tuple[int, int]] = [capped[0]]
    for start, end in capped[1:]:
        ps, pe = merged[-1]
        if start - pe <= merge_gap:
            merged[-1] = (ps, end)
        else:
            merged.append((start, end))
    return merged


def _last_anchor_match_frame(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    n: int,
    match_thresh: float = 0.14,
    target_class_id: int = PERSON,
) -> int | None:
    last: int | None = None
    for i in range(n):
        bb = path.get(i)
        if bb is None:
            continue
        abs_i = clip_start + i
        if _target_visible(all_dets.get(abs_i, []), bb, target_class_id, match_thresh):
            last = i
    return last


def _last_target_evidence_frame(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    start: int,
    end: int,
    target_class_id: int,
) -> int | None:
    last: int | None = None
    for i in range(start, end):
        bb = path.get(i)
        if bb is None:
            continue
        dets = all_dets.get(clip_start + i, [])
        if _target_visible(dets, bb, target_class_id, 0.14) or _partial_target_evidence(
            dets, bb, target_class_id
        ):
            last = i
    return last


def finalize_anchor_path(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    frame_w: float,
    occlusion_windows: list[tuple[int, int]],
    post_exit_frames: int = 5,
    edge_exit_hold_frames: int = 2,
    exit_gap_frames: int = 3,
    raw_tail_frames: int = 12,
    target_class_id: int = PERSON,
) -> tuple[dict[int, BBox | None], list[tuple[int, int]], int]:
    """Keep anchor through vehicle occlusions; end clip after last window + raw tail."""
    n = max(path.keys()) + 1 if path else 0
    out = dict(path)
    windows = [(s, e) for s, e in occlusion_windows if e > s]
    last_match = _last_anchor_match_frame(out, all_dets, clip_start, n, target_class_id=target_class_id)

    if not windows:
        clip_len = min(n, (last_match or 0) + post_exit_frames)
        trimmed = {i: out.get(i) for i in range(clip_len)}
        return trimmed, windows, clip_len

    # Drop windows that begin after the subject exited (no YOLO match for several frames).
    kept: list[tuple[int, int]] = []
    for s, e in windows:
        no_match_gap = (s - last_match - 1) if last_match is not None else 0
        if last_match is not None and s > last_match and no_match_gap >= exit_gap_frames:
            for j in range(s, min(e, n)):
                out[j] = None
            continue
        kept.append((s, e))
    windows = kept

    capped_windows: list[tuple[int, int]] = []
    for s, e in windows:
        first_edge_exit: int | None = None
        for j in range(s, min(e, n)):
            bbj = out.get(j)
            if bbj is not None and (bbj.x2 >= frame_w * 0.98 or bbj.cx >= frame_w * 0.94):
                first_edge_exit = j
                break
        last_ev = _last_target_evidence_frame(out, all_dets, clip_start, s, min(e, n), target_class_id)
        if first_edge_exit is not None:
            e = min(e, first_edge_exit + edge_exit_hold_frames)
        if last_ev is not None:
            bb = out.get(last_ev)
            if bb is not None and bb.cx > frame_w * 0.78:
                e = min(e, last_ev + post_exit_frames)
        if e > s:
            capped_windows.append((s, e))
    windows = capped_windows

    # Clear anchor in gaps between occlusion windows and after final exit.
    occupied = [False] * n
    for s, e in windows:
        for j in range(s, min(e, n)):
            occupied[j] = True

    for j in range(n):
        if occupied[j]:
            continue
        if last_match is not None and j > last_match + post_exit_frames + 6:
            out[j] = None
        elif out.get(j) is not None and not _target_visible(
            all_dets.get(clip_start + j, []), out[j], target_class_id, 0.14  # type: ignore[arg-type]
        ):
            if last_match is not None and j <= last_match + post_exit_frames:
                continue
            # Drop stale extrapolated boxes outside occlusion (e.g. subject already left).
            out[j] = None

    cleared_after_match = sum(
        1 for j in range(n) if path.get(j) is not None and out.get(j) is None and (last_match is None or j <= last_match)
    )
    _dbg(
        "H1",
        "persist_occlusion.py:finalize_anchor_path",
        "finalize_cleared_anchors",
        {
            "last_match": last_match,
            "cleared_after_match": cleared_after_match,
            "window_count": len(windows),
            "clip_len_before": n,
        },
    )

    last_oc_end = max((e for _, e in windows), default=0)
    raw_tail = (last_match or 0) + post_exit_frames + raw_tail_frames
    clip_len = min(n, max(last_oc_end + post_exit_frames, raw_tail))
    windows = [(s, min(e, clip_len)) for s, e in windows if s < clip_len]
    trimmed = {i: out.get(i) for i in range(clip_len)}
    return trimmed, windows, clip_len


def mask_subject_windows(
    dets_map: dict[int, list[Detection]],
    path: dict[int, BBox | None],
    occlusion_windows: list[tuple[int, int]],
    thresh: float = 0.22,
) -> dict[int, list[Detection]]:
    """Hide YOLO person detections on the baseline during every occlusion window."""
    from lumen.pipelines.comparison_pipeline import mask_anchor_detections

    out: dict[int, list[Detection]] = {}
    for i, dets in dets_map.items():
        if any(s <= i < e for s, e in occlusion_windows):
            out[i] = mask_anchor_detections(dets, path.get(i), thresh)
        else:
            out[i] = dets
    return out
