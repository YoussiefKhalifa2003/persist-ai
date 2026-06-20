"""Global PERSIST-AI occlusion / latent-tracking rules (any clip)."""

from __future__ import annotations

from lumen.data.pedestrian_clip_finder import (
    _person_visible,
    _vehicle_near_anchor,
)
from lumen.pipelines.comparison_pipeline import iou
from lumen.types import BBox, Detection

PERSON = 0

def frame_is_persist_latent(
    frame_idx: int,
    anchor: BBox | None,
    dets: list[Detection],
    occlusion_windows: list[tuple[int, int]],
    visible_thresh: float = 0.14,
) -> bool:
    """True when the locked target is hidden but PERSIST-AI should hold ghost state."""
    if anchor is None:
        return False
    if _person_visible(dets, anchor, thresh=visible_thresh):
        return False
    if _vehicle_near_anchor(dets, anchor):
        return True
    return any(start <= frame_idx < end for start, end in occlusion_windows)


def find_all_occlusion_windows(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    n: int,
    min_len: int = 2,
    merge_gap: int = 4,
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
        if _person_visible(dets, anchor, thresh=0.14):
            flags.append(False)
            continue
        flags.append(_vehicle_near_anchor(dets, anchor))

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

    capped = [(s, min(e, s + 28)) for s, e in raw]
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
) -> int | None:
    last: int | None = None
    for i in range(n):
        bb = path.get(i)
        if bb is None:
            continue
        abs_i = clip_start + i
        people = [d for d in all_dets.get(abs_i, []) if d.class_id == PERSON]
        if any(iou(bb, d.bbox) > match_thresh for d in people):
            last = i
    return last


def finalize_anchor_path(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    frame_w: float,
    occlusion_windows: list[tuple[int, int]],
    post_exit_frames: int = 5,
    exit_gap_frames: int = 3,
    raw_tail_frames: int = 12,
) -> tuple[dict[int, BBox | None], list[tuple[int, int]], int]:
    """Keep anchor through vehicle occlusions; end clip after last window + raw tail."""
    n = max(path.keys()) + 1 if path else 0
    out = dict(path)
    windows = [(s, e) for s, e in occlusion_windows if e > s]
    last_match = _last_anchor_match_frame(out, all_dets, clip_start, n)

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

    # Clear anchor in gaps between occlusion windows and after final exit.
    occupied = [False] * n
    for s, e in windows:
        for j in range(s, min(e, n)):
            occupied[j] = True

    for j in range(n):
        if occupied[j]:
            continue
        if last_match is not None and j > last_match:
            out[j] = None
        elif out.get(j) is not None and not _person_visible(
            all_dets.get(clip_start + j, []), out[j], 0.14  # type: ignore[arg-type]
        ):
            # Drop stale extrapolated boxes outside occlusion (e.g. subject already left).
            out[j] = None

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
