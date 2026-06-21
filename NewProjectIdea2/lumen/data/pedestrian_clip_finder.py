"""Find sidewalk clip: one pedestrian occluded by bus or foreground."""

from __future__ import annotations

from dataclasses import dataclass

from lumen.pipelines.comparison_pipeline import fill_gaps, find_occlusion_by_gt_dropout, iou
from lumen.types import BBox, Detection

PERSON = 0
OCCLUDER_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
BUS_CLASSES = OCCLUDER_CLASSES


@dataclass
class PedestrianClip:
    track_id: int
    start: int
    end: int
    oc_start: int
    oc_end: int
    anchor_path: dict[int, BBox | None]
    score: float
    occlusion_windows: list[tuple[int, int]] | None = None


def _track_people(dets_map: dict[int, list[Detection]]) -> dict[int, list[tuple[int, BBox]]]:
    tracks: dict[int, list[tuple[int, BBox]]] = {}
    active: dict[int, BBox] = {}
    next_id = 1
    for fidx in sorted(dets_map.keys()):
        dets = [d for d in dets_map[fidx] if d.class_id == PERSON]
        matched: dict[int, BBox] = {}
        used: set[int] = set()
        for tid, prev in list(active.items()):
            best, best_iou, best_j = None, 0.0, -1
            for j, det in enumerate(dets):
                if j in used:
                    continue
                s = iou(prev, det.bbox)
                if s > best_iou:
                    best_iou, best, best_j = s, det.bbox, j
            if best is not None and best_iou > 0.10:
                matched[tid] = best
                used.add(best_j)
        for j, det in enumerate(dets):
            if j in used:
                continue
            matched[next_id] = det.bbox
            next_id += 1
        active = matched
        tracks[fidx] = [(tid, bb) for tid, bb in active.items()]
    return tracks


def build_path(track_map: dict[int, list[tuple[int, BBox]]], tid: int, n: int) -> dict[int, BBox | None]:
    path = {i: next((bb for t, bb in track_map.get(i, []) if t == tid), None) for i in range(n)}
    return fill_gaps(path)


def extrapolate_path(
    path: dict[int, BBox | None],
    n: int,
    max_extra: int = 55,
    vx_override: float | None = None,
    cx_cap: float | None = None,
    extrapolate_until: int | None = None,
    cy_override: float | None = None,
) -> dict[int, BBox | None]:
    """Extend last real detection with stable velocity and median box size."""
    filled = fill_gaps(path)
    real = [i for i in range(n) if path.get(i) is not None]
    if not real:
        return filled
    sizes = [(path[i].w, path[i].h) for i in real if path[i] is not None and path[i].h >= 55]
    mw = sorted(s[0] for s in sizes)[len(sizes) // 2] if sizes else 45.0
    mh = sorted(s[1] for s in sizes)[len(sizes) // 2] if sizes else 95.0
    last_i = real[-1]
    last_bb = filled[last_i]
    assert last_bb is not None
    tail = real[-min(8, len(real)) :]
    vxs: list[float] = []
    for j in range(1, len(tail)):
        a, b = filled[tail[j - 1]], filled[tail[j]]
        if a and b:
            vxs.append((b.cx - a.cx) / max(1, tail[j] - tail[j - 1]))
    if vx_override is not None:
        vx = vx_override
    else:
        vx = sum(vxs) / len(vxs) if vxs else 2.5
    vx = max(0.8, min(vx, 4.0))
    cy = cy_override if cy_override is not None else last_bb.cy
    end_i = min(n, extrapolate_until if extrapolate_until is not None else last_i + 1 + max_extra)
    for j in range(last_i + 1, end_i):
        if filled.get(j) is not None:
            continue
        t = j - last_i
        cx = last_bb.cx + vx * t
        if cx_cap is not None:
            cx = min(cx, cx_cap)
        filled[j] = BBox(cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2)
    return filled


def _median_sidewalk_cy(path: dict[int, BBox | None], n: int) -> float | None:
    good = [path[i].cy for i in range(n) if path.get(i) is not None and path[i].h >= 65]
    if not good:
        good = [path[i].cy for i in range(n) if path.get(i) is not None]
    if not good:
        return None
    good.sort()
    return good[len(good) // 2]


def snap_ghost_off_vehicles(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    occlusion_windows: list[tuple[int, int]],
    frame_w: float,
) -> dict[int, BBox | None]:
    """Normalize ghost boxes to sidewalk height; advance cx with last-known velocity."""
    if not occlusion_windows:
        return path
    out = dict(path)
    n = max(out.keys()) + 1 if out else 0
    sidewalk_cy = _median_sidewalk_cy(out, n) or 200.0
    ref_sizes = [(out[i].w, out[i].h) for i in range(n) if out.get(i) is not None and out[i].h >= 55]
    mw = sorted(s[0] for s in ref_sizes)[len(ref_sizes) // 2] if ref_sizes else 45.0
    mh = sorted(s[1] for s in ref_sizes)[len(ref_sizes) // 2] if ref_sizes else 95.0

    for oc_start, oc_end in occlusion_windows:
        pre_idx = [i for i in range(max(0, oc_start - 12), oc_start) if out.get(i) is not None]
        if not pre_idx:
            continue
        start_bb = out[pre_idx[-1]]
        vx = 2.5
        if len(pre_idx) >= 2:
            a, b = pre_idx[-2], pre_idx[-1]
            vx = max(0.8, min((out[b].cx - out[a].cx) / max(1, b - a), 4.0))
        for i in range(oc_start, min(oc_end, n)):
            abs_i = clip_start + i
            dets = all_dets.get(abs_i, [])
            if out.get(i) is not None and _person_visible(dets, out[i], thresh=0.14):  # type: ignore[arg-type]
                continue
            t = i - oc_start + 1
            cx = min(start_bb.cx + vx * t, frame_w * 0.86)
            out[i] = BBox(cx - mw / 2, sidewalk_cy - mh / 2, cx + mw / 2, sidewalk_cy + mh / 2)
    return out


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


def trim_anchor_on_subject_exit(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    oc_start: int,
    oc_end: int,
    frame_w: float,
    ghost_hold_frames: int = 22,
    post_ghost_tail: int = 6,
) -> tuple[dict[int, BBox | None], int]:
    """Stop tracking when the woman leaves; keep a short bus-occlusion ghost only."""
    n = max(path.keys()) + 1 if path else 0
    out = dict(path)
    last_match = _last_anchor_match_frame(out, all_dets, clip_start, n)
    if last_match is None:
        return out, n

    # Drop extrapolated 'visible' boxes after the subject is last seen (pre-bus gap).
    for j in range(last_match + 2, oc_start):
        out[j] = None

    ghost_end = min(oc_end, oc_start + ghost_hold_frames)
    for j in range(ghost_end, n):
        out[j] = None

    clip_end = min(n, ghost_end + post_ghost_tail)
    trimmed = {i: out.get(i) for i in range(clip_end)}
    return trimmed, clip_end


def terminate_anchor_path(
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    oc_end: int,
    frame_w: float,
    max_after_oc: int = 15,
    no_det_clear: int = 5,
) -> dict[int, BBox | None]:
    """Clear ghost anchor after subject leaves frame or post-occlusion dropout."""
    if not path:
        return path
    n = max(path.keys()) + 1
    out = dict(path)
    hard_cut = min(n, oc_end + max_after_oc)
    for i in range(hard_cut, n):
        out[i] = None

    miss = 0
    for i in range(oc_end, hard_cut):
        bb = out.get(i)
        abs_i = clip_start + i
        people = [d for d in all_dets.get(abs_i, []) if d.class_id == PERSON]
        if bb is not None and bb.cx > frame_w * 0.88:
            for j in range(i, hard_cut):
                out[j] = None
            break
        if people and bb is not None:
            if any(iou(bb, d.bbox) > 0.18 for d in people):
                miss = 0
                continue
        if not people:
            miss += 1
        else:
            miss += 1
        if i >= oc_end and (miss >= no_det_clear or not people):
            for j in range(i, hard_cut):
                out[j] = None
            break
    return out


def _largest_cx_cluster(dets: list[Detection], gap: float = 90.0) -> list[Detection]:
    """Keep the main pedestrian group; drop isolated false detections far from the cluster."""
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: d.bbox.cx)
    clusters: list[list[Detection]] = [[ordered[0]]]
    for det in ordered[1:]:
        if det.bbox.cx - clusters[-1][-1].bbox.cx > gap:
            clusters.append([det])
        else:
            clusters[-1].append(det)
    return max(clusters, key=len)


def build_leftmost_woman_path(
    all_dets: dict[int, list[Detection]],
    start: int,
    end: int,
    extrapolate_until: int | None = None,
    frame_w: float = 544.0,
) -> dict[int, BBox | None]:
    """Lock onto the tan-coat woman via cluster + nearest-neighbor (no ID switching)."""
    n = end - start
    path: dict[int, BBox | None] = {}
    prev_bb: BBox | None = None
    vx = 2.8

    for i in range(n):
        abs_i = start + i
        people = [d for d in all_dets.get(abs_i, []) if d.class_id == PERSON and d.bbox.h > 45]
        if not people:
            path[i] = None
            continue
        cy_med = sorted(d.bbox.cy for d in people)[len(people) // 2]
        on_sidewalk = [d for d in people if abs(d.bbox.cy - cy_med) < 55]
        cluster = _largest_cx_cluster(on_sidewalk)
        if not cluster:
            path[i] = None
            continue

        if prev_bb is None:
            pick = min(cluster, key=lambda d: d.bbox.cx)
        else:
            pred_cx = prev_bb.cx + vx
            pick = min(cluster, key=lambda d: abs(d.bbox.cx - pred_cx))
            step = pick.bbox.cx - prev_bb.cx
            prev_idx = max((j for j in range(i) if path.get(j) is not None), default=max(0, i - 1))
            gap = max(1, i - prev_idx)
            max_step = 12.0 + gap * 4.5
            if step < -10 or step > max_step:
                path[i] = None
                continue
            if abs(pick.bbox.cx - pred_cx) > 55 + gap * 3:
                path[i] = None
                continue
            vx = max(0.8, min(0.6 * vx + 0.4 * step / max(1, gap), 4.0))

        bb = pick.bbox
        if bb.cx > frame_w * 0.92:
            path[i] = None
            continue
        path[i] = bb
        prev_bb = bb

    real = [i for i in range(n) if path.get(i) is not None]
    if len(real) >= 2:
        tail = real[-min(10, len(real)) :]
        vxs = [
            (path[tail[j]].cx - path[tail[j - 1]].cx) / max(1, tail[j] - tail[j - 1])  # type: ignore[union-attr]
            for j in range(1, len(tail))
            if path[tail[j]] and path[tail[j - 1]]
        ]
        if vxs:
            vx = max(0.8, min(sum(vxs) / len(vxs), 3.5))

    sidewalk_cy = _median_sidewalk_cy(path, n)
    return extrapolate_path(
        path,
        n,
        vx_override=vx,
        cx_cap=frame_w * 0.86,
        extrapolate_until=extrapolate_until,
        cy_override=sidewalk_cy,
    )


def _find_red_bus_occlusion(
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    path: dict[int, BBox | None],
    n: int,
) -> tuple[int, int] | None:
    """Bus dropout window for tan-coat sidewalk demo (class 5/7 large vehicles)."""
    flags: list[bool] = []
    for i in range(n):
        abs_i = clip_start + i
        anchor = path.get(i)
        if anchor is None:
            flags.append(False)
            continue
        big_bus = any(
            d.class_id in {5, 7} and d.bbox.area > 12000 for d in all_dets.get(abs_i, [])
        )
        if not big_bus:
            big_bus = any(d.class_id == 2 and d.bbox.area > 22000 for d in all_dets.get(abs_i, []))
        vis = _person_visible(all_dets.get(abs_i, []), anchor)
        flags.append(big_bus and not vis)
    best_s, best_len = _longest_run(flags)
    if best_len >= 6:
        pad = max(4, best_len // 4)
        return max(0, best_s - pad), min(n, best_s + best_len + pad)
    return None


def build_tan_coat_clip(
    all_dets: dict[int, list[Detection]],
    start: int = 55,
    end: int = 180,
    oc_start: int | None = None,
    oc_end: int | None = None,
    frame_w: float = 544.0,
) -> PedestrianClip:
    """Curated clip: tan-coat woman with vehicle-occlusion ghost windows."""
    from lumen.pipelines.persist_occlusion import (
        finalize_anchor_path,
        find_all_occlusion_windows,
    )

    n_full = end - start
    anchor = build_leftmost_woman_path(all_dets, start, start + n_full, extrapolate_until=n_full, frame_w=frame_w)
    windows = find_all_occlusion_windows(anchor, all_dets, start, n_full)
    anchor = snap_ghost_off_vehicles(anchor, all_dets, start, windows, frame_w)
    anchor, windows, clip_len = finalize_anchor_path(
        anchor, all_dets, start, frame_w, windows, post_exit_frames=5
    )

    if oc_start is None or oc_end is None:
        if windows:
            oc_start = windows[-1][0]
            oc_end = windows[-1][1]
        else:
            oc_start, oc_end = 0, 0

    return PedestrianClip(
        track_id=0,
        start=start,
        end=start + clip_len,
        oc_start=oc_start,
        oc_end=oc_end,
        anchor_path=anchor,
        score=999.0,
        occlusion_windows=windows,
    )


def _person_visible(all_dets: list[Detection], anchor: BBox, thresh: float = 0.18) -> bool:
    return any(d.class_id == PERSON and iou(anchor, d.bbox) > thresh for d in all_dets)


def _x_overlap(a: BBox, b: BBox) -> float:
    ix1, ix2 = max(a.x1, b.x1), min(a.x2, b.x2)
    if ix2 <= ix1:
        return 0.0
    return (ix2 - ix1) / max(1.0, min(a.w, b.w))


def _vehicle_near_anchor(all_dets: list[Detection], anchor: BBox) -> bool:
    for d in all_dets:
        if d.class_id not in OCCLUDER_CLASSES:
            continue
        if d.class_id == 2 and d.bbox.area < max(4000, anchor.area * 1.5):
            continue
        if iou(anchor, d.bbox) > 0.02:
            return True
        if _x_overlap(anchor, d.bbox) > 0.20 and abs(anchor.cy - d.bbox.cy) < anchor.h * 3.5:
            return True
    return False


def _longest_run(flags: list[bool]) -> tuple[int, int]:
    best_s, best_len, rs, rl = 0, 0, None, 0
    for i, flag in enumerate(flags):
        if flag:
            rs, rl = (i, 1) if rs is None else (rs, rl + 1)
        elif rs is not None:
            if rl > best_len:
                best_s, best_len = rs, rl
            rs, rl = None, 0
    if rs is not None and rl > best_len:
        best_s, best_len = rs, rl
    return best_s, best_len


def _cooccur_score(track_map: dict[int, list[tuple[int, BBox]]], tid: int, path: dict[int, BBox | None], n: int) -> float:
    score = 0.0
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            continue
        others = sum(
            1
            for t, bb in track_map.get(i, [])
            if t != tid and iou(anchor, bb) < 0.05 and abs(anchor.cy - bb.cy) < anchor.h * 0.9
        )
        if others >= 2:
            score += 1
        elif others == 1:
            score += 0.35
    return score


def find_occlusion_for_track(
    all_dets: dict[int, list[Detection]],
    path: dict[int, BBox | None],
    n: int,
    min_len: int = 4,
) -> tuple[int, int] | None:
    """Best occlusion window: bus present + target missing, then overlap, then GT dropout."""
    dropout = []
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            dropout.append(False)
            continue
        bus = any(d.class_id in {5, 7} and d.bbox.area > 6000 for d in all_dets[i])
        if not bus:
            bus = any(d.class_id == 2 and d.bbox.area > 18000 for d in all_dets[i])
        vis = _person_visible(all_dets[i], anchor)
        dropout.append(bus and not vis)

    best_s, best_len = _longest_run(dropout)
    if best_len >= min_len:
        pad = max(5, best_len // 3)
        return max(8, best_s - pad), min(n - 8, best_s + best_len + pad)

    bus_flags = []
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            bus_flags.append(False)
            continue
        bus_flags.append(_vehicle_near_anchor(all_dets[i], anchor))

    best_s, best_len = _longest_run(bus_flags)
    if best_len >= min_len:
        pad = max(6, best_len // 2)
        return max(8, best_s - pad // 2), min(n - 8, best_s + best_len + pad // 2)

    blocked = []
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            blocked.append(False)
            continue
        bus = _vehicle_near_anchor(all_dets[i], anchor)
        vis = _person_visible(all_dets[i], anchor)
        blocked.append(bus and not vis)
    best_s, best_len = _longest_run(blocked)
    if best_len >= min_len:
        pad = max(6, best_len // 2)
        return max(8, best_s - pad // 2), min(n - 8, best_s + best_len + pad // 2)

    oc_start, oc_end = find_occlusion_by_gt_dropout(all_dets, path, n, min_len=max(6, min_len))
    if oc_end - oc_start >= min_len + 4:
        return oc_start, oc_end
    return None


def make_pedestrian_clip(
    track_id: int,
    path: dict[int, BBox | None],
    all_dets: dict[int, list[Detection]],
    n: int,
    oc_start: int | None = None,
    oc_end: int | None = None,
    clip_start: int | None = None,
    clip_end: int | None = None,
    score: float = 0.0,
) -> PedestrianClip | None:
    if sum(1 for v in path.values() if v is not None) < 25:
        return None

    path = extrapolate_path(path, n)

    if oc_start is None or oc_end is None:
        oc = find_occlusion_for_track(all_dets, path, n)
        if oc is None:
            return None
        oc_start, oc_end = oc

    if clip_start is None:
        clip_start = max(0, oc_start - 22)
    if clip_end is None:
        clip_end = min(n, oc_end + 22)
    if clip_end - clip_start < 45:
        return None

    trimmed = fill_gaps({i - clip_start: path[i] for i in range(clip_start, clip_end)})
    return PedestrianClip(
        track_id=track_id,
        start=clip_start,
        end=clip_end,
        oc_start=oc_start - clip_start,
        oc_end=oc_end - clip_start,
        anchor_path=trimmed,
        score=score,
    )


def find_women_group_clip(
    all_dets: dict[int, list[Detection]],
    n: int,
    prefer_track: int | None = None,
) -> PedestrianClip | None:
    """Pick a walking person occluded by bus or larger foreground."""
    people = {i: [d for d in all_dets[i] if d.class_id == PERSON] for i in range(n)}
    track_map = _track_people(people)
    presence: dict[int, int] = {}
    for i in range(n):
        for tid, _ in track_map.get(i, []):
            presence[tid] = presence.get(tid, 0) + 1

    scored: list[tuple[int, float, dict[int, BBox | None]]] = []
    for tid, pres in presence.items():
        if pres < 25:
            continue
        path = build_path(track_map, tid, n)
        co = _cooccur_score(track_map, tid, path, n)
        bus_hits = sum(
            1
            for i in range(n)
            if path.get(i) is not None and _vehicle_near_anchor(all_dets[i], path[i])  # type: ignore[arg-type]
        )
        main_bus = sum(
            1
            for i in range(100, min(n, 185))
            if path.get(i) is not None and _vehicle_near_anchor(all_dets[i], path[i])  # type: ignore[arg-type]
        )
        score = co * 3.0 + main_bus * 4.0 + bus_hits * 1.0 + pres * 0.05
        scored.append((tid, score, path))

    scored.sort(key=lambda x: -x[1])
    if prefer_track is not None:
        scored = [s for s in scored if s[0] == prefer_track] + [s for s in scored if s[0] != prefer_track]

    best: PedestrianClip | None = None
    for tid, score, path in scored[:15]:
        clip = make_pedestrian_clip(tid, path, all_dets, n, score=score)
        if clip is None:
            continue
        if best is None or clip.score > best.score:
            best = clip
    return best


def scan_video_for_pedestrian_clip(
    all_dets: dict[int, list[Detection]],
    total: int,
    window: int = 200,
    step: int = 20,
    prefer_track: int | None = None,
) -> PedestrianClip | None:
    best: PedestrianClip | None = None
    for off in range(0, max(1, total - 45), step):
        end = min(total, off + window)
        sub = {i: all_dets[off + i] for i in range(end - off)}
        clip = find_women_group_clip(sub, end - off, prefer_track=prefer_track)
        if clip is None:
            continue
        full = PedestrianClip(
            track_id=clip.track_id,
            start=off + clip.start,
            end=off + clip.end,
            oc_start=clip.oc_start,
            oc_end=clip.oc_end,
            anchor_path=clip.anchor_path,
            score=clip.score,
        )
        if best is None or full.score > best.score:
            best = full
    if best is None and total >= 45:
        return find_women_group_clip(all_dets, total, prefer_track=prefer_track)
    return best


def clip_from_manual(
    all_dets: dict[int, list[Detection]],
    n: int,
    track_id: int,
    clip_start: int,
    clip_end: int,
    oc_start: int,
    oc_end: int,
) -> PedestrianClip:
    people = {i: [d for d in all_dets[i] if d.class_id == PERSON] for i in range(n)}
    track_map = _track_people(people)
    path = build_path(track_map, track_id, n)
    path = extrapolate_path(path, n)
    trimmed = fill_gaps({i - clip_start: path[i] for i in range(clip_start, clip_end)})
    return PedestrianClip(
        track_id=track_id,
        start=clip_start,
        end=clip_end,
        oc_start=oc_start,
        oc_end=oc_end,
        anchor_path=trimmed,
        score=999.0,
    )
