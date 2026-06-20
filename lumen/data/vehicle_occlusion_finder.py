"""Find a car track occluded by a larger vehicle in YOLO detections."""

from __future__ import annotations

from dataclasses import dataclass

from lumen.pipelines.comparison_pipeline import fill_gaps, iou
from lumen.types import BBox, Detection

CAR_CLASSES = {2, 3, 5, 7}
SUBJECT_CLASSES = {2, 3}
LARGE_CLASSES = {5, 7}


@dataclass
class VehicleClip:
    track_id: int
    start: int
    end: int
    oc_start: int
    oc_end: int
    anchor_path: dict[int, BBox | None]
    score: float


def _track_vehicles(
    dets_map: dict[int, list[Detection]],
    target_classes: set[int],
) -> dict[int, list[tuple[int, BBox]]]:
    tracks: dict[int, list[tuple[int, BBox]]] = {}
    active: dict[int, BBox] = {}
    next_id = 1

    for fidx in sorted(dets_map.keys()):
        dets = [d for d in dets_map[fidx] if d.class_id in target_classes]
        matched: dict[int, BBox] = {}
        used: set[int] = set()

        for tid, prev in list(active.items()):
            best, best_iou, best_j = None, 0.0, -1
            for j, det in enumerate(dets):
                if j in used:
                    continue
                score = iou(prev, det.bbox)
                if score > best_iou:
                    best_iou, best, best_j = score, det.bbox, j
            if best is not None and best_iou > 0.12:
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


def build_track_path(
    track_map: dict[int, list[tuple[int, BBox]]],
    track_id: int,
    n: int,
) -> dict[int, BBox | None]:
    path: dict[int, BBox | None] = {}
    for i in range(n):
        hit = next((bb for tid, bb in track_map.get(i, []) if tid == track_id), None)
        path[i] = hit
    return fill_gaps(path)


def _car_visible_at_anchor(all_dets: list[Detection], anchor: BBox, thresh: float = 0.22) -> bool:
    return any(d.class_id in SUBJECT_CLASSES and iou(anchor, d.bbox) > thresh for d in all_dets)


def _occlusion_flags(
    all_dets: dict[int, list[Detection]],
    path: dict[int, BBox | None],
    n: int,
    require_large: bool = True,
) -> list[bool]:
    flags = []
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            flags.append(False)
            continue
        visible = _car_visible_at_anchor(all_dets[i], anchor)
        truck_bus = any(
            d.class_id in LARGE_CLASSES and iou(anchor, d.bbox) > 0.06
            for d in all_dets[i]
        )
        large_vehicle = any(
            d.class_id in CAR_CLASSES
            and iou(anchor, d.bbox) > 0.10
            and d.bbox.area > anchor.area * 1.75
            for d in all_dets[i]
        )
        if require_large:
            flags.append((truck_bus or large_vehicle) and not visible)
        else:
            large_near = truck_bus or any(
                d.class_id in CAR_CLASSES
                and d.class_id not in SUBJECT_CLASSES
                and iou(anchor, d.bbox) > 0.10
                for d in all_dets[i]
            )
            flags.append((large_near or truck_bus) and not visible)
    return flags


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


def find_vehicle_occlusion_clip(
    all_dets: dict[int, list[Detection]],
    n: int,
    min_occlusion: int = 6,
    clip_len: int = 80,
) -> VehicleClip | None:
    car_dets = {i: [d for d in all_dets[i] if d.class_id in SUBJECT_CLASSES] for i in range(n)}
    track_map = _track_vehicles(car_dets, SUBJECT_CLASSES)

    track_presence: dict[int, int] = {}
    for i in range(n):
        for tid, _ in track_map.get(i, []):
            track_presence[tid] = track_presence.get(tid, 0) + 1

    candidates = sorted(track_presence.items(), key=lambda x: -x[1])[:20]
    best: VehicleClip | None = None

    for tid, _ in candidates:
        path = build_track_path(track_map, tid, n)
        if sum(1 for v in path.values() if v is not None) < clip_len // 2:
            continue

        occluded = _occlusion_flags(all_dets, path, n)
        best_s, best_len = _longest_run(occluded)
        if best_len < min_occlusion:
            continue

        pad = max(8, best_len // 3)
        oc_start = max(10, best_s - pad)
        oc_end = min(n - 10, best_s + best_len + pad)
        start = max(0, oc_start - 22)
        end = min(n, oc_end + 22)
        if end - start < 45:
            continue

        trimmed_path = fill_gaps({i - start: path[i] for i in range(start, end)})

        clip = VehicleClip(
            track_id=tid,
            start=start,
            end=end,
            oc_start=oc_start - start,
            oc_end=oc_end - start,
            anchor_path=trimmed_path,
            score=best_len + 0.01 * track_presence[tid],
        )
        if best is None or clip.score > best.score:
            best = clip

    return best


def find_overlap_occlusion_clip(
    all_dets: dict[int, list[Detection]],
    n: int,
    min_overlap_run: int = 5,
) -> VehicleClip | None:
    """Car partially covered by another vehicle bbox (common in traffic)."""
    car_dets = {i: [d for d in all_dets[i] if d.class_id in SUBJECT_CLASSES] for i in range(n)}
    track_map = _track_vehicles(car_dets, SUBJECT_CLASSES)
    track_presence: dict[int, int] = {}
    for i in range(n):
        for tid, _ in track_map.get(i, []):
            track_presence[tid] = track_presence.get(tid, 0) + 1

    best: VehicleClip | None = None
    for tid, _ in sorted(track_presence.items(), key=lambda x: -x[1])[:15]:
        path = build_track_path(track_map, tid, n)
        overlapped = []
        for i in range(n):
            anchor = path.get(i)
            if anchor is None:
                overlapped.append(False)
                continue
            hit = False
            for d in all_dets[i]:
                if d.class_id not in CAR_CLASSES:
                    continue
                if iou(anchor, d.bbox) > 0.12 and d.bbox.area >= anchor.area * 0.65:
                    if not (d.class_id in SUBJECT_CLASSES and iou(anchor, d.bbox) > 0.55):
                        hit = True
                        break
            overlapped.append(hit)
        best_s, best_len = _longest_run(overlapped)
        if best_len < min_overlap_run:
            continue
        pad = max(5, best_len // 3)
        oc_start = max(8, best_s)
        oc_end = min(n - 8, best_s + best_len + pad)
        start = max(0, oc_start - 18)
        end = min(n, oc_end + 18)
        if end - start < 35:
            continue
        trimmed = fill_gaps({i - start: path[i] for i in range(start, end)})
        clip = VehicleClip(
            track_id=tid,
            start=start,
            end=end,
            oc_start=oc_start - start,
            oc_end=oc_end - start,
            anchor_path=trimmed,
            score=best_len + 0.004 * track_presence[tid],
        )
        if best is None or clip.score > best.score:
            best = clip
    return best


def find_vehicle_dropout_clip(
    all_dets: dict[int, list[Detection]],
    n: int,
    min_dropout: int = 5,
) -> VehicleClip | None:
    """Fallback: car track with a sustained YOLO miss (still on interpolated path)."""
    car_dets = {i: [d for d in all_dets[i] if d.class_id in SUBJECT_CLASSES] for i in range(n)}
    track_map = _track_vehicles(car_dets, SUBJECT_CLASSES)
    track_presence: dict[int, int] = {}
    for i in range(n):
        for tid, _ in track_map.get(i, []):
            track_presence[tid] = track_presence.get(tid, 0) + 1

    best: VehicleClip | None = None
    for tid, _ in sorted(track_presence.items(), key=lambda x: -x[1])[:15]:
        path = build_track_path(track_map, tid, n)
        missing = []
        for i in range(n):
            anchor = path.get(i)
            if anchor is None:
                missing.append(False)
            else:
                missing.append(not _car_visible_at_anchor(all_dets[i], anchor, 0.20))
        best_s, best_len = _longest_run(missing)
        if best_len < min_dropout:
            continue
        before = sum(1 for j in range(max(0, best_s - 15), best_s) if path.get(j) and _car_visible_at_anchor(all_dets[j], path[j], 0.18))
        after = sum(1 for j in range(best_s + best_len, min(n, best_s + best_len + 15)) if path.get(j) and _car_visible_at_anchor(all_dets[j], path[j], 0.18))
        if before < 2 or after < 2:
            continue
        pad = max(6, best_len // 4)
        oc_start = max(8, best_s - pad)
        oc_end = min(n - 8, best_s + best_len + pad)
        start = max(0, oc_start - 20)
        end = min(n, oc_end + 20)
        if end - start < 40:
            continue
        trimmed = fill_gaps({i - start: path[i] for i in range(start, end)})
        clip = VehicleClip(
            track_id=tid,
            start=start,
            end=end,
            oc_start=oc_start - start,
            oc_end=oc_end - start,
            anchor_path=trimmed,
            score=best_len + 0.005 * track_presence[tid],
        )
        if best is None or clip.score > best.score:
            best = clip
    return best


def scan_video_for_vehicle_clip(
    all_dets: dict[int, list[Detection]],
    window: int = 150,
    step: int = 40,
) -> VehicleClip | None:
    """Slide over long detection runs to find the best occlusion segment."""
    n = len(all_dets)
    best: VehicleClip | None = None
    for offset in range(0, max(1, n - 50), step):
        end = min(n, offset + window)
        sub = {i: all_dets[offset + i] for i in range(end - offset)}
        clip = find_vehicle_occlusion_clip(sub, end - offset)
        if clip is None:
            continue
        full = VehicleClip(
            track_id=clip.track_id,
            start=offset + clip.start,
            end=offset + clip.end,
            oc_start=clip.oc_start,
            oc_end=clip.oc_end,
            anchor_path=clip.anchor_path,
            score=clip.score,
        )
        if best is None or full.score > best.score:
            best = full
    return best


def force_car_demo_clip(
    all_dets: dict[int, list[Detection]],
    n: int,
) -> VehicleClip | None:
    """Last resort: longest car track, occlusion in middle third (masked for demo)."""
    car_dets = {i: [d for d in all_dets[i] if d.class_id in SUBJECT_CLASSES] for i in range(n)}
    track_map = _track_vehicles(car_dets, SUBJECT_CLASSES)
    track_presence: dict[int, int] = {}
    for i in range(n):
        for tid, _ in track_map.get(i, []):
            track_presence[tid] = track_presence.get(tid, 0) + 1
    if not track_presence:
        return None
    tid = max(track_presence.items(), key=lambda x: x[1])[0]
    path = build_track_path(track_map, tid, n)
    span = [i for i in range(n) if path.get(i) is not None]
    if len(span) < 30:
        return None
    start, end = span[0], span[-1] + 1
    clip_len = end - start
    oc_start = start + clip_len // 3
    oc_end = start + (2 * clip_len) // 3
    trimmed = fill_gaps({i - start: path[i] for i in range(start, end)})
    return VehicleClip(
        track_id=tid,
        start=start,
        end=end,
        oc_start=oc_start - start,
        oc_end=oc_end - start,
        anchor_path=trimmed,
        score=float(track_presence[tid]),
    )


def find_best_vehicle_clip(
    all_dets: dict[int, list[Detection]],
    n: int,
    require_truck_bus: bool = True,
) -> VehicleClip | None:
    """Pick clip where a car is hidden behind a bus or truck (no synthetic fallback)."""
    clip = scan_video_for_vehicle_clip(all_dets, n)
    if clip is not None:
        return clip
    clip = find_vehicle_occlusion_clip(all_dets, n)
    if clip is not None:
        return clip
    clip = find_overlap_occlusion_clip(all_dets, n, min_overlap_run=5)
    if clip is not None:
        return clip
    return None


def scan_full_video_for_truck_occlusion(
    all_dets: dict[int, list[Detection]],
    total_frames: int,
    window: int = 160,
    step: int = 35,
) -> VehicleClip | None:
    best: VehicleClip | None = None
    for offset in range(0, max(1, total_frames - 60), step):
        end = min(total_frames, offset + window)
        sub = {i: all_dets[offset + i] for i in range(end - offset)}
        clip = find_best_vehicle_clip(sub, end - offset, require_truck_bus=True)
        if clip is None:
            continue
        full = VehicleClip(
            track_id=clip.track_id,
            start=offset + clip.start,
            end=offset + clip.end,
            oc_start=clip.oc_start,
            oc_end=clip.oc_end,
            anchor_path=clip.anchor_path,
            score=clip.score + 0.001 * (end - offset),
        )
        if best is None or full.score > best.score:
            best = full
    return best
