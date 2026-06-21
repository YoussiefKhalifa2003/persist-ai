"""Build target paths from a user-selected detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from lumen.core.motion_model import MotionModel
from lumen.data.pedestrian_clip_finder import extrapolate_path
from lumen.pipelines.comparison_pipeline import fill_gaps, iou
from lumen.pipelines.persist_occlusion import finalize_anchor_path, find_all_occlusion_windows
from lumen.types import BBox, Detection

PERSON = 0
VEHICLES = {1, 2, 3, 5, 7}


@dataclass
class SelectableTargetClip:
    class_id: int
    start: int
    end: int
    selected_frame: int
    anchor_path: dict[int, BBox | None]
    occlusion_windows: list[tuple[int, int]]
    visible_frames: set[int]
    predicted_paths: dict[int, list[tuple[float, float]]]
    confidence_by_frame: dict[int, float]
    target_memory: "TargetMemory | None" = None
    tracking_quality: str = "high"
    failure_mode: str | None = None
    identity_switch_guard: bool = True
    prediction_mode_by_frame: dict[int, str] | None = None
    uncertainty_radius_by_frame: dict[int, float] | None = None


@dataclass
class TargetMemory:
    """Selected target identity state; YOLO boxes are observations, not identity."""

    target_id: str
    class_id: int
    selected_frame: int
    selected_bbox: BBox
    stable_width: float
    stable_height: float
    aspect_ratio: float
    foot_y: float
    appearance_template: object | None
    full_body_frames: set[int]
    state_by_frame: dict[int, str]

    def stable_box_at(self, cx: float, foot_y: float | None = None) -> BBox:
        fy = self.foot_y if foot_y is None else foot_y
        return BBox(
            cx - self.stable_width / 2,
            fy - self.stable_height,
            cx + self.stable_width / 2,
            fy,
        )


def class_name(class_id: int) -> str:
    return {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }.get(class_id, f"class {class_id}")


def candidate_quality(det: Detection) -> bool:
    return det.bbox.area >= 64 and det.confidence >= 0.05 and det.bbox.w >= 4 and det.bbox.h >= 8


def candidate_tracking_quality(det: Detection) -> str:
    if det.class_id == PERSON:
        if det.bbox.h >= 40 and det.confidence >= 0.18:
            return "high"
        if det.bbox.h >= 24 and det.confidence >= 0.12:
            return "degraded"
        return "low"
    if det.class_id in VEHICLES:
        if det.bbox.area >= 1200 and det.confidence >= 0.18:
            return "high"
        if det.bbox.area >= 420 and det.confidence >= 0.12:
            return "degraded"
        return "low"
    return "low"


def build_candidates(
    dets_map: dict[int, list[Detection]],
    abs_frame: int,
    supported_classes: set[int],
) -> list[dict]:
    candidates: list[dict] = []
    for idx, det in enumerate(dets_map.get(abs_frame, [])):
        if det.class_id not in supported_classes or not candidate_quality(det):
            continue
        candidates.append(
            {
                "id": f"{abs_frame}:{idx}",
                "index": idx,
                "frame": abs_frame,
                "class_id": det.class_id,
                "class_name": class_name(det.class_id),
                "confidence": round(det.confidence, 3),
                "bbox": [round(v, 2) for v in det.bbox.as_xyxy()],
                "selectable": True,
                "tracking_quality": candidate_tracking_quality(det),
            }
        )
    return candidates


def _center_distance(a: BBox, b: BBox) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _size_ratio_ok(a: BBox, b: BBox, class_id: int) -> bool:
    if a.area <= 0 or b.area <= 0:
        return False
    ratio = max(a.area, b.area) / max(1.0, min(a.area, b.area))
    return ratio <= (4.5 if class_id in VEHICLES else 2.8)


def _gate(prev: BBox, det: Detection, gap: int) -> bool:
    if det.class_id == PERSON:
        max_dist = 70.0 + gap * 18.0
    else:
        max_dist = 150.0 + gap * 35.0
    return _center_distance(prev, det.bbox) <= max_dist and _size_ratio_ok(prev, det.bbox, det.class_id)


def _person_step_limit(prev: BBox, velocity: tuple[float, float], gap: int) -> float:
    """Scale-aware physical motion gate for a locked pedestrian."""
    speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1]) ** 0.5
    scale_allowance = max(14.0, min(34.0, prev.w * 0.62 + prev.h * 0.08))
    return scale_allowance + min(28.0, speed * max(1, gap) * 1.35) + max(0, gap - 1) * 7.0


def _person_step_ok(prev: BBox, candidate: BBox, velocity: tuple[float, float], gap: int) -> bool:
    pred_cx = prev.cx + velocity[0] * max(1, gap)
    pred_cy = prev.cy + velocity[1] * max(1, gap)
    dx = abs(candidate.cx - pred_cx)
    dy = abs(candidate.cy - pred_cy)
    limit = _person_step_limit(prev, velocity, gap)
    return dx <= limit and dy <= max(13.0, limit * 0.65)


def _flow_predict(prev_frame, next_frame, prev_bb: BBox, class_id: int) -> tuple[BBox, float] | None:
    """Predict the clicked target's next box from local image motion."""
    if prev_frame is None or next_frame is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    h, w = prev_frame.shape[:2]
    x1 = max(0, min(w - 1, int(prev_bb.x1)))
    y1 = max(0, min(h - 1, int(prev_bb.y1)))
    x2 = max(0, min(w, int(prev_bb.x2)))
    y2 = max(0, min(h, int(prev_bb.y2)))
    if x2 - x1 < 10 or y2 - y1 < 18:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    if class_id == PERSON:
        # Torso pixels are more identity-stable than legs/feet in crowded scenes.
        my1 = y1 + int((y2 - y1) * 0.16)
        my2 = y1 + int((y2 - y1) * 0.78)
        mx1 = x1 + int((x2 - x1) * 0.12)
        mx2 = x1 + int((x2 - x1) * 0.88)
    else:
        my1, my2 = y1, y2
        mx1, mx2 = x1, x2
    mask[my1:my2, mx1:mx2] = 255

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=70,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=5,
        mask=mask,
    )
    if pts is None or len(pts) < 6:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        pts,
        None,
        winSize=(19, 19),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if nxt is None or status is None:
        return None
    good_prev = pts[status.reshape(-1) == 1].reshape(-1, 2)
    good_next = nxt[status.reshape(-1) == 1].reshape(-1, 2)
    if len(good_prev) < 5:
        return None

    deltas = good_next - good_prev
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))
    residual = np.sqrt(((deltas[:, 0] - dx) ** 2) + ((deltas[:, 1] - dy) ** 2))
    stable = float(np.mean(residual < (7.0 if class_id == PERSON else 12.0)))
    if stable < 0.48:
        return None
    max_dx = 35.0 if class_id == PERSON else 75.0
    max_dy = 22.0 if class_id == PERSON else 40.0
    dx = max(-max_dx, min(max_dx, dx))
    dy = max(-max_dy, min(max_dy, dy))
    return (
        BBox(prev_bb.x1 + dx, prev_bb.y1 + dy, prev_bb.x2 + dx, prev_bb.y2 + dy),
        stable,
    )


def _appearance_vector(frame, bb: BBox, class_id: int):
    """Compact color signature for identity locking; None when crop is unusable."""
    if frame is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(bb.x1)))
    y1 = max(0, min(h - 1, int(bb.y1)))
    x2 = max(0, min(w, int(bb.x2)))
    y2 = max(0, min(h, int(bb.y2)))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if class_id == PERSON:
        # Favor torso/coat color over legs, pavement, and head pixels.
        crop = crop[max(0, int(ch * 0.18)) : max(1, int(ch * 0.78)), int(cw * 0.12) : max(1, int(cw * 0.88))]
    else:
        crop = crop[int(ch * 0.08) : max(1, int(ch * 0.92)), int(cw * 0.06) : max(1, int(cw * 0.94))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [24, 12, 8], [0, 180, 20, 256, 20, 256])
    cv2.normalize(hist, hist)
    return hist


def _appearance_distance(a, b) -> float:
    if a is None or b is None:
        return 0.35
    import cv2

    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _blend_appearance(a, b, alpha: float = 0.12):
    if a is None:
        return b
    if b is None:
        return a
    out = (1.0 - alpha) * a + alpha * b
    try:
        import cv2

        cv2.normalize(out, out)
    except ImportError:
        pass
    return out


def _pick_next(
    prev: BBox,
    dets: list[Detection],
    class_id: int,
    gap: int,
    template,
    root_template=None,
    root_size: tuple[float, float] | None = None,
    frame=None,
    velocity: tuple[float, float] = (0.0, 0.0),
    flow_pred: BBox | None = None,
) -> tuple[BBox, object | None] | None:
    pred = BBox(
        prev.x1 + velocity[0] * gap,
        prev.y1 + velocity[1] * gap,
        prev.x2 + velocity[0] * gap,
        prev.y2 + velocity[1] * gap,
    )
    if flow_pred is not None:
        pred = BBox(
            pred.x1 * 0.35 + flow_pred.x1 * 0.65,
            pred.y1 * 0.35 + flow_pred.y1 * 0.65,
            pred.x2 * 0.35 + flow_pred.x2 * 0.65,
            pred.y2 * 0.35 + flow_pred.y2 * 0.65,
        )
    same_class = [
        d
        for d in dets
        if d.class_id == class_id
        and (_gate(prev, d, gap) or _center_distance(pred, d.bbox) <= (60.0 if class_id == PERSON else 135.0))
        and _size_ratio_ok(prev, d.bbox, class_id)
        and (class_id != PERSON or _person_step_ok(prev, d.bbox, velocity, gap))
    ]
    if not same_class:
        return None

    scored: list[tuple[float, float, BBox, object | None, float]] = []
    for det in same_class:
        prev_dist = _center_distance(prev, det.bbox)
        near_continuous_body = class_id == PERSON and prev_dist <= max(5.0, prev.w * 0.45)
        if class_id == PERSON and root_size is not None:
            rw, rh = root_size
            root_area = max(1.0, rw * rh)
            min_w = rw * (0.58 if near_continuous_body else 0.74)
            min_h = max(48.0 if near_continuous_body else 55.0, rh * (0.68 if near_continuous_body else 0.74))
            min_area = root_area * (0.46 if near_continuous_body else 0.62)
            if (
                det.bbox.h < min_h
                or det.bbox.w < min_w
                or det.bbox.area < min_area
                or det.bbox.h > rh * 1.35
                or det.bbox.w > rw * 1.45
                or det.bbox.area > root_area * 1.65
            ):
                continue
        vec = _appearance_vector(frame, det.bbox, class_id)
        local_app = _appearance_distance(template, vec)
        root_app = _appearance_distance(root_template, vec)
        if class_id == PERSON and root_template is not None and root_app > 0.64 and not near_continuous_body:
            continue
        app = min(local_app, root_app)
        motion = _center_distance(pred, det.bbox) / (90.0 if class_id == PERSON else 180.0)
        overlap = iou(prev, det.bbox)
        size_penalty = abs(det.bbox.area - prev.area) / max(prev.area, det.bbox.area, 1.0)
        if class_id == PERSON:
            continuity = prev_dist / max(1.0, _person_step_limit(prev, velocity, gap))
            score = motion * 0.55 + continuity * 1.45 + app * 3.1 + size_penalty * 0.25 - overlap * 0.35
        else:
            score = motion * 1.15 + app * 1.4 + size_penalty * 0.2 - overlap * 0.28
        scored.append((score, app, det.bbox, vec, prev_dist))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    best_score, best_app, best_bb, best_vec, best_prev_dist = scored[0]
    if class_id == PERSON and len(scored) > 1:
        near_limit = max(5.0, prev.w * 0.45)
        near_picks = [item for item in scored if item[4] <= near_limit]
        if near_picks:
            near_picks.sort(key=lambda item: (item[4], item[0]))
            if (
                len(near_picks) > 1
                and _center_distance(near_picks[0][2], near_picks[1][2]) < max(8.0, prev.w * 0.42)
                and abs(near_picks[0][0] - near_picks[1][0]) < 0.24
            ):
                return None
            best_score, best_app, best_bb, best_vec, best_prev_dist = near_picks[0]
        else:
            continuity_pick = min(scored, key=lambda item: item[4])
            _cont_score, cont_app, cont_bb, cont_vec, cont_prev_dist = continuity_pick
            jump_thresh = max(13.0, prev.w * 0.72)
            if (
                cont_bb is not best_bb
                and best_prev_dist > jump_thresh
                and cont_prev_dist < best_prev_dist * 0.58
                and cont_app - best_app < 0.38
            ):
                best_score, best_app, best_bb, best_vec, best_prev_dist = continuity_pick
    if template is not None and class_id == PERSON and len(scored) > 1:
        # When people are shoulder-to-shoulder, do not accept a visually different body
        # just because it is nearby. Mark missing and let occlusion/latent logic take over.
        if best_app > 0.58 and scored[1][0] - best_score < 0.28:
            return None
        if best_app < 0.24 and scored[1][1] < 0.24 and scored[1][0] - best_score < 0.16:
            return None
        if best_app > 0.48 and _center_distance(pred, best_bb) > max(28.0, best_bb.w * 0.85):
            return None
    return best_bb, best_vec


def _track_direction(
    all_dets: dict[int, list[Detection]],
    path: dict[int, BBox | None],
    selected_rel: int,
    selected_bb: BBox,
    class_id: int,
    clip_start: int,
    clip_len: int,
    direction: int,
    max_gap: int,
    frame_provider: Callable[[int], object | None] | None = None,
    visible_frames: set[int] | None = None,
) -> None:
    prev_i = selected_rel
    prev_bb = selected_bb
    template = _appearance_vector(
        frame_provider(clip_start + selected_rel) if frame_provider else None,
        selected_bb,
        class_id,
    )
    root_template = template
    root_size = (selected_bb.w, selected_bb.h)
    velocity = (0.0, 0.0)
    flow_bb: BBox | None = selected_bb
    flow_rel = selected_rel
    i = selected_rel + direction
    gap = 1
    while 0 <= i < clip_len:
        frame = frame_provider(clip_start + i) if frame_provider else None
        flow_pred: BBox | None = None
        if frame_provider and flow_bb is not None:
            prev_flow_frame = frame_provider(clip_start + flow_rel)
            flow_result = _flow_predict(prev_flow_frame, frame, flow_bb, class_id)
            if flow_result is not None:
                candidate_flow, flow_stability = flow_result
                if class_id != PERSON or (
                    flow_stability >= 0.68
                    and _person_step_ok(flow_bb, candidate_flow, velocity, max(1, abs(i - flow_rel)))
                ):
                    flow_pred = candidate_flow
                    flow_bb = flow_pred
                    flow_rel = i
        picked = _pick_next(
            prev_bb,
            all_dets.get(clip_start + i, []),
            class_id,
            min(gap, max_gap),
            template,
            root_template,
            root_size,
            frame,
            velocity,
            flow_pred,
        )
        match = picked[0] if picked else None
        if match is None:
            path[i] = None
            gap += 1
            if gap > max_gap:
                i += direction
                continue
        else:
            path[i] = match
            if visible_frames is not None:
                visible_frames.add(i)
            step = max(1, abs(i - prev_i))
            if class_id == PERSON:
                vx = (match.cx - prev_bb.cx) / step
                vy = (match.y2 - prev_bb.y2) / step
                velocity = (max(-12.0, min(12.0, vx)), max(-3.0, min(3.0, vy)))
            else:
                velocity = (
                    (match.cx - prev_bb.cx) / step,
                    (match.cy - prev_bb.cy) / step,
                )
            if direction < 0:
                velocity = (-velocity[0], -velocity[1])
            if picked and picked[1] is not None and _appearance_distance(template, picked[1]) < 0.44:
                template = _blend_appearance(template, picked[1])
            prev_i = i
            prev_bb = match
            flow_bb = match
            flow_rel = i
            gap = 1
        i += direction


def _extrapolate_generic(path: dict[int, BBox | None], clip_len: int, class_id: int) -> dict[int, BBox | None]:
    if class_id == PERSON:
        return extrapolate_path(path, clip_len, max_extra=55, extrapolate_until=clip_len)
    filled = fill_gaps(path)
    real = [i for i in range(clip_len) if path.get(i) is not None]
    if len(real) < 2:
        return filled
    sizes = [(path[i].w, path[i].h) for i in real if path[i] is not None]
    mw = sorted(s[0] for s in sizes)[len(sizes) // 2]
    mh = sorted(s[1] for s in sizes)[len(sizes) // 2]
    tail = real[-min(8, len(real)) :]
    vx = sum(
        (filled[tail[j]].cx - filled[tail[j - 1]].cx) / max(1, tail[j] - tail[j - 1])  # type: ignore[union-attr]
        for j in range(1, len(tail))
    ) / max(1, len(tail) - 1)
    vy = sum(
        (filled[tail[j]].cy - filled[tail[j - 1]].cy) / max(1, tail[j] - tail[j - 1])  # type: ignore[union-attr]
        for j in range(1, len(tail))
    ) / max(1, len(tail) - 1)
    vx = max(-8.0, min(vx, 8.0))
    vy = max(-5.0, min(vy, 5.0))
    last_i = real[-1]
    last_bb = filled[last_i]
    assert last_bb is not None
    for j in range(last_i + 1, min(clip_len, last_i + 36)):
        if filled.get(j) is not None:
            continue
        t = j - last_i
        cx = last_bb.cx + vx * t
        cy = last_bb.cy + vy * t
        filled[j] = BBox(cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2)
    return filled


def _stabilize_person_boxes(path: dict[int, BBox | None], selected_bb: BBox, clip_len: int) -> dict[int, BBox | None]:
    """Keep selected-person anchors full-body sized through partial observations."""
    out = dict(path)
    candidates = [
        (bb.w, bb.h)
        for bb in out.values()
        if bb is not None and bb.h >= selected_bb.h * 0.78 and bb.w >= selected_bb.w * 0.78
    ]
    if candidates:
        widths = sorted(w for w, _ in candidates)
        heights = sorted(h for _, h in candidates)
        ref_w = widths[len(widths) // 2]
        ref_h = heights[len(heights) // 2]
    else:
        ref_w, ref_h = selected_bb.w, selected_bb.h
    ref_w = max(ref_w, selected_bb.w)
    ref_h = max(ref_h, selected_bb.h)

    stable_cys = [
        bb.cy
        for bb in out.values()
        if bb is not None and bb.h >= ref_h * 0.78 and bb.w >= ref_w * 0.72
    ]
    stable_cys.sort()
    median_cy = stable_cys[len(stable_cys) // 2] if stable_cys else selected_bb.cy

    last_full: BBox | None = None
    for i in range(clip_len):
        bb = out.get(i)
        if bb is None:
            continue
        partial = bb.w < ref_w * 0.82 or bb.h < ref_h * 0.82 or bb.area < (ref_w * ref_h) * 0.74
        if partial:
            cy = last_full.cy if last_full is not None else median_cy
            out[i] = BBox(bb.cx - ref_w / 2, cy - ref_h / 2, bb.cx + ref_w / 2, cy + ref_h / 2)
        else:
            last_full = bb
    return out


def _remove_visible_frames_from_windows(
    windows: list[tuple[int, int]],
    visible_frames: set[int],
    min_len: int = 2,
) -> list[tuple[int, int]]:
    refined: list[tuple[int, int]] = []
    for start, end in windows:
        run_start: int | None = None
        for i in range(start, end):
            if i in visible_frames:
                if run_start is not None and i - run_start >= min_len:
                    refined.append((run_start, i))
                run_start = None
                continue
            if run_start is None:
                run_start = i
        if run_start is not None and end - run_start >= min_len:
            refined.append((run_start, end))
    return refined


def _build_target_memory(
    target_id: str,
    class_id: int,
    selected_rel: int,
    selected_bb: BBox,
    path: dict[int, BBox | None],
    visible_frames: set[int],
    windows: list[tuple[int, int]],
    confidence_by_frame: dict[int, float],
    frame_provider: Callable[[int], object | None] | None,
    clip_start: int,
) -> TargetMemory:
    reliable = [
        path[i]
        for i in sorted(visible_frames)
        if path.get(i) is not None and path[i].h >= selected_bb.h * 0.78 and path[i].w >= selected_bb.w * 0.72
    ]
    if reliable:
        widths = sorted(bb.w for bb in reliable if bb is not None)
        heights = sorted(bb.h for bb in reliable if bb is not None)
        foots = sorted(bb.y2 for bb in reliable if bb is not None)
        stable_w = max(selected_bb.w, widths[len(widths) // 2])
        stable_h = max(selected_bb.h, heights[len(heights) // 2])
        foot_y = foots[len(foots) // 2]
    else:
        stable_w, stable_h, foot_y = selected_bb.w, selected_bb.h, selected_bb.y2

    state_by_frame: dict[int, str] = {}
    n = max(path.keys()) + 1 if path else 0
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            state_by_frame[i] = "EXITED"
        elif i in visible_frames:
            state_by_frame[i] = "VISIBLE"
        elif _inside_windows(i, windows):
            state_by_frame[i] = "OCCLUDED" if confidence_by_frame.get(i, 0.0) >= 0.45 else "PREDICTED"
        else:
            state_by_frame[i] = "PREDICTED"

    template = _appearance_vector(
        frame_provider(clip_start + selected_rel) if frame_provider else None,
        selected_bb,
        class_id,
    )
    return TargetMemory(
        target_id=target_id,
        class_id=class_id,
        selected_frame=selected_rel,
        selected_bbox=selected_bb,
        stable_width=stable_w,
        stable_height=stable_h,
        aspect_ratio=stable_w / max(1.0, stable_h),
        foot_y=foot_y,
        appearance_template=template,
        full_body_frames=set(visible_frames),
        state_by_frame=state_by_frame,
    )


def _inside_windows(frame_idx: int, windows: list[tuple[int, int]]) -> bool:
    return any(start <= frame_idx < end for start, end in windows)


def _recent_velocity(
    path: dict[int, BBox | None],
    visible_frames: set[int],
    frame_idx: int,
    lookback: int = 12,
) -> tuple[float, float]:
    refs = [
        i
        for i in range(max(0, frame_idx - lookback), frame_idx + 1)
        if i in visible_frames and path.get(i) is not None
    ]
    if len(refs) < 2:
        return (0.0, 0.0)
    velocities: list[tuple[float, float]] = []
    for a, b in zip(refs, refs[1:]):
        ba, bb = path[a], path[b]
        if ba is None or bb is None:
            continue
        dt = max(1, b - a)
        velocities.append(((bb.cx - ba.cx) / dt, (bb.cy - ba.cy) / dt))
    if not velocities:
        return (0.0, 0.0)
    vx = sorted(v[0] for v in velocities)[len(velocities) // 2]
    vy = sorted(v[1] for v in velocities)[len(velocities) // 2]
    return (
        max(-9.0, min(9.0, vx)),
        max(-5.0, min(5.0, vy)),
    )


def _recent_motion_profile(
    path: dict[int, BBox | None],
    visible_frames: set[int],
    frame_idx: int,
    lookback: int = 16,
) -> tuple[float, float, float]:
    refs = [
        i
        for i in range(max(0, frame_idx - lookback), frame_idx + 1)
        if i in visible_frames and path.get(i) is not None
    ]
    if len(refs) < 2:
        return 0.0, 0.0, 0.0
    velocities: list[tuple[float, float]] = []
    speeds: list[float] = []
    for a, b in zip(refs, refs[1:]):
        ba, bb = path[a], path[b]
        if ba is None or bb is None:
            continue
        dt = max(1, b - a)
        vx = (bb.cx - ba.cx) / dt
        vy = (bb.y2 - ba.y2) / dt
        velocities.append((vx, vy))
        speeds.append((vx * vx + vy * vy) ** 0.5)
    if not velocities:
        return 0.0, 0.0, 0.0
    vx = sorted(v[0] for v in velocities)[len(velocities) // 2]
    vy = sorted(v[1] for v in velocities)[len(velocities) // 2]
    speed = sorted(speeds)[len(speeds) // 2]
    return max(-9.0, min(9.0, vx)), max(-5.0, min(5.0, vy)), speed


def _apply_kalman_prediction(
    path: dict[int, BBox | None],
    visible_frames: set[int],
    windows: list[tuple[int, int]],
    frame_w: float,
    target_class_id: int,
) -> tuple[
    dict[int, BBox | None],
    dict[int, list[tuple[float, float]]],
    dict[int, float],
    dict[int, str],
    dict[int, float],
]:
    """Use visible selected-target boxes as measurements and latent windows as prediction."""
    n = max(path.keys()) + 1 if path else 0
    out = dict(path)
    predicted_paths: dict[int, list[tuple[float, float]]] = {}
    confidence_by_frame: dict[int, float] = {}
    prediction_mode_by_frame: dict[int, str] = {}
    uncertainty_radius_by_frame: dict[int, float] = {}
    model = MotionModel(process_noise=0.45 if target_class_id == PERSON else 0.9, measurement_noise=3.5)
    initialized = False
    last_wh: tuple[float, float] | None = None
    last_reliable_velocity = (0.0, 0.0)
    last_reliable_speed = 0.0
    latent_steps = 0

    for i in range(n):
        bb = out.get(i)
        if i in visible_frames and bb is not None:
            if not initialized:
                model.init_from_bbox(bb)
                initialized = True
            else:
                model.predict(latent=False)
                model.update(bb)
            last_wh = (bb.w, bb.h)
            latent_steps = 0
            rvx, rvy, rspeed = _recent_motion_profile(out, visible_frames, i)
            if rspeed >= 0.08:
                last_reliable_velocity = (rvx, rvy)
                last_reliable_speed = rspeed
            elif i > 0:
                last_reliable_velocity = (0.0, 0.0)
                last_reliable_speed = 0.0
            confidence_by_frame[i] = 1.0
        elif initialized and bb is not None and _inside_windows(i, windows):
            cx, cy, vx, vy = model.predict(latent=True)
            damping = max(0.12, 1.0 - latent_steps * (0.075 if target_class_id == PERSON else 0.045))
            if target_class_id == PERSON:
                # Keep walking targets on the same ground-plane band unless the
                # measured path proves otherwise. This avoids ghost boxes sliding
                # down foreground occluders.
                rvx, rvy, rspeed = _recent_motion_profile(out, visible_frames, i)
                if rspeed < 0.08:
                    rvx, rvy, rspeed = (*last_reliable_velocity, last_reliable_speed)
                if rspeed < 0.38:
                    vx, vy = 0.0, 0.0
                else:
                    vx = rvx * damping
                    vy = max(-2.0, min(2.0, rvy * damping))
                cx = bb.cx * 0.35 + (out.get(i - 1).cx + vx if out.get(i - 1) else cx) * 0.65
                cy = bb.cy * 0.25 + (out.get(i - 1).cy + vy if out.get(i - 1) else cy) * 0.75
            else:
                vx *= damping
                vy *= damping
            w, h = last_wh or (bb.w, bb.h)
            cx = max(w / 2, min(frame_w - w / 2, cx))
            out[i] = BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            latent_steps += 1
            confidence_by_frame[i] = max(0.22, 0.94 - latent_steps * 0.055)
        elif bb is not None:
            confidence_by_frame[i] = 0.65

        cur = out.get(i)
        if cur is None:
            continue
        vx, vy, speed = _recent_motion_profile(out, visible_frames, i)
        if speed < 0.08 and initialized:
            vx, vy = last_reliable_velocity
            speed = last_reliable_speed
        if i not in visible_frames:
            missing_age = 1
            j = i - 1
            while j >= 0 and j not in visible_frames:
                missing_age += 1
                j -= 1
            uncertainty_damping = max(0.10, 1.0 - missing_age * (0.08 if target_class_id == PERSON else 0.05))
            vx *= uncertainty_damping
            vy *= uncertainty_damping
            speed *= uncertainty_damping
        if target_class_id == PERSON:
            if speed < 0.38:
                mode = "stationary"
                vx = 0.0
                vy = 0.0
            elif i in visible_frames:
                mode = "walking"
            else:
                mode = "uncertain"
        elif target_class_id in VEHICLES:
            mode = "vehicle"
        else:
            mode = "uncertain"
        prediction_mode_by_frame[i] = mode
        horizon = 18 if target_class_id == PERSON else 12
        points = [(cur.cx, cur.cy)]
        confidence = confidence_by_frame.get(i, 0.55)
        uncertainty_radius_by_frame[i] = round(
            (5.0 if target_class_id == PERSON else 9.0)
            + (1.0 - max(0.0, min(1.0, confidence))) * 34.0
            + (0.0 if i in visible_frames else 8.0),
            3,
        )
        for step in range(1, horizon + 1):
            if mode == "stationary":
                damped_step = 0.0
            else:
                damping = 1.0 if i in visible_frames else max(0.18, 1.0 - step * 0.055)
                damped_step = step * damping
            px = max(0.0, min(frame_w, cur.cx + vx * damped_step))
            py = cur.cy + vy * damped_step
            points.append((px, py))
        predicted_paths[i] = points

    return out, predicted_paths, confidence_by_frame, prediction_mode_by_frame, uncertainty_radius_by_frame


def build_selectable_target_clip(
    all_dets: dict[int, list[Detection]],
    clip_start: int,
    clip_end: int,
    selected_abs_frame: int,
    selected_index: int,
    frame_w: float,
    min_visible_frames: int = 10,
    frame_provider: Callable[[int], object | None] | None = None,
) -> SelectableTargetClip:
    clip_len = clip_end - clip_start
    selected_rel = selected_abs_frame - clip_start
    if not 0 <= selected_rel < clip_len:
        raise ValueError("Selected frame is outside the scene clip.")
    try:
        selected = all_dets[selected_abs_frame][selected_index]
    except (KeyError, IndexError) as exc:
        raise ValueError("Selected target is no longer available.") from exc
    if selected.bbox.area <= 0 or selected.bbox.w <= 0 or selected.bbox.h <= 0:
        raise ValueError("Selected target is invalid.")
    tracking_quality = candidate_tracking_quality(selected)
    failure_mode: str | None = None
    if tracking_quality != "high":
        failure_mode = "Selected detection is small or low-confidence; rendering with uncertainty."

    path: dict[int, BBox | None] = {i: None for i in range(clip_len)}
    path[selected_rel] = selected.bbox
    visible_frames: set[int] = {selected_rel}
    max_gap = 12 if selected.class_id == PERSON else 8
    _track_direction(
        all_dets,
        path,
        selected_rel,
        selected.bbox,
        selected.class_id,
        clip_start,
        clip_len,
        1,
        max_gap,
        frame_provider,
        visible_frames,
    )
    _track_direction(
        all_dets,
        path,
        selected_rel,
        selected.bbox,
        selected.class_id,
        clip_start,
        clip_len,
        -1,
        max_gap,
        frame_provider,
        visible_frames,
    )

    visible_count = sum(1 for bb in path.values() if bb is not None)
    if visible_count < min_visible_frames:
        tracking_quality = "low"
        failure_mode = "Limited target evidence; rendering with prediction-only fallback."

    path = _extrapolate_generic(path, clip_len, selected.class_id)
    if selected.class_id == PERSON:
        path = _stabilize_person_boxes(path, selected.bbox, clip_len)
    windows = find_all_occlusion_windows(
        path,
        all_dets,
        clip_start,
        clip_len,
        target_class_id=selected.class_id,
    )
    windows = _remove_visible_frames_from_windows(windows, visible_frames)
    if windows:
        path, windows, final_len = finalize_anchor_path(
            path,
            all_dets,
            clip_start,
            frame_w,
            windows,
            target_class_id=selected.class_id,
            post_exit_frames=5,
        )
    else:
        last_visible = max(visible_frames) if visible_frames else selected_rel
        final_len = min(clip_len, last_visible + 5)
        path = {i: path.get(i) for i in range(final_len)}
        for j in range(last_visible + 1, final_len):
            path[j] = None
    visible_frames = {i for i in visible_frames if i < final_len and path.get(i) is not None}
    windows = _remove_visible_frames_from_windows(windows, visible_frames)
    ghost_frames = sum(e - s for s, e in windows)
    visible_ratio = len(visible_frames) / max(1, final_len)
    min_visible_ratio = 0.45 if selected.class_id == PERSON else 0.32
    max_ghost_frames = max(28, int(len(visible_frames) * 0.75))
    if visible_ratio < min_visible_ratio or (windows and ghost_frames > max_ghost_frames):
        tracking_quality = "low" if visible_ratio < min_visible_ratio * 0.5 else "degraded"
        failure_mode = "Tracking quality is limited; rendering with uncertainty."
    (
        path,
        predicted_paths,
        confidence_by_frame,
        prediction_mode_by_frame,
        uncertainty_radius_by_frame,
    ) = _apply_kalman_prediction(
        path,
        visible_frames,
        windows,
        frame_w,
        selected.class_id,
    )
    target_memory = _build_target_memory(
        f"{selected_abs_frame}:{selected_index}",
        selected.class_id,
        selected_rel,
        selected.bbox,
        path,
        visible_frames,
        windows,
        confidence_by_frame,
        frame_provider,
        clip_start,
    )
    return SelectableTargetClip(
        class_id=selected.class_id,
        start=clip_start,
        end=clip_start + final_len,
        selected_frame=selected_rel,
        anchor_path=path,
        occlusion_windows=windows,
        visible_frames=visible_frames,
        predicted_paths=predicted_paths,
        confidence_by_frame=confidence_by_frame,
        target_memory=target_memory,
        tracking_quality=tracking_quality,
        failure_mode=failure_mode,
        identity_switch_guard=True,
        prediction_mode_by_frame=prediction_mode_by_frame,
        uncertainty_radius_by_frame=uncertainty_radius_by_frame,
    )
