"""Baseline vs PERSIST-AI side-by-side demo engine (Video 2 / Video 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lumen.core.track_manager import TrackManager
from lumen.trackers.baseline_adapter import BaselineTracker
from lumen.types import BBox, Detection, TrackOutput


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def fill_gaps(path: dict[int, BBox | None]) -> dict[int, BBox | None]:
    n = max(path.keys()) + 1 if path else 0
    filled = dict(path)
    i = 0
    while i < n:
        if filled.get(i) is not None:
            i += 1
            continue
        gs = i
        while i < n and filled.get(i) is None:
            i += 1
        ge = i
        left, right = filled.get(gs - 1), filled.get(ge) if ge < n else None
        if left and right and ge > gs:
            for j in range(gs, ge):
                t = (j - gs + 1) / (ge - gs + 1)
                cx = left.cx + (right.cx - left.cx) * t
                cy = left.cy + (right.cy - left.cy) * t
                w, h = left.w, left.h
                filled[j] = BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return filled


def match_track_to_anchor(
    tracks: list[tuple[int, BBox]] | list[TrackOutput],
    anchor: BBox | None,
    min_iou: float = 0.12,
) -> tuple[int, BBox] | None:
    if anchor is None:
        return None
    best: tuple[int, BBox] | None = None
    best_iou = min_iou
    for item in tracks:
        if isinstance(item, TrackOutput):
            tid, bb = item.track_id, item.bbox
        else:
            tid, bb = item
        score = iou(bb, anchor)
        if score > best_iou:
            best_iou, best = score, (tid, bb)
    return best


def mask_anchor_detections(
    dets: list[Detection], anchor: BBox | None, thresh: float = 0.25
) -> list[Detection]:
    if anchor is None:
        return dets
    return [d for d in dets if iou(d.bbox, anchor) < thresh]


def lock_track_ids_from_gt(
    track_maps: dict[int, list[tuple[int, BBox]]],
    anchor_path: dict[int, BBox | None],
    until_frame: int,
    min_iou: float = 0.32,
) -> int | None:
    """Vote track id that best matches GT anchor before occlusion."""
    votes: dict[int, float] = {}
    for i in range(until_frame):
        anchor = anchor_path.get(i)
        if anchor is None:
            continue
        for tid, bb in track_maps.get(i, []):
            score = iou(bb, anchor)
            if score >= min_iou:
                votes[tid] = votes.get(tid, 0.0) + score
    if not votes:
        return None
    return max(votes.items(), key=lambda x: x[1])[0]


def find_occlusion_by_gt_dropout(
    raw: dict[int, list[Detection]],
    path: dict[int, BBox | None],
    n: int,
    min_len: int = 10,
) -> tuple[int, int]:
    """Occlusion when YOLO misses the GT anchor person."""
    blocked = []
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            blocked.append(False)
            continue
        hit = any(iou(anchor, d.bbox) > 0.22 for d in raw[i] if d.class_id == 0)
        fg = any(
            iou(anchor, d.bbox) > 0.04 and d.bbox.area > anchor.area * 1.15
            for d in raw[i]
            if d.class_id == 0
        )
        blocked.append(fg and not hit)
    best_s, best_len = 0, 0
    rs, rl = None, 0
    for i, flag in enumerate(blocked):
        if flag:
            rs, rl = (i, 1) if rs is None else (rs, rl + 1)
        elif rs is not None:
            if rl > best_len:
                best_s, best_len = rs, rl
            rs, rl = None, 0
    if rs is not None and rl > best_len:
        best_s, best_len = rs, rl
    if best_len >= min_len:
        pad = max(8, best_len // 3)
        return max(10, best_s - pad), min(n - 10, best_s + best_len + pad)
    return find_occlusion_by_foreground(raw, path, n, min_len)


def find_occlusion_by_foreground(
    raw: dict[int, list[Detection]],
    path: dict[int, BBox | None],
    n: int,
    min_len: int = 12,
) -> tuple[int, int]:
    """Frames where a larger detection overlaps the subject anchor."""
    blocked = []
    for i in range(n):
        anchor = path.get(i)
        if anchor is None:
            blocked.append(False)
            continue
        blocked.append(
            any(
                iou(anchor, d.bbox) > 0.05 and d.bbox.area > anchor.area * 1.2
                for d in raw[i]
            )
        )
    best_s, best_len, rs, rl = 0, 0, None, 0
    for i, flag in enumerate(blocked):
        if flag:
            rs, rl = (i, 1) if rs is None else (rs, rl + 1)
        elif rs is not None:
            if rl > best_len:
                best_s, best_len = rs, rl
            rs, rl = None, 0
    if rs is not None and rl > best_len:
        best_s, best_len = rs, rl
    if best_len >= min_len:
        pad = max(6, best_len // 4)
        s = max(12, best_s - pad)
        e = min(n - 12, best_s + best_len + pad)
        if e - s >= 20:
            return s, e
    return n // 3, 2 * n // 3


def mask_subject_window(
    dets_map: dict[int, list[Detection]],
    path: dict[int, BBox | None],
    oc_start: int,
    oc_end: int,
    thresh: float = 0.22,
) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    for i, dets in dets_map.items():
        if oc_start <= i < oc_end:
            anchor = path.get(i)
            out[i] = mask_anchor_detections(dets, anchor, thresh)
        else:
            out[i] = dets
    return out


@dataclass
class LumenVisualState:
    """Extra overlays for the PERSIST-AI panel during ghost tracking."""

    exit_zones: list[tuple[BBox, float]] = field(default_factory=list)
    confidence: float = 1.0
    predicted_path: list[tuple[float, float]] = field(default_factory=list)
    latent_badge: str = ""


@dataclass
class FrameComparison:
    baseline_bbox: BBox | None
    baseline_lost: bool
    baseline_new_id: bool
    baseline_track_id: int | None
    lumen_bbox: BBox | None
    lumen_ghost: bool
    lumen_track_id: int | None
    in_occlusion: bool
    lumen_visual: LumenVisualState = field(default_factory=LumenVisualState)


@dataclass
class DemoComparisonEngine:
    """Runs masked baseline + PERSIST-AI and locks one subject via anchor path."""

    cfg: dict
    target_classes: list[int] = field(default_factory=lambda: [0])
    anchor_path: dict[int, BBox | None] = field(default_factory=dict)
    oc_start: int = 0
    oc_end: int = 0
    occlusion_windows: list[tuple[int, int]] = field(default_factory=list)
    lock_until_frame: int = 0

    locked_baseline_id: int | None = None
    locked_lumen_id: int | None = None
    _baseline_map: dict[int, list[tuple[int, BBox]]] = field(default_factory=dict, init=False)
    _lumen: TrackManager | None = field(default=None, init=False)
    _passed_occlusion: bool = field(default=False, init=False)

    def build(
        self,
        masked_dets: dict[int, list[Detection]],
        vehicle_dets: dict[int, list[Detection]] | None = None,
        raw_dets: dict[int, list[Detection]] | None = None,
    ) -> None:
        raw_dets = raw_dets or masked_dets
        self._baseline_map = BaselineTracker(self.cfg, "bytetrack").track_from_detections_multi(
            masked_dets, self.target_classes
        )
        raw_map = BaselineTracker(self.cfg, "bytetrack").track_from_detections_multi(
            raw_dets, self.target_classes
        )
        limit = self.lock_until_frame or self.oc_start
        self.locked_baseline_id = lock_track_ids_from_gt(raw_map, self.anchor_path, limit)
        self._lumen = TrackManager(self.cfg)
        self._lock_lumen_id(masked_dets, vehicle_dets, limit)
        if self.locked_baseline_id is None:
            self.locked_baseline_id = lock_track_ids_from_gt(
                self._baseline_map, self.anchor_path, limit
            )
        self._lumen = TrackManager(self.cfg)

    def _lock_lumen_id(
        self,
        masked_dets: dict[int, list[Detection]],
        vehicle_dets: dict[int, list[Detection]] | None,
        limit: int,
    ) -> None:
        assert self._lumen is not None
        votes: dict[int, float] = {}
        for i in sorted(masked_dets.keys()):
            if i >= limit:
                break
            anchor = self.anchor_path.get(i)
            if anchor is None:
                continue
            v_dets = (vehicle_dets or {}).get(i, [])
            l_out = self._lumen.update(masked_dets[i], v_dets)
            for t in l_out:
                score = iou(t.bbox, anchor)
                if score >= 0.28:
                    votes[t.track_id] = votes.get(t.track_id, 0.0) + score
        if votes:
            self.locked_lumen_id = max(votes.items(), key=lambda x: x[1])[0]

    def step(
        self,
        frame_idx: int,
        masked_dets: list[Detection],
        vehicle_dets: list[Detection] | None = None,
    ) -> FrameComparison:
        assert self._lumen is not None
        anchor = self.anchor_path.get(frame_idx)
        from lumen.pipelines.persist_occlusion import frame_is_persist_latent

        in_oc = frame_is_persist_latent(
            frame_idx, anchor, masked_dets, self.occlusion_windows
        )

        l_out = self._lumen.update(masked_dets, vehicle_dets or [])
        b_raw = self._baseline_map.get(frame_idx, [])

        lumen_t: TrackOutput | None = None
        if self.locked_lumen_id is not None:
            for t in l_out:
                if t.track_id == self.locked_lumen_id:
                    lumen_t = t
                    break
        if lumen_t is None and anchor is not None and self.locked_lumen_id is not None:
            for t in l_out:
                if t.track_id == self.locked_lumen_id:
                    lumen_t = t
                    break

        base_match: tuple[int, BBox] | None = None
        if self.locked_baseline_id is not None:
            for tid, bb in b_raw:
                if tid == self.locked_baseline_id:
                    base_match = (tid, bb)
                    break
        if base_match is None and anchor is not None and not in_oc:
            m = match_track_to_anchor(b_raw, anchor, 0.28)
            if m:
                base_match = m

        baseline_has = base_match is not None
        baseline_lost = not baseline_has
        lumen_ghost = bool(lumen_t and lumen_t.is_ghost) or in_oc
        lumen_bb: BBox | None = None
        lumen_visual = LumenVisualState()

        if in_oc:
            self._passed_occlusion = True
            baseline_has = False
            baseline_lost = True
            base_bb = None
            if lumen_t is not None:
                lumen_bb = lumen_t.bbox
                lumen_visual = LumenVisualState(
                    exit_zones=list(lumen_t.exit_zones),
                    confidence=lumen_t.confidence,
                    predicted_path=list(lumen_t.predicted_path),
                    latent_badge="GHOST" if lumen_t.is_ghost else "LATENT",
                )
            elif anchor is not None:
                lumen_bb = anchor
                lumen_visual = LumenVisualState(confidence=max(0.25, 1.0 - (frame_idx - self.oc_start) * 0.04))

        baseline_new = False
        if self._passed_occlusion and not in_oc:
            if (
                baseline_has
                and self.locked_baseline_id is not None
                and base_match[0] != self.locked_baseline_id
            ):
                baseline_new = True
            elif not baseline_has and anchor is not None:
                m = match_track_to_anchor(b_raw, anchor, 0.12)
                if m and self.locked_baseline_id and m[0] != self.locked_baseline_id:
                    baseline_new = True
                    base_match = m
                    baseline_has = True
                    baseline_lost = False

        base_bb = base_match[1] if baseline_has else None
        if not in_oc:
            if lumen_t is not None:
                lumen_bb = lumen_t.bbox
            elif baseline_has:
                lumen_bb = base_bb
            else:
                lumen_bb = None

        return FrameComparison(
            baseline_bbox=base_bb,
            baseline_lost=baseline_lost,
            baseline_new_id=baseline_new,
            baseline_track_id=base_match[0] if base_match else None,
            lumen_bbox=lumen_bb,
            lumen_ghost=lumen_ghost,
            lumen_track_id=lumen_t.track_id if lumen_t else self.locked_lumen_id,
            in_occlusion=in_oc,
            lumen_visual=lumen_visual,
        )
