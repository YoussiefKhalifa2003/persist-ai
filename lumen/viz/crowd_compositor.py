"""Side-by-side compositor: PERSIST-AI (left) vs raw YOLO detections (right)."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from lumen.brand import BRAND
from lumen.pipelines.comparison_pipeline import LumenVisualState
from lumen.types import BBox, Detection
from lumen.viz.real_compositor import RealBeat, BEAT_COLOR, _draw_ghost_trail
from lumen.viz.silhouette import SubjectSilhouette

GRAY = (140, 140, 140)
VEHICLE = (0, 140, 255)
from lumen.brand import BRAND

TARGET_PERSIST = (0, 220, 0)
TARGET_GHOST = (0, 255, 255)

PERSON_CLASS = 0


def _ghost_hidden_by_vehicle(bb: BBox, occluders: list[Detection] | None) -> bool:
    if not occluders:
        return False
    for d in occluders:
        if d.class_id not in {2, 5, 7} or d.bbox.area < 6000:
            continue
        if _iou(bb, d.bbox) > 0.012:
            return True
    return False


def _draw_latent_sidewalk_marker(vis: np.ndarray, bb: BBox, label: str) -> None:
    cx, cy = int(bb.cx), int(bb.cy)
    cv2.circle(vis, (cx, cy), 6, TARGET_GHOST, -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), 10, TARGET_GHOST, 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        label,
        (max(8, cx - 40), max(48, cy - bb.h // 2 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        TARGET_GHOST,
        2,
        cv2.LINE_AA,
    )


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _is_target_det(det: Detection, target_bb: BBox | None, thresh: float = 0.28) -> bool:
    return target_bb is not None and _iou(det.bbox, target_bb) >= thresh


def _draw_thin_box(vis: np.ndarray, bb: BBox, color: tuple[int, int, int], thickness: int = 1) -> None:
    pt1 = (int(bb.x1), int(bb.y1))
    pt2 = (int(bb.x2), int(bb.y2))
    cv2.rectangle(vis, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_target_box(
    vis: np.ndarray,
    bb: BBox,
    color: tuple[int, int, int],
    label: str,
    dashed: bool = False,
    thickness: int = 3,
) -> None:
    pt1 = (int(bb.x1), int(bb.y1))
    pt2 = (int(bb.x2), int(bb.y2))
    if dashed:
        for x in range(pt1[0], pt2[0], 14):
            cv2.line(vis, (x, pt1[1]), (min(x + 7, pt2[0]), pt1[1]), color, thickness)
            cv2.line(vis, (x, pt2[1]), (min(x + 7, pt2[0]), pt2[1]), color, thickness)
        for y in range(pt1[1], pt2[1], 14):
            cv2.line(vis, (pt1[0], y), (pt1[0], min(y + 7, pt2[1])), color, thickness)
            cv2.line(vis, (pt2[0], y), (pt2[0], min(y + 7, pt2[1])), color, thickness)
    else:
        cv2.rectangle(vis, pt1, pt2, color, thickness, cv2.LINE_AA)
    cv2.putText(
        vis,
        label,
        (pt1[0], max(pt1[1] - 10, 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_raw_dets(vis: np.ndarray, dets: list[Detection]) -> None:
    """Right panel: all YOLO boxes, no TARGET highlight."""
    for d in dets:
        if d.class_id == PERSON_CLASS:
            _draw_thin_box(vis, d.bbox, GRAY, 1)
        else:
            _draw_thin_box(vis, d.bbox, VEHICLE, 2)


def _draw_exit_zones(vis: np.ndarray, zones: list[tuple[BBox, float]]) -> None:
    overlay = vis.copy()
    for zone, _weight in zones:
        pt1 = (int(zone.x1), int(zone.y1))
        pt2 = (int(zone.x2), int(zone.y2))
        cv2.rectangle(overlay, pt1, pt2, (0, 220, 255), -1)
    cv2.addWeighted(overlay, 0.22, vis, 0.78, 0, vis)


def _draw_confidence_bar(vis: np.ndarray, bb: BBox, confidence: float) -> None:
    x1 = int(bb.x1)
    y = int(bb.y2) + 6
    w = int(bb.w)
    cv2.rectangle(vis, (x1, y), (x1 + w, y + 6), (60, 60, 60), -1)
    fill = int(w * max(0.0, min(confidence, 1.0)))
    cv2.rectangle(vis, (x1, y), (x1 + fill, y + 6), (0, 200, 0), -1)


def _draw_predicted_path(vis: np.ndarray, path: list[tuple[float, float]]) -> None:
    if len(path) < 2:
        return
    pts = [(int(x), int(y)) for x, y in path]
    for i in range(len(pts) - 1):
        cv2.line(vis, pts[i], pts[i + 1], (0, 180, 255), 1, cv2.LINE_AA)
        if i % 2 == 0:
            cv2.circle(vis, pts[i], 2, (0, 180, 255), -1, cv2.LINE_AA)


def _draw_latent_badge(vis: np.ndarray, bb: BBox, badge: str) -> None:
    if not badge:
        return
    cv2.putText(
        vis,
        badge,
        (int(bb.x2) - 52, int(bb.y1) + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _render_lumen_panel(
    frame: np.ndarray,
    lumen_target_bb: BBox | None,
    lumen_ghost: bool,
    target_anchor: BBox | None,
    crowd_dets: list[Detection],
    ghost_trail: list[tuple[int, int]] | None,
    silhouette: SubjectSilhouette | None,
    occluder_dets: list[Detection] | None,
    lumen_visual: LumenVisualState | None,
    target_label: str,
    ghost_label: str,
) -> np.ndarray:
    vis = frame.copy()
    ref = target_anchor or lumen_target_bb
    for d in crowd_dets:
        if not _is_target_det(d, ref):
            _draw_thin_box(vis, d.bbox, GRAY, 1)

    if occluder_dets and not lumen_ghost:
        for d in occluder_dets:
            _draw_thin_box(vis, d.bbox, VEHICLE, 2)

    if lumen_visual and lumen_visual.exit_zones and lumen_ghost:
        _draw_exit_zones(vis, lumen_visual.exit_zones)

    if lumen_target_bb is not None:
        if lumen_ghost:
            if ghost_trail:
                _draw_ghost_trail(vis, ghost_trail, (0, 200, 255))
            if silhouette is not None:
                silhouette.draw_ghost(vis, lumen_target_bb, tint=(0, 220, 255), alpha=0.45)
            if lumen_visual and lumen_visual.predicted_path:
                _draw_predicted_path(vis, lumen_visual.predicted_path)
            _draw_target_box(vis, lumen_target_bb, TARGET_GHOST, ghost_label, dashed=True, thickness=3)
            if lumen_visual:
                _draw_confidence_bar(vis, lumen_target_bb, lumen_visual.confidence)
                _draw_latent_badge(vis, lumen_target_bb, lumen_visual.latent_badge)
        else:
            _draw_target_box(vis, lumen_target_bb, TARGET_PERSIST, target_label, thickness=3)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(vis, BRAND, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 0), 2, cv2.LINE_AA)
    return vis


def _render_raw_panel(frame: np.ndarray, raw_dets: list[Detection]) -> np.ndarray:
    vis = frame.copy()
    _draw_raw_dets(vis, raw_dets)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        vis, "RAW DETECTION", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 2, cv2.LINE_AA
    )
    return vis


def _assemble_canvas(
    left: np.ndarray,
    right: np.ndarray | None,
    beat: RealBeat,
    header: str,
    beat_label: str | None,
    frame_idx: int,
    total_frames: int,
    layout: Literal["split", "raw_only", "lumen_only"],
) -> np.ndarray:
    h, w = left.shape[:2]
    gap = 8
    footer_h = 72

    if layout == "raw_only":
        canvas_w = w
        canvas = np.zeros((h + footer_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (22, 22, 22)
        canvas[0:h, 0:w] = left
    elif layout == "lumen_only":
        canvas_w = w
        canvas = np.zeros((h + footer_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (22, 22, 22)
        canvas[0:h, 0:w] = left
    else:
        assert right is not None
        canvas_w = w * 2 + gap
        canvas = np.zeros((h + footer_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (22, 22, 22)
        canvas[0:h, 0:w] = left
        canvas[0:h, w + gap : w + gap + w] = right

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(canvas, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if total_frames:
        cv2.putText(
            canvas,
            f"Frame {frame_idx + 1}/{total_frames}",
            (canvas.shape[1] - 165, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    color = BEAT_COLOR.get(beat, (200, 200, 200))
    cv2.rectangle(canvas, (0, h), (canvas.shape[1], h + footer_h), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        beat_label or beat.value,
        (14, h + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_crowd_frame(
    frame: np.ndarray,
    beat: RealBeat,
    crowd_dets: list[Detection],
    target_anchor: BBox | None,
    baseline_target_bb: BBox | None = None,
    baseline_target_lost: bool = False,
    baseline_target_new: bool = False,
    lumen_target_bb: BBox | None = None,
    lumen_ghost: bool = False,
    ghost_trail: list[tuple[int, int]] | None = None,
    silhouette: SubjectSilhouette | None = None,
    occluder_dets: list[Detection] | None = None,
    frame_idx: int = 0,
    total_frames: int = 0,
    header: str = "VIDEO 2 - REAL WORLD | MOT17 mall camera",
    beat_label: str | None = None,
    target_label: str = "TARGET",
    ghost_label: str = "TARGET (ghost)",
    raw_dets: list[Detection] | None = None,
    lumen_visual: LumenVisualState | None = None,
    layout: Literal["split", "raw_only", "lumen_only"] = "split",
) -> np.ndarray:
    """PERSIST-AI panel left, raw YOLO right. baseline_* params kept for API compat (ignored on raw)."""
    all_raw = raw_dets if raw_dets is not None else crowd_dets

    lumen_panel = _render_lumen_panel(
        frame,
        lumen_target_bb,
        lumen_ghost,
        target_anchor,
        crowd_dets,
        ghost_trail,
        silhouette,
        occluder_dets,
        lumen_visual,
        target_label,
        ghost_label,
    )
    raw_panel = _render_raw_panel(frame, all_raw)

    if layout == "raw_only":
        return _assemble_canvas(raw_panel, None, beat, header, beat_label, frame_idx, total_frames, layout)
    if layout == "lumen_only":
        return _assemble_canvas(lumen_panel, None, beat, header, beat_label, frame_idx, total_frames, layout)
    return _assemble_canvas(
        lumen_panel, raw_panel, beat, header, beat_label, frame_idx, total_frames, "split"
    )
