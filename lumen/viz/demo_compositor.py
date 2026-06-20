from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from lumen.types import BBox, TrackOutput, TrackState


class DemoPhase(str, Enum):
    VISIBLE = "VISIBLE - person detected by camera"
    OCCLUDED = "OCCLUDED - person hidden behind object"
    BASELINE_LOST = "BASELINE FORGOT THEM - ID deleted"
    PERSIST_GHOST = "PERSIST-AI KEEPS TRACK - ghost prediction active"
    REAPPEARED = "REAPPEARED - person visible again"


PHASE_STEP = {
    DemoPhase.VISIBLE: (1, "STEP 1: Normal tracking"),
    DemoPhase.OCCLUDED: (2, "STEP 2: Person disappears from view"),
    DemoPhase.BASELINE_LOST: (3, "STEP 3: Baseline deletes the track"),
    DemoPhase.PERSIST_GHOST: (3, "STEP 3: PERSIST-AI maintains ghost state"),
    DemoPhase.REAPPEARED: (4, "STEP 4: Person returns - PERSIST-AI was right"),
}


def _glow_box(vis: np.ndarray, pt1, pt2, color, thickness=4):
    for t in range(thickness, 0, -1):
        c = tuple(int(v * (0.4 + 0.15 * t)) for v in color)
        cv2.rectangle(vis, pt1, pt2, c, t)


def _draw_box(
    vis: np.ndarray,
    bb: BBox,
    color: tuple[int, int, int],
    label: str,
    dashed: bool = False,
    thickness: int = 3,
    glow: bool = False,
) -> None:
    pt1 = (int(bb.x1), int(bb.y1))
    pt2 = (int(bb.x2), int(bb.y2))
    if glow:
        _glow_box(vis, pt1, pt2, color, thickness + 3)
    if dashed:
        for x in range(pt1[0], pt2[0], 14):
            cv2.line(vis, (x, pt1[1]), (min(x + 7, pt2[0]), pt1[1]), color, thickness)
            cv2.line(vis, (x, pt2[1]), (min(x + 7, pt2[0]), pt2[1]), color, thickness)
        for y in range(pt1[1], pt2[1], 14):
            cv2.line(vis, (pt1[0], y), (pt1[0], min(y + 7, pt2[1])), color, thickness)
            cv2.line(vis, (pt2[0], y), (pt2[0], min(y + 7, pt2[1])), color, thickness)
    else:
        cv2.rectangle(vis, pt1, pt2, color, thickness)
    tw = max(140, len(label) * 10)
    cv2.rectangle(vis, (pt1[0], pt1[1] - 26), (pt1[0] + tw, pt1[1]), color, -1)
    cv2.putText(vis, label, (pt1[0] + 5, pt1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def render_panel(
    frame: np.ndarray,
    tracks: list[TrackOutput],
    title: str,
    highlight_id: int | None,
    baseline_lost: bool = False,
    show_all: bool = False,
    occluder: BBox | None = None,
) -> np.ndarray:
    vis = frame.copy()

    if occluder is not None:
        overlay = vis.copy()
        cv2.rectangle(
            overlay,
            (int(occluder.x1), int(occluder.y1)),
            (int(occluder.x2), int(occluder.y2)),
            (40, 40, 120),
            -1,
        )
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
        cv2.rectangle(
            vis,
            (int(occluder.x1), int(occluder.y1)),
            (int(occluder.x2), int(occluder.y2)),
            (80, 80, 200),
            2,
        )
        cv2.putText(
            vis, "OCCLUDER", (int(occluder.x1) + 5, int(occluder.y1) + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2,
        )

    for t in tracks:
        is_hi = highlight_id is None or t.track_id == highlight_id
        if not show_all and not is_hi:
            continue
        if t.is_ghost and is_hi:
            _draw_box(vis, t.bbox, (0, 255, 255), f"GHOST ID {t.track_id}", dashed=True, glow=True)
            for zone, _ in t.exit_zones:
                overlay = vis.copy()
                cv2.rectangle(
                    overlay,
                    (int(zone.x1), int(zone.y1)),
                    (int(zone.x2), int(zone.y2)),
                    (0, 255, 255),
                    -1,
                )
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
                cv2.putText(
                    vis, "EXIT ZONE", (int(zone.x1) + 4, int(zone.y1) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                )
            if t.predicted_path and len(t.predicted_path) > 1:
                pts = np.array(t.predicted_path, dtype=np.int32)
                for i in range(len(pts) - 1):
                    cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (255, 180, 0), 3)
        elif not t.is_ghost:
            color = (0, 255, 80) if "PERSIST-AI" in title else (255, 140, 0)
            if not is_hi:
                color = tuple(int(c * 0.4) for c in color)
            label = f"PERSON ID {t.track_id}" if is_hi else f"id{t.track_id}"
            _draw_box(vis, t.bbox, color, label, thickness=3 if is_hi else 1, glow=is_hi)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 52), (15, 15, 15), -1)
    cv2.putText(vis, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    if baseline_lost:
        cv2.rectangle(vis, (vis.shape[1] - 160, 8), (vis.shape[1] - 8, 44), (0, 0, 180), -1)
        cv2.putText(vis, "ID LOST", (vis.shape[1] - 145, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return vis


def infer_phase(
    person_detected: bool,
    lumen_ghost: bool,
    baseline_has_id: bool,
    lumen_has_id: bool,
) -> DemoPhase:
    if lumen_ghost and not baseline_has_id:
        return DemoPhase.PERSIST_GHOST
    if not person_detected and not baseline_has_id and not lumen_has_id:
        return DemoPhase.BASELINE_LOST
    if not person_detected and lumen_has_id:
        return DemoPhase.PERSIST_GHOST
    if not person_detected:
        return DemoPhase.OCCLUDED
    if person_detected and not baseline_has_id and lumen_has_id:
        return DemoPhase.REAPPEARED
    if person_detected:
        return DemoPhase.VISIBLE
    return DemoPhase.VISIBLE


def compose_demo_frame(
    frame: np.ndarray,
    baseline_tracks: list[TrackOutput],
    lumen_tracks: list[TrackOutput],
    highlight_id: int | None,
    frame_idx: int,
    total_frames: int,
    phase: DemoPhase,
    baseline_lost: bool,
    header: str = "PERSIST-AI Demo - Real street footage",
    show_all: bool = False,
    occluder: BBox | None = None,
    baseline_highlight_id: int | None = None,
) -> np.ndarray:
    base_hi = baseline_highlight_id if baseline_highlight_id is not None else highlight_id
    left = render_panel(
        frame, baseline_tracks, "BASELINE (normal tracker)", base_hi,
        baseline_lost=baseline_lost, show_all=show_all, occluder=occluder,
    )
    right = render_panel(
        frame, lumen_tracks, "PERSIST-AI (object permanence)", highlight_id,
        show_all=show_all, occluder=occluder,
    )

    gap = 10
    h, w = frame.shape[:2]
    canvas = np.zeros((h + 120, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)
    canvas[60 : 60 + h, 0:w] = left
    canvas[60 : 60 + h, w + gap : w + gap + w] = right

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(canvas, header, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(
        canvas, f"Frame {frame_idx + 1}/{total_frames}",
        (canvas.shape[1] - 200, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2,
    )

    step_num, step_label = PHASE_STEP.get(phase, (0, ""))
    phase_color = {
        DemoPhase.VISIBLE: (0, 255, 0),
        DemoPhase.OCCLUDED: (0, 200, 255),
        DemoPhase.BASELINE_LOST: (0, 0, 255),
        DemoPhase.PERSIST_GHOST: (0, 255, 255),
        DemoPhase.REAPPEARED: (0, 255, 128),
    }[phase]

    cv2.rectangle(canvas, (0, h + 60), (canvas.shape[1], h + 120), (0, 0, 0), -1)
    if step_num:
        cv2.putText(canvas, step_label, (14, h + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(canvas, phase.value, (14, h + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2)
    return canvas


def compose_concept_intro(frame: np.ndarray, text: str, sub: str) -> np.ndarray:
    vis = frame.copy()
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
    cv2.putText(vis, text, (40, vis.shape[0] // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(vis, sub, (40, vis.shape[0] // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis
