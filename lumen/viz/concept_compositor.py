"""Minimal compositor for VIDEO 1 (building blocks) — one box, one message."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from lumen.types import BBox, TrackOutput


class ConceptBeat(str, Enum):
    INTRO = "When someone hides behind a car, normal AI forgets them."
    VISIBLE = "Both trackers see the person."
    HIDDEN = "Person hidden from camera."
    BASELINE_LOST = "Baseline: track deleted."
    PERSIST_KEEPS = "PERSIST-AI: ghost moving through occluder."
    RETURN = "Baseline got NEW id. PERSIST-AI kept id 1."
    OUTRO = "Object permanence for autonomous perception."


BEAT_COLOR = {
    ConceptBeat.VISIBLE: (0, 220, 0),
    ConceptBeat.HIDDEN: (0, 200, 255),
    ConceptBeat.BASELINE_LOST: (0, 0, 255),
    ConceptBeat.PERSIST_KEEPS: (0, 255, 255),
    ConceptBeat.RETURN: (0, 255, 128),
}


def _draw_person_box(
    vis: np.ndarray,
    bb: BBox,
    color: tuple[int, int, int],
    label: str,
    dashed: bool = False,
) -> None:
    pt1 = (int(bb.x1), int(bb.y1))
    pt2 = (int(bb.x2), int(bb.y2))
    if dashed:
        for x in range(pt1[0], pt2[0], 12):
            cv2.line(vis, (x, pt1[1]), (min(x + 6, pt2[0]), pt1[1]), color, 2)
            cv2.line(vis, (x, pt2[1]), (min(x + 6, pt2[0]), pt2[1]), color, 2)
        for y in range(pt1[1], pt2[1], 12):
            cv2.line(vis, (pt1[0], y), (pt1[0], min(y + 6, pt2[1])), color, 2)
            cv2.line(vis, (pt2[0], y), (pt2[0], min(y + 6, pt2[1])), color, 2)
    else:
        cv2.rectangle(vis, pt1, pt2, color, 2)
    cv2.putText(vis, label, (pt1[0], max(pt1[1] - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_ghost_trail(vis: np.ndarray, trail: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(trail) < 2:
        return
    for i in range(len(trail) - 1):
        cv2.line(vis, trail[i], trail[i + 1], color, 2, cv2.LINE_AA)
    for pt in trail[::2]:
        cv2.circle(vis, pt, 3, color, -1, cv2.LINE_AA)


def compose_title_card(width: int, height: int, title: str, subtitle: str = "") -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    cv2.putText(canvas, "VIDEO 1 - BUILDING BLOCKS", (width // 2 - 220, height // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
    cv2.putText(canvas, title, (max(20, width // 2 - len(title) * 7), height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    if subtitle:
        cv2.putText(canvas, subtitle, (max(20, width // 2 - len(subtitle) * 6), height // 2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    return canvas


def compose_concept_frame(
    frame: np.ndarray,
    baseline_tracks: list[TrackOutput],
    lumen_tracks: list[TrackOutput],
    beat: ConceptBeat,
    baseline_lost: bool = False,
    ghost_bbox: BBox | None = None,
    ghost_trail: list[tuple[int, int]] | None = None,
    baseline_label: str | None = None,
    lumen_label: str = "ID 1",
    lumen_is_ghost: bool = False,
) -> np.ndarray:
    h, w = frame.shape[:2]
    gap = 8
    footer_h = 72

    left = frame.copy()
    right = frame.copy()

    if baseline_tracks and baseline_label:
        t = baseline_tracks[0]
        color = (0, 0, 255) if "NEW" in baseline_label else (255, 160, 0)
        _draw_person_box(left, t.bbox, color, baseline_label)
    elif baseline_lost:
        cv2.putText(left, "NO TRACK", (w // 2 - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

    if lumen_is_ghost and ghost_bbox is not None:
        if ghost_trail:
            _draw_ghost_trail(right, ghost_trail, (0, 200, 255))
        _draw_person_box(right, ghost_bbox, (0, 255, 255), lumen_label, dashed=True)
        cx, cy = int(ghost_bbox.cx), int(ghost_bbox.cy)
        cv2.circle(right, (cx, cy), 8, (0, 255, 255), -1)
    elif lumen_tracks:
        t = lumen_tracks[0]
        _draw_person_box(right, t.bbox, (0, 220, 0), lumen_label)

    for vis, label in ((left, "BASELINE"), (right, "PERSIST-AI")):
        cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    canvas = np.zeros((h + footer_h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 22)
    canvas[0:h, 0:w] = left
    canvas[0:h, w + gap : w + gap + w] = right

    color = BEAT_COLOR.get(beat, (200, 200, 200))
    cv2.rectangle(canvas, (0, h), (canvas.shape[1], h + footer_h), (0, 0, 0), -1)
    cv2.putText(canvas, beat.value, (16, h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    return canvas
