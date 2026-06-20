"""Stable, minimal compositor for VIDEO 2 — matches Video 1 clarity."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from lumen.types import BBox, TrackOutput


class RealBeat(str, Enum):
    VISIBLE = "Both trackers see the person."
    OCCLUDED = "Person hidden - baseline forgets, PERSIST-AI keeps ghost."
    RETURN = "Person back - baseline NEW id, PERSIST-AI kept id 1."


BEAT_COLOR = {
    RealBeat.VISIBLE: (0, 220, 0),
    RealBeat.OCCLUDED: (0, 255, 255),
    RealBeat.RETURN: (0, 255, 128),
}


class StableBeat:
    """Hold each caption for several frames so nothing flashes."""

    def __init__(self, hold_frames: int = 10):
        self.hold = hold_frames
        self.current = RealBeat.VISIBLE
        self.pending: RealBeat | None = None
        self.counter = 0

    def update(self, proposed: RealBeat) -> RealBeat:
        # Occlusion is the core demo beat — show it immediately, no hold delay.
        if proposed == RealBeat.OCCLUDED and proposed != self.current:
            self.current = proposed
            self.pending = None
            self.counter = 0
            return self.current
        # Leaving occlusion (subject exit / reappear) should not linger on OCCLUDED caption.
        if self.current == RealBeat.OCCLUDED and proposed != RealBeat.OCCLUDED:
            self.current = proposed
            self.pending = None
            self.counter = 0
            return self.current
        if proposed == self.current:
            self.pending = None
            self.counter = 0
            return self.current
        if proposed != self.pending:
            self.pending = proposed
            self.counter = 1
        else:
            self.counter += 1
        if self.counter >= self.hold:
            self.current = self.pending
            self.pending = None
            self.counter = 0
        return self.current


def _draw_box(vis: np.ndarray, bb: BBox, color: tuple[int, int, int], label: str, dashed: bool = False) -> None:
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


def compose_real_intro(frame: np.ndarray, title: str, sub: str) -> np.ndarray:
    vis = frame.copy()
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
    cv2.putText(vis, title, (40, vis.shape[0] // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(vis, sub, (40, vis.shape[0] // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    return vis


def _draw_ghost_trail(vis: np.ndarray, trail: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(trail) < 2:
        return
    for i in range(len(trail) - 1):
        cv2.line(vis, trail[i], trail[i + 1], color, 2, cv2.LINE_AA)
    for pt in trail[::2]:
        cv2.circle(vis, pt, 3, color, -1, cv2.LINE_AA)


def compose_real_frame(
    frame: np.ndarray,
    beat: RealBeat,
    baseline_bbox: BBox | None = None,
    baseline_lost: bool = False,
    baseline_new_id: bool = False,
    lumen_bbox: BBox | None = None,
    lumen_ghost: bool = False,
    ghost_trail: list[tuple[int, int]] | None = None,
    frame_idx: int = 0,
    total_frames: int = 0,
    header: str = "VIDEO 2 - REAL WORLD | MOT17 mall camera",
    beat_label: str | None = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    gap = 8
    footer_h = 72

    left = frame.copy()
    right = frame.copy()

    if baseline_bbox is not None:
        label = "ID 2 (NEW)" if baseline_new_id else "ID 1"
        color = (0, 0, 255) if baseline_new_id else (255, 160, 0)
        _draw_box(left, baseline_bbox, color, label)
    elif baseline_lost:
        overlay = left.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 80), -1)
        cv2.addWeighted(overlay, 0.25, left, 0.75, 0, left)
        cv2.putText(left, "NO TRACK", (w // 2 - 90, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.putText(left, "track deleted", (w // 2 - 95, h // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)

    if lumen_bbox is not None:
        if lumen_ghost:
            if ghost_trail:
                _draw_ghost_trail(right, ghost_trail, (0, 200, 255))
            _draw_box(right, lumen_bbox, (0, 255, 255), "ID 1 (ghost)", dashed=True)
        else:
            _draw_box(right, lumen_bbox, (0, 220, 0), "ID 1")

    for vis, label in ((left, "BASELINE"), (right, "PERSIST-AI")):
        cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    canvas = np.zeros((h + footer_h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 22)
    canvas[0:h, 0:w] = left
    canvas[0:h, w + gap : w + gap + w] = right

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(canvas, header, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if total_frames:
        cv2.putText(canvas, f"Frame {frame_idx + 1}/{total_frames}", (canvas.shape[1] - 170, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    color = BEAT_COLOR.get(beat, (200, 200, 200))
    cv2.rectangle(canvas, (0, h), (canvas.shape[1], h + footer_h), (0, 0, 0), -1)
    cv2.putText(canvas, beat_label or beat.value, (16, h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    return canvas


class SmoothBBox:
    """EMA-smooth ghost boxes so they do not jump frame-to-frame."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.value: BBox | None = None

    def update(self, bb: BBox) -> BBox:
        if self.value is None:
            self.value = bb
            return bb
        a = self.alpha
        x1 = a * bb.x1 + (1 - a) * self.value.x1
        y1 = a * bb.y1 + (1 - a) * self.value.y1
        x2 = a * bb.x2 + (1 - a) * self.value.x2
        y2 = a * bb.y2 + (1 - a) * self.value.y2
        self.value = BBox(x1, y1, x2, y2)
        return self.value

    def reset(self, bb: BBox) -> BBox:
        self.value = bb
        return bb


class StickyFlag:
    """Avoid NO TRACK / ghost labels flickering for a single frame."""

    def __init__(self, on_frames: int = 4, off_frames: int = 6):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.state = False
        self.run = 0

    def update(self, raw: bool) -> bool:
        if raw:
            self.run = self.run + 1 if self.state else 1
            if not self.state and self.run >= self.on_frames:
                self.state = True
                self.run = 0
        else:
            self.run = self.run + 1 if not self.state else 1
            if self.state and self.run >= self.off_frames:
                self.state = False
                self.run = 0
        return self.state
