"""Staged demo: Act 1 raw fullscreen → Act 2 Activate PERSIST-AI → Act 3 split."""

from __future__ import annotations

import cv2
import numpy as np

from lumen.viz.crowd_compositor import compose_crowd_frame, _render_raw_panel
from lumen.viz.real_compositor import RealBeat


def compose_raw_fullscreen(
    frame: np.ndarray,
    raw_dets: list,
    beat: RealBeat,
    header: str,
    beat_label: str,
    frame_idx: int,
    total_frames: int,
    split_w: int,
) -> np.ndarray:
    """Act 1: raw detection only, scaled to split-width canvas."""
    raw = _render_raw_panel(frame, raw_dets)
    h = raw.shape[0]
    footer_h = 72
    canvas = np.zeros((h + footer_h, split_w, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 22)
    # center raw panel in wide canvas
    x_off = max(0, (split_w - raw.shape[1]) // 2)
    canvas[0:h, x_off : x_off + raw.shape[1]] = raw
    cv2.rectangle(canvas, (0, 0), (split_w, 40), (0, 0, 0), -1)
    cv2.putText(canvas, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Frame {frame_idx + 1}/{total_frames}",
        (split_w - 165, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (0, h), (split_w, h + footer_h), (0, 0, 0), -1)
    cv2.putText(canvas, beat_label, (14, h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 220, 0), 2, cv2.LINE_AA)
    return canvas


def compose_activate_button_frame(
    frame: np.ndarray,
    pulse: float,
    split_w: int,
    header: str,
) -> np.ndarray:
    """Act 2: frozen frame + pulsing Activate PERSIST-AI button."""
    h, w = frame.shape[:2]
    footer_h = 72
    canvas = np.zeros((h + footer_h, split_w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    x_off = max(0, (split_w - w) // 2)
    canvas[0:h, x_off : x_off + w] = frame

    cv2.rectangle(canvas, (0, 0), (split_w, 40), (0, 0, 0), -1)
    cv2.putText(canvas, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    cx = split_w // 2
    cy = h // 2 + 20
    scale = 1.0 + 0.08 * pulse
    bw, bh = int(280 * scale), int(56 * scale)
    bx1, by1 = cx - bw // 2, cy - bh // 2
    bx2, by2 = cx + bw // 2, cy + bh // 2
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 0), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 120), 3, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Activate PERSIST-AI",
        (cx - 118, cy + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Object permanence for real-world perception",
        (cx - 210, cy + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(canvas, (0, h), (split_w, h + footer_h), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "Press to enable persistent tracking through occlusion",
        (14, h + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_staged_split_frame(**kwargs) -> np.ndarray:
    """Act 3: delegate to split crowd compositor."""
    return compose_crowd_frame(**kwargs, layout="split")
