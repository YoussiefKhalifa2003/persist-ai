from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lumen.viz.overlay_renderer import render_frame


def render_side_by_side(
    frame: np.ndarray,
    baseline_tracks: list,
    lumen_tracks: list,
    clip_name: str = "",
    frame_idx: int = 0,
) -> np.ndarray:
    left = render_frame(frame, baseline_tracks, title="Baseline (ByteTrack)")
    right = render_frame(frame, lumen_tracks, title="PERSIST-AI")
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.zeros((h + 40, w, 3), dtype=np.uint8)
    canvas[40 : 40 + left.shape[0], : left.shape[1]] = left
    canvas[40 : 40 + right.shape[0], left.shape[1] :] = right
    header = f"{clip_name} | frame {frame_idx}"
    cv2.putText(canvas, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas
