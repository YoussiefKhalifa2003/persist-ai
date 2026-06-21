from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lumen.detector.yolo_ultralytics import YOLODetector
from lumen.trackers.baseline_adapter import BaselineTracker
from lumen.viz.overlay_renderer import render_frame


class BaselinePipeline:
    def __init__(self, config: dict, method: str = "bytetrack"):
        self.config = config
        self.method = method
        self.tracker = BaselineTracker(config, method=method)

    def run_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
    ) -> dict[int, list[tuple[int, object]]]:
        tracks = self.tracker.track_video(video_path, max_frames=max_frames)
        if output_path:
            self._render(video_path, tracks, output_path, max_frames, label=f"Baseline ({self.method})")
        if max_frames:
            tracks = {k: v for k, v in tracks.items() if k < max_frames}
        return tracks

    def _render(self, video_path, tracks, output_path, max_frames, label):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            from lumen.types import TrackOutput, TrackState

            outputs = [
                TrackOutput(tid, bb, TrackState.ACTIVE, 1.0)
                for tid, bb in tracks.get(idx, [])
            ]
            frame = render_frame(frame, outputs, title=label)
            out.write(frame)
            idx += 1
        cap.release()
        out.release()
