from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lumen.core.track_manager import TrackManager
from lumen.detector.yolo_ultralytics import YOLODetector
from lumen.viz.overlay_renderer import render_frame


class LumenPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.detector = YOLODetector(config)
        self.manager = TrackManager(config)

    def run_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
        detections_cache: dict | None = None,
    ) -> dict[int, list]:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        all_outputs: dict[int, list] = {}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            if detections_cache and idx in detections_cache:
                dets = detections_cache[idx]
            else:
                dets = self.detector.detect_frame(frame)
            outputs = self.manager.update(dets)
            all_outputs[idx] = outputs
            if writer:
                vis = render_frame(frame, outputs, title="PERSIST-AI")
                writer.write(vis)
            idx += 1

        cap.release()
        if writer:
            writer.release()
        return all_outputs
