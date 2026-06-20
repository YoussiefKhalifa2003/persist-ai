from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from lumen.detector.yolo_ultralytics import YOLODetector
from lumen.types import BBox, Detection


class BaselineTracker:
    """Ultralytics built-in ByteTrack/BoT-SORT wrapper."""

    TRACKER_MAP = {
        "bytetrack": "bytetrack.yaml",
        "botsort": "botsort.yaml",
        "occluboost": "botsort.yaml",  # fallback; use boxmot if installed
    }

    def __init__(self, config: dict, method: str = "bytetrack"):
        self.config = config
        self.method = method
        self.detector = YOLODetector(config)
        self.tracker_yaml = self.TRACKER_MAP.get(method, "bytetrack.yaml")

    def track_video(
        self, video_path: str | Path, save_txt: Path | None = None, max_frames: int | None = None
    ) -> dict[int, list[tuple[int, BBox]]]:
        """Returns frame_idx -> list of (track_id, bbox)."""
        import cv2

        model = self.detector.model
        cap = cv2.VideoCapture(str(video_path))
        tracks: dict[int, list[tuple[int, BBox]]] = {}
        frame_idx = 0
        while True:
            if max_frames and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            results = model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_yaml,
                conf=self.detector.conf,
                iou=self.detector.iou,
                classes=self.detector.classes,
                device=self.config.get("device", "cpu"),
                verbose=False,
            )
            r = results[0]
            frame_tracks: list[tuple[int, BBox]] = []
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid in zip(r.boxes, r.boxes.id):
                    tid_int = int(tid)
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    frame_tracks.append((tid_int, BBox.from_xyxy(xyxy)))
            tracks[frame_idx] = frame_tracks
            frame_idx += 1
        cap.release()
        return tracks

    def track_from_detections(
        self,
        frames: list[np.ndarray],
        detections: dict[int, list[Detection]],
        target_classes: list[int] | None = None,
        max_gap: int = 8,
    ) -> dict[int, list[tuple[int, BBox]]]:
        """Simple IoU-based greedy tracker for fair comparison on cached dets."""
        classes = set(target_classes if target_classes is not None else self.detector.classes or [0])
        tracks: dict[int, list[tuple[int, BBox]]] = {}
        active: dict[int, BBox] = {}
        next_id = 1

        for fidx in sorted(detections.keys()):
            dets = [d for d in detections[fidx] if d.class_id in classes]
            matched: dict[int, BBox] = {}
            used = set()

            for tid, prev in list(active.items()):
                best, best_iou, best_j = None, 0.0, -1
                for j, det in enumerate(dets):
                    if j in used:
                        continue
                    score = self._iou(prev, det.bbox)
                    if score > best_iou:
                        best_iou, best, best_j = score, det.bbox, j
                if best is not None and best_iou > 0.2:
                    matched[tid] = best
                    used.add(best_j)

            for j, det in enumerate(dets):
                if j in used:
                    continue
                matched[next_id] = det.bbox
                next_id += 1

            active = matched
            tracks[fidx] = [(tid, bb) for tid, bb in active.items()]

        return tracks

    def track_from_detections_multi(
        self,
        detections: dict[int, list[Detection]],
        target_classes: list[int] | None = None,
    ) -> dict[int, list[tuple[int, BBox]]]:
        """Cached-detection tracking without frame images."""
        return self.track_from_detections([], detections, target_classes=target_classes)

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0
