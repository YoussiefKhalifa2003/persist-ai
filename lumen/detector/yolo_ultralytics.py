from __future__ import annotations

from pathlib import Path

import numpy as np

from lumen.types import BBox, Detection


class YOLODetector:
    """Ultralytics YOLOv8 detection wrapper with optional embedding stub."""

    def __init__(self, config: dict):
        det_cfg = config.get("detector", {})
        self.model_name = det_cfg.get("model", "yolov8m.pt")
        self.imgsz = det_cfg.get("imgsz", 1280)
        self.conf = det_cfg.get("conf", 0.25)
        self.iou = det_cfg.get("iou", 0.45)
        self.classes = det_cfg.get("classes")
        self.half = det_cfg.get("half", False)
        self.device = config.get("device", "cuda:0")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
            device=self.device,
            verbose=False,
        )[0]
        dets: list[Detection] = []
        if results.boxes is None:
            return dets
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            emb = self._simple_embedding(frame, BBox.from_xyxy(xyxy))
            dets.append(
                Detection(
                    bbox=BBox.from_xyxy(xyxy),
                    confidence=conf,
                    class_id=cls_id,
                    embedding=emb,
                )
            )
        return dets

    def detect_video(
        self, video_path: str | Path, cache_path: str | Path | None = None
    ) -> dict[int, list[Detection]]:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        all_dets: dict[int, list[Detection]] = {}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_dets[idx] = self.detect_frame(frame)
            idx += 1
        cap.release()

        if cache_path:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                str(k): [
                    {
                        "bbox": d.bbox.as_xyxy(),
                        "conf": d.confidence,
                        "cls": d.class_id,
                        "emb": d.embedding,
                    }
                    for d in v
                ]
                for k, v in all_dets.items()
            }
            np.savez_compressed(cache_path, detections=serializable)

        return all_dets

    @staticmethod
    def _simple_embedding(frame: np.ndarray, bbox: BBox) -> list[float]:
        """Lightweight appearance stub when ReID model unavailable."""
        import cv2

        h, w = frame.shape[:2]
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        if x2 <= x1 or y2 <= y1:
            return [0.0] * 32
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return [0.0] * 32
        small = cv2.resize(crop, (32, 64))
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist[:32].tolist()
