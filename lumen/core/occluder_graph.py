from __future__ import annotations

from lumen.types import BBox, Detection
from lumen.utils.geometry import center_inside, iou

VEHICLE_CLASSES = {2, 5, 7}  # car, bus, truck in COCO


def find_occluder(
    pedestrian_bbox: BBox,
    vehicle_detections: list[Detection],
    iou_threshold: float = 0.1,
) -> tuple[int | None, BBox | None]:
    best_iou = 0.0
    best_id = None
    best_bbox = None

    for idx, det in enumerate(vehicle_detections):
        if det.class_id not in VEHICLE_CLASSES:
            continue
        score = iou(pedestrian_bbox, det.bbox)
        inside = center_inside(pedestrian_bbox, det.bbox, margin=10)
        combined = max(score, 0.15 if inside else 0.0)
        if combined >= iou_threshold and combined > best_iou:
            best_iou = combined
            best_id = idx
            best_bbox = det.bbox

    return best_id, best_bbox
