"""Find MOT17 clip where a GT person has YOLO detection dropouts (occlusion proxy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from lumen.data.mot17_parser import parse_mot17_gt
from lumen.detector.yolo_ultralytics import YOLODetector
from lumen.types import BBox


@dataclass
class ClipCandidate:
    seq_dir: Path
    gt_track_id: int
    start_frame: int  # 1-based MOT
    num_frames: int
    gap_frames: int
    score: float


def _iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def gt_anchor_map(gt_path: Path, track_id: int, start: int, end: int, scale: float) -> dict[int, BBox]:
    gt = parse_mot17_gt(gt_path)
    out: dict[int, BBox] = {}
    for f in range(start, end + 1):
        fgt = gt.get(f)
        if not fgt:
            continue
        for o in fgt.objects:
            if o["track_id"] != track_id:
                continue
            b = o["bbox"]
            out[f] = BBox(b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale)
    return out


def scan_track_dropouts(
    seq_dir: Path,
    detector: YOLODetector,
    track_id: int,
    frame_start: int,
    frame_end: int,
    scale: float = 0.5,
    iou_thresh: float = 0.15,
) -> tuple[list[bool], list[BBox | None]]:
    """For each frame, did YOLO detect the GT person?"""
    gt_path = seq_dir / "gt" / "gt.txt"
    anchors = gt_anchor_map(gt_path, track_id, frame_start, frame_end, scale)
    img_dir = seq_dir / "img1"
    imgs = sorted(img_dir.glob("*.jpg"))

    detected: list[bool] = []
    anchor_list: list[BBox | None] = []
    for f in range(frame_start, frame_end + 1):
        idx = f - 1
        anchor = anchors.get(f)
        anchor_list.append(anchor)
        if anchor is None or idx >= len(imgs):
            detected.append(False)
            continue
        img = cv2.imread(str(imgs[idx]))
        img = cv2.resize(img, None, fx=scale, fy=scale)
        dets = [d for d in detector.detect_frame(img) if d.class_id == 0]
        hit = any(_iou(anchor, d.bbox) >= iou_thresh for d in dets)
        detected.append(hit)
    return detected, anchor_list


def find_best_clip(
    seq_dir: Path,
    cfg: dict,
    scan_start: int = 100,
    scan_end: int = 220,
    clip_len: int = 70,
    scale: float = 0.5,
) -> ClipCandidate:
    gt_path = seq_dir / "gt" / "gt.txt"
    gt = parse_mot17_gt(gt_path)
    track_counts: dict[int, int] = {}
    for f in range(scan_start, scan_end + 1):
        fgt = gt.get(f)
        if not fgt:
            continue
        for o in fgt.objects:
            cx = o["bbox"].cx * scale
            cy = o["bbox"].cy * scale
            if 300 < cx < 650 and 120 < cy < 420:
                track_counts[o["track_id"]] = track_counts.get(o["track_id"], 0) + 1

    candidates = sorted(track_counts.items(), key=lambda x: -x[1])[:6]
    detector = YOLODetector(cfg)
    best: ClipCandidate | None = None

    for tid, _ in candidates:
        detected, anchors = scan_track_dropouts(seq_dir, detector, tid, scan_start, scan_end, scale)
        # sliding window: maximize consecutive miss run inside clip
        for win_start in range(0, len(detected) - clip_len):
            window = detected[win_start : win_start + clip_len]
            miss_run = 0
            best_miss = 0
            for hit in window:
                if not hit:
                    miss_run += 1
                    best_miss = max(best_miss, miss_run)
                else:
                    miss_run = 0
            before = sum(window[: clip_len // 3])
            after = sum(window[2 * clip_len // 3 :])
            if best_miss < 4 or before < 5 or after < 5:
                continue
            score = best_miss + 0.1 * (before + after)
            start_mot = scan_start + win_start
            cand = ClipCandidate(seq_dir, tid, start_mot, clip_len, best_miss, score)
            if best is None or cand.score > best.score:
                best = cand

    if best is None:
        return ClipCandidate(seq_dir, 17, 128, clip_len, 0, 0.0)
    return best
