from __future__ import annotations

from pathlib import Path

import pandas as pd

from lumen.types import BBox, FrameGT


def parse_mot17_gt(gt_path: str | Path) -> dict[int, FrameGT]:
    """Parse MOTChallenge gt.txt: frame, id, bb_left, top, w, h, ..."""
    gt_path = Path(gt_path)
    rows = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            rows.append(
                {
                    "frame": frame,
                    "track_id": tid,
                    "bbox": BBox(x, y, x + w, y + h),
                    "category": "person",
                }
            )

    by_frame: dict[int, list] = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(
            {"track_id": r["track_id"], "bbox": r["bbox"], "category": r["category"]}
        )
    return {f: FrameGT(frame_idx=f, objects=objs) for f, objs in by_frame.items()}


def list_mot17_sequences(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("MOT17")])


def get_mot17_video_path(seq_root: str | Path) -> Path | None:
    seq_root = Path(seq_root)
    img_dir = seq_root / "img1"
    if not img_dir.exists():
        return None
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        return None
    return img_dir


def mot17_sequence_to_video(seq_root: str | Path, output: str | Path, fps: int = 30) -> Path:
    """Create MP4 from MOT17 image sequence for tracking."""
    import cv2

    img_dir = get_mot17_video_path(seq_root)
    if img_dir is None:
        raise FileNotFoundError(f"No images in {seq_root}")
    imgs = sorted((Path(img_dir)).glob("*.jpg"))
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    out = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for p in imgs:
        out.write(cv2.imread(str(p)))
    out.release()
    return Path(output)
