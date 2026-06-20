from __future__ import annotations

import json
from pathlib import Path

from lumen.types import BBox, FrameGT


def parse_bdd_mot_label(label_path: str | Path) -> dict[int, FrameGT]:
    """Parse BDD100K box_track_20 JSON label file."""
    label_path = Path(label_path)
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_frame: dict[int, list] = {}
    frames = data.get("frames", data) if isinstance(data, dict) else data
    if isinstance(frames, dict):
        for frame_key, frame_data in frames.items():
            fidx = int(frame_key)
            objs = []
            for obj in frame_data.get("objects", []):
                if "box" not in obj:
                    continue
                x1, y1, x2, y2 = obj["box"]["x1"], obj["box"]["y1"], obj["box"]["x2"], obj["box"]["y2"]
                objs.append(
                    {
                        "track_id": obj.get("id", obj.get("track_id", -1)),
                        "bbox": BBox(x1, y1, x2, y2),
                        "category": obj.get("category", "unknown"),
                    }
                )
            by_frame[fidx] = objs
    elif isinstance(frames, list):
        for entry in frames:
            fidx = entry.get("frame", entry.get("index", 0))
            objs = []
            for obj in entry.get("objects", []):
                box = obj.get("box", obj)
                if isinstance(box, dict):
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                else:
                    continue
                objs.append(
                    {
                        "track_id": obj.get("id", -1),
                        "bbox": BBox(x1, y1, x2, y2),
                        "category": obj.get("category", "unknown"),
                    }
                )
            by_frame[fidx] = objs

    return {f: FrameGT(frame_idx=f, objects=objs) for f, objs in by_frame.items()}


def find_bdd_label_files(labels_root: str | Path) -> list[Path]:
    labels_root = Path(labels_root)
    if not labels_root.exists():
        return []
    return sorted(labels_root.rglob("*.json"))
