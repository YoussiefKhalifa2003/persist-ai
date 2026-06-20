"""Render side-by-side comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lumen.detector.yolo_ultralytics import YOLODetector
from lumen.core.track_manager import TrackManager
from lumen.trackers.baseline_adapter import BaselineTracker
from lumen.types import TrackOutput, TrackState
from lumen.utils.io import load_config
from lumen.viz.side_by_side import render_side_by_side


def _apply_cfg(cfg: dict) -> dict:
    import torch

    if cfg.get("device", "cuda:0").startswith("cuda") and not torch.cuda.is_available():
        cfg["device"] = "cpu"
        cfg.setdefault("detector", {})["half"] = False
        cfg["detector"]["model"] = "yolov8n.pt"
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw/synthetic/occlusion_demo.mp4")
    parser.add_argument("--output", default="results/demo_videos/hero_01_side_by_side.mp4")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = _apply_cfg(load_config(args.config))
    video = Path(args.video)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(cfg)
    baseline = BaselineTracker(cfg, "bytetrack")
    lumen_mgr = TrackManager(cfg)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h + 40))

    all_dets: dict = {}
    frames_list: list = []
    idx = 0
    while idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames_list.append(frame)
        all_dets[idx] = detector.detect_frame(frame)
        idx += 1
    cap.release()

    baseline_tracks = baseline.track_from_detections(frames_list, all_dets)

    lumen_mgr = TrackManager(cfg)
    idx = 0
    for frame in frames_list:
        l_out = lumen_mgr.update(all_dets[idx])
        b_out = [
            TrackOutput(t, b, TrackState.ACTIVE, 1.0)
            for t, b in baseline_tracks.get(idx, [])
        ]
        canvas = render_side_by_side(frame, b_out, l_out, clip_name=video.stem, frame_idx=idx)
        writer.write(canvas)
        idx += 1

    writer.release()

    failure = out.with_name("failure_case_01.mp4")
    import shutil

    shutil.copy(out, failure)
    print(f"Wrote {out}")
    print(f"Wrote {failure}")


if __name__ == "__main__":
    main()
