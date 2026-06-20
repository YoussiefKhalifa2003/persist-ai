"""Build VIDEO 2 — crowd scene, red TARGET (black suit), PERSIST-AI ghost silhouette."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from lumen.data.occlusion_finder import gt_anchor_map
from lumen.pipelines.comparison_pipeline import (
    DemoComparisonEngine,
    fill_gaps,
    find_occlusion_by_gt_dropout,
    iou,
    mask_subject_window,
)
from lumen.utils.io import load_config, save_json
from lumen.viz.crowd_compositor import compose_crowd_frame
from lumen.viz.real_compositor import RealBeat, SmoothBBox, StableBeat, StickyFlag, compose_real_intro
from lumen.viz.silhouette import SubjectSilhouette


def apply_cfg(cfg: dict) -> dict:
    import torch

    if str(cfg.get("device", "cuda:0")).startswith("cuda") and not torch.cuda.is_available():
        cfg["device"] = "cpu"
        det = cfg.setdefault("detector", {})
        det["half"] = False
        det["model"] = "yolov8n.pt"
        det["imgsz"] = 960
        det["conf"] = 0.28
        det["classes"] = [0]
    return cfg


def load_frames(seq_dir: Path, start: int, end: int, scale: float = 0.5) -> list:
    imgs = sorted((seq_dir / "img1").glob("*.jpg"))[start:end]
    out = []
    for p in imgs:
        img = cv2.imread(str(p))
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append(img)
    return out


def display_target_bbox(anchor, tracked):
    from lumen.types import BBox

    if tracked is not None and anchor is not None and iou(tracked, anchor) > 0.15:
        return tracked
    return anchor


def beat_from_comparison(comp, baseline_new: bool) -> RealBeat:
    if baseline_new:
        return RealBeat.RETURN
    if comp.lumen_ghost and comp.baseline_lost:
        return RealBeat.OCCLUDED
    return RealBeat.VISIBLE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", default="data/raw/mot17/ablation/MOT17-11-FRCNN")
    parser.add_argument("--start-frame", type=int, default=255)
    parser.add_argument("--num-frames", type=int, default=90)
    parser.add_argument("--gt-track-id", type=int, default=1, help="Black suit subject")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--output", default="results/demo_videos/VIDEO2_REAL_WORLD.mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    if not (seq_dir / "img1").exists():
        raise SystemExit(f"Missing {seq_dir}/img1")

    cfg = apply_cfg(load_config(args.config))
    cfg.setdefault("lumen", {})
    cfg["lumen"].update({
        "use_exit_zone": False,
        "latent_enter_frames": 2,
        "latent_max_frames": 60,
        "target_classes": [0],
    })

    num = args.num_frames
    start_mot = args.start_frame
    frames = load_frames(seq_dir, start_mot - 1, start_mot - 1 + num, args.scale)
    fh, fw = frames[0].shape[:2]

    gt_file = seq_dir / "gt" / "gt.txt"
    gt_map = gt_anchor_map(gt_file, args.gt_track_id, start_mot, start_mot + num - 1, args.scale)
    anchor_path = fill_gaps({i: gt_map.get(start_mot + i) for i in range(num)})
    if not any(anchor_path.values()):
        raise SystemExit(f"GT track {args.gt_track_id} not found in clip")

    from lumen.detector.yolo_ultralytics import YOLODetector

    detector = YOLODetector(cfg)
    raw_dets = {}
    for i, frame in enumerate(tqdm(frames, desc="Detect")):
        raw_dets[i] = [d for d in detector.detect_frame(frame) if d.class_id == 0]

    oc_start, oc_end = find_occlusion_by_gt_dropout(raw_dets, anchor_path, num)
    masked_dets = mask_subject_window(raw_dets, anchor_path, oc_start, oc_end)

    engine = DemoComparisonEngine(
        cfg=cfg,
        target_classes=[0],
        anchor_path=anchor_path,
        oc_start=oc_start,
        oc_end=oc_end,
        lock_until_frame=oc_start,
    )
    engine.build(masked_dets, raw_dets=raw_dets)

    beat_sm = StableBeat(hold_frames=5)
    lost_sm = StickyFlag(on_frames=2, off_frames=3)
    ghost_sm = StickyFlag(on_frames=2, off_frames=3)
    smooth = SmoothBBox(alpha=0.42)
    silhouette = SubjectSilhouette()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (fw * 2 + 8, fh + 72))

    intro = compose_real_intro(
        frames[num // 3],
        "PERSIST-AI: Real World",
        "Red = prioritized target (black suit) in a crowded scene",
    )
    for _ in range(18):
        writer.write(
            compose_crowd_frame(intro, RealBeat.VISIBLE, [], None, frame_idx=0, total_frames=num)
        )

    ghost_trail: list[tuple[int, int]] = []
    phase_log: list[str] = []

    print(
        f"GT track {args.gt_track_id} (black suit) | MOT {start_mot}-{start_mot + num - 1} | "
        f"occl {oc_start}-{oc_end} | baseline id {engine.locked_baseline_id} | lumen id {engine.locked_lumen_id}"
    )

    for i in tqdm(range(num), desc="Render"):
        comp = engine.step(i, masked_dets[i])
        anchor = anchor_path.get(i)

        baseline_lost = lost_sm.update(comp.baseline_lost)
        lumen_ghost = ghost_sm.update(comp.lumen_ghost)

        beat = beat_sm.update(beat_from_comparison(comp, comp.baseline_new_id))
        phase_log.append(beat.name)

        base_bb = anchor if (not baseline_lost and anchor is not None) else None
        lumen_bb = comp.lumen_bbox or anchor
        if lumen_bb and lumen_ghost:
            lumen_bb = smooth.update(lumen_bb)
            ghost_trail.append((int(lumen_bb.cx), int(lumen_bb.cy)))
        else:
            if lumen_bb:
                smooth.reset(lumen_bb)
            ghost_trail = []
            if base_bb is not None and not baseline_lost:
                silhouette.update_from_frame(frames[i], base_bb)

        writer.write(
            compose_crowd_frame(
                frames[i],
                beat,
                crowd_dets=raw_dets[i],
                raw_dets=raw_dets[i],
                target_anchor=anchor,
                lumen_target_bb=lumen_bb,
                lumen_ghost=lumen_ghost,
                ghost_trail=ghost_trail if lumen_ghost else None,
                silhouette=silhouette if lumen_ghost else None,
                lumen_visual=comp.lumen_visual if lumen_ghost else None,
                frame_idx=i,
                total_frames=num,
                target_label="TARGET",
                ghost_label="TARGET (ghost)",
            )
        )

    writer.release()
    save_json(
        out.with_suffix(".json"),
        {
            "source": str(seq_dir),
            "gt_track_id": args.gt_track_id,
            "subject": "black suit (GT locked)",
            "start_frame_mot": start_mot,
            "occlusion_clip_indices": [oc_start, oc_end],
            "locked_baseline_id": engine.locked_baseline_id,
            "locked_lumen_id": engine.locked_lumen_id,
            "frames": num,
            "fps": args.fps,
            "phases": phase_log,
        },
    )
    print(f"Done: {out} ({num / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
