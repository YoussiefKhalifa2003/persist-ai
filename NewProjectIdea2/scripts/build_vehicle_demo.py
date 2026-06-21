"""Build VIDEO 3 — real traffic, car occluded by bus/truck, crowd-style viz."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from lumen.data.vehicle_occlusion_finder import (
    CAR_CLASSES,
    scan_full_video_for_truck_occlusion,
)
from lumen.pipelines.comparison_pipeline import DemoComparisonEngine, mask_subject_window
from lumen.utils.io import load_config, save_json
from lumen.viz.crowd_compositor import compose_crowd_frame
from lumen.viz.real_compositor import RealBeat, SmoothBBox, StableBeat, StickyFlag, compose_real_intro
from lumen.viz.silhouette import SubjectSilhouette


VEHICLE_YOLO = [2, 3, 5, 7]
SUBJECT_CLASSES = [2, 3]

VEHICLE_BEAT = {
    RealBeat.VISIBLE: "Both trackers see the target car.",
    RealBeat.OCCLUDED: "Car hidden behind bus/truck - baseline forgets, PERSIST-AI keeps ghost.",
    RealBeat.RETURN: "Car back - baseline NEW id, PERSIST-AI kept TARGET.",
}


def apply_cfg(cfg: dict) -> dict:
    import torch

    if str(cfg.get("device", "cuda:0")).startswith("cuda") and not torch.cuda.is_available():
        cfg["device"] = "cpu"
        det = cfg.setdefault("detector", {})
        det["half"] = False
        det["model"] = "yolov8n.pt"
        det["imgsz"] = 640
        det["conf"] = 0.22
    det = cfg.setdefault("detector", {})
    det["classes"] = VEHICLE_YOLO
    det["conf"] = min(float(det.get("conf", 0.25)), 0.24)
    return cfg


def load_video_frames(video_path: Path, max_frames: int, scale: float, skip: int = 0) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if skip:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def resolve_video_path(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise SystemExit(f"Video not found: {p}")
    for hits in (
        sorted(Path("data/raw/driving").rglob("*.mp4")),
    ):
        if hits:
            return hits[0]
    import importlib.util

    dl = Path(__file__).with_name("download_driving_sample.py")
    spec = importlib.util.spec_from_file_location("download_driving_sample", dl)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.download_traffic_sample()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="MP4 path; tries all in data/raw/driving if unset")
    parser.add_argument("--scan-frames", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--scale", type=float, default=0.65)
    parser.add_argument("--output", default="results/demo_videos/VIDEO3_VEHICLES.mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    cfg = apply_cfg(load_config(args.config))
    cfg.setdefault("lumen", {})
    cfg["lumen"].update({
        "use_exit_zone": True,
        "latent_enter_frames": 2,
        "latent_max_frames": 80,
        "target_classes": SUBJECT_CLASSES,
        "pedestrian_only": False,
    })

    scan_frames = load_video_frames(video_path, args.scan_frames, args.scale, skip=0)
    if len(scan_frames) < 120:
        raise SystemExit(f"Not enough frames in {video_path}")

    from lumen.detector.yolo_ultralytics import YOLODetector

    detector = YOLODetector(cfg)
    all_dets: dict[int, list] = {}
    for i, frame in enumerate(tqdm(scan_frames, desc="Detect vehicles")):
        all_dets[i] = [d for d in detector.detect_frame(frame) if d.class_id in CAR_CLASSES]

    clip = scan_full_video_for_truck_occlusion(all_dets, len(scan_frames))
    if clip is None:
        for alt in sorted(Path("data/raw/driving").glob("*.mp4")):
            if alt.resolve() == video_path.resolve():
                continue
            print(f"Trying alternate video: {alt.name}")
            alt_frames = load_video_frames(alt, min(args.scan_frames, 480), args.scale, skip=0)
            alt_dets = {}
            for i, frame in enumerate(tqdm(alt_frames, desc=f"Detect {alt.name}")):
                alt_dets[i] = [d for d in detector.detect_frame(frame) if d.class_id in CAR_CLASSES]
            clip = scan_full_video_for_truck_occlusion(alt_dets, len(alt_frames))
            if clip is not None:
                scan_frames, all_dets, video_path = alt_frames, alt_dets, alt
                break
    if clip is None:
        raise SystemExit(
            "No car-behind-bus/truck clip found. Try a longer scan or another video in data/raw/driving/."
        )

    frames = scan_frames[clip.start : clip.end]
    num = len(frames)
    raw_dets = {i: all_dets[clip.start + i] for i in range(num)}
    anchor_path = clip.anchor_path
    oc_start, oc_end = clip.oc_start, clip.oc_end

    masked_dets = mask_subject_window(raw_dets, anchor_path, oc_start, oc_end, thresh=0.18)
    vehicle_dets = {i: [d for d in raw_dets[i] if d.class_id in {5, 7}] for i in range(num)}

    engine = DemoComparisonEngine(
        cfg=cfg,
        target_classes=SUBJECT_CLASSES,
        anchor_path=anchor_path,
        oc_start=oc_start,
        oc_end=oc_end,
        lock_until_frame=oc_start,
    )
    engine.build(masked_dets, vehicle_dets, raw_dets=raw_dets)

    beat_sm = StableBeat(hold_frames=5)
    lost_sm = StickyFlag(on_frames=2, off_frames=3)
    ghost_sm = StickyFlag(on_frames=2, off_frames=3)
    smooth = SmoothBBox(alpha=0.40)
    silhouette = SubjectSilhouette()

    fh, fw = frames[0].shape[:2]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (fw * 2 + 8, fh + 72))

    intro = compose_real_intro(
        frames[num // 3],
        "PERSIST-AI: Vehicles",
        "Red = target car | must pass behind bus or truck",
    )
    for _ in range(18):
        writer.write(
            compose_crowd_frame(
                intro,
                RealBeat.VISIBLE,
                [],
                None,
                frame_idx=0,
                total_frames=num,
                header="VIDEO 3 - VEHICLES | Real traffic",
            )
        )

    ghost_trail: list[tuple[int, int]] = []
    phase_log: list[str] = []

    print(
        f"{video_path.name} | clip {clip.start}-{clip.end} | car track {clip.track_id} | "
        f"occlusion {oc_start}-{oc_end} | score {clip.score:.1f}"
    )

    for i in tqdm(range(num), desc="Render vehicles"):
        comp = engine.step(i, masked_dets[i], vehicle_dets[i])
        anchor = anchor_path.get(i)

        baseline_lost = lost_sm.update(comp.baseline_lost)
        lumen_ghost = ghost_sm.update(comp.lumen_ghost)

        if comp.baseline_new_id:
            beat = beat_sm.update(RealBeat.RETURN)
        elif comp.lumen_ghost and comp.baseline_lost:
            beat = beat_sm.update(RealBeat.OCCLUDED)
        else:
            beat = beat_sm.update(RealBeat.VISIBLE)
        phase_log.append(beat.name)

        base_bb = comp.baseline_bbox or anchor
        lumen_bb = comp.lumen_bbox
        if lumen_bb and lumen_ghost:
            lumen_bb = smooth.update(lumen_bb)
            ghost_trail.append((int(lumen_bb.cx), int(lumen_bb.cy)))
        else:
            if lumen_bb:
                smooth.reset(lumen_bb)
            ghost_trail = []
            if base_bb is not None and not baseline_lost:
                silhouette.update_from_frame(frames[i], base_bb)

        occluders = []
        if lumen_ghost and anchor is not None:
            occluders = [
                d for d in raw_dets[i]
                if d.class_id in {5, 7}
                or (d.class_id in {2, 3, 7} and d.bbox.area > (anchor.area if anchor else 0) * 1.5)
            ]

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
                occluder_dets=occluders if occluders else None,
                lumen_visual=comp.lumen_visual if lumen_ghost else None,
                frame_idx=i,
                total_frames=num,
                header="VIDEO 3 - VEHICLES | Real traffic",
                beat_label=VEHICLE_BEAT.get(beat, beat.value),
                target_label="TARGET CAR",
                ghost_label="TARGET CAR (ghost)",
            )
        )

    writer.release()
    save_json(
        out.with_suffix(".json"),
        {
            "source_video": str(video_path),
            "clip_start": clip.start,
            "clip_end": clip.end,
            "vehicle_track_id": clip.track_id,
            "occlusion": [oc_start, oc_end],
            "requires_truck_bus": True,
            "frames": num,
            "fps": args.fps,
            "phases": phase_log,
        },
    )
    print(f"Done: {out} ({num / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
