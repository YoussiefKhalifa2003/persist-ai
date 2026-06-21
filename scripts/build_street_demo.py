"""Build VIDEO 3 — YouTube sidewalk: buses + 3 women, PERSIST-AI tracks tan-coat TARGET."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lumen.brand import BRAND, TAGLINE
from lumen.data.pedestrian_clip_finder import (
    build_path,
    build_tan_coat_clip,
    clip_from_manual,
    find_women_group_clip,
    scan_video_for_pedestrian_clip,
    _person_visible,
    _track_people,
)
from lumen.pipelines.comparison_pipeline import DemoComparisonEngine, mask_subject_window
from lumen.pipelines.persist_occlusion import (
    frame_is_persist_latent,
    mask_subject_windows,
)
from lumen.utils.io import load_config, save_json
from lumen.viz.crowd_compositor import compose_crowd_frame
from lumen.viz.real_compositor import RealBeat, SmoothBBox, StableBeat, StickyFlag, compose_real_intro
from lumen.viz.silhouette import SubjectSilhouette
from lumen.viz.staged_compositor import (
    compose_activate_button_frame,
    compose_raw_fullscreen,
)


DEFAULT_URL = "https://www.youtube.com/watch?v=XKlcEGaLoPM"
DETECT_CLASSES = [0, 2, 5, 7]
OCCLUDER_CLASSES = {2, 5, 7}

DEFAULT_CLIP = {"start": 55, "end": 170, "oc_start": 85, "oc_end": 112}

STREET_BEAT = {
    RealBeat.VISIBLE: f"{BRAND} locked on tan-coat woman — raw YOLO sees everyone.",
    RealBeat.OCCLUDED: f"Vehicle blocks view — raw loses her, {BRAND} keeps ghost.",
    RealBeat.RETURN: f"She reappears — {BRAND} kept the same TARGET.",
}

IDLE_CAPTION = "Target exited — standard detection only."

ACT1_CAPTION = "Standard perception: only what the camera sees."
ACT1_FRAMES = 45  # ~3s @ 15fps
ACT2_FRAMES = 22  # ~1.5s button animation


def apply_cfg(cfg: dict) -> dict:
    import torch

    if str(cfg.get("device", "cuda:0")).startswith("cuda") and not torch.cuda.is_available():
        cfg["device"] = "cpu"
        det = cfg.setdefault("detector", {})
        det["half"] = False
        det["model"] = "yolov8n.pt"
        det["imgsz"] = 640
        det["conf"] = 0.20
    det = cfg.setdefault("detector", {})
    det["classes"] = DETECT_CLASSES
    return cfg


def load_video(path: Path, max_frames: int, scale: float) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames


def download_youtube(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import sys

    ytdlp = Path(sys.executable).with_name("yt-dlp.exe")
    if not ytdlp.exists():
        raise SystemExit("Install yt-dlp: pip install yt-dlp")
    subprocess.run(
        [str(ytdlp), "-f", "18/b[ext=mp4]/bv*+ba/b", "--merge-output-format", "mp4", "-o", str(dest), url],
        check=True,
    )
    return dest


def resolve_video(url: str | None, local: Path | None) -> Path:
    if local and local.exists():
        return local
    dest = Path("data/raw/youtube/sidewalk_demo.mp4")
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    return download_youtube(url or DEFAULT_URL, dest)


def render_frame_bundle(
    i: int,
    num: int,
    frames: list,
    raw_dets: dict,
    anchor_path: dict,
    occlusion_windows: list[tuple[int, int]],
    engine: DemoComparisonEngine,
    masked_dets: dict,
    bus_dets: dict,
    beat_sm: StableBeat,
    lost_sm: StickyFlag,
    ghost_sm: StickyFlag,
    smooth: SmoothBBox,
    silhouette: SubjectSilhouette,
    ghost_trail: list,
) -> tuple[np.ndarray, RealBeat, list, dict]:
    comp = engine.step(i, masked_dets[i], bus_dets[i])
    anchor = anchor_path.get(i)
    visible = bool(anchor and _person_visible(raw_dets[i], anchor, thresh=0.14))
    in_oc = frame_is_persist_latent(i, anchor, raw_dets[i], occlusion_windows)

    baseline_lost = lost_sm.update(comp.baseline_lost or in_oc)
    lumen_ghost = ghost_sm.update(comp.lumen_ghost and bool(anchor) and in_oc)
    if in_oc:
        baseline_lost = True
        lumen_ghost = True

    if comp.baseline_new_id and not in_oc and anchor:
        beat = beat_sm.update(RealBeat.RETURN)
    elif in_oc:
        beat = beat_sm.update(RealBeat.OCCLUDED)
    elif anchor and visible:
        beat = beat_sm.update(RealBeat.VISIBLE)
    elif anchor:
        beat = beat_sm.update(RealBeat.VISIBLE)
    else:
        beat = beat_sm.update(RealBeat.VISIBLE)

    if in_oc and anchor:
        lumen_bb = smooth.update(anchor)
        ghost_trail.append((int(lumen_bb.cx), int(lumen_bb.cy)))
    elif anchor and visible:
        lumen_bb = anchor
        smooth.reset(lumen_bb)
        ghost_trail.clear()
        silhouette.update_from_frame(frames[i], anchor)
    else:
        lumen_bb = None
        ghost_trail.clear()

    occluders = [d for d in raw_dets[i] if d.class_id in OCCLUDER_CLASSES] if lumen_ghost else None

    caption = STREET_BEAT.get(beat, beat.value)
    if not anchor and not in_oc:
        caption = IDLE_CAPTION

    meta = {
        "frame": i,
        "phase": beat.name,
        "occlusion": in_oc,
        "ghost_drawn": bool(lumen_ghost and lumen_bb is not None),
        "anchor_cx": round(anchor.cx, 1) if anchor else None,
    }

    split = compose_crowd_frame(
        frames[i],
        beat,
        crowd_dets=[d for d in raw_dets[i] if d.class_id == 0],
        raw_dets=raw_dets[i],
        target_anchor=anchor,
        lumen_target_bb=lumen_bb,
        lumen_ghost=lumen_ghost and lumen_bb is not None,
        ghost_trail=ghost_trail if lumen_ghost else None,
        silhouette=silhouette if lumen_ghost else None,
        occluder_dets=occluders,
        lumen_visual=comp.lumen_visual if lumen_ghost else None,
        frame_idx=i,
        total_frames=num,
        header="VIDEO 3 - STREET | PERSIST-AI (left) vs Raw YOLO (right)",
        beat_label=caption,
        target_label="TARGET",
        ghost_label="TARGET (ghost)",
        layout="split",
    )
    return split, beat, ghost_trail, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--video", default=None)
    parser.add_argument("--scan-frames", type=int, default=600)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--output", default="results/demo_videos/VIDEO3_VEHICLES.mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target-track-id", type=int, default=None)
    parser.add_argument("--clip-start", type=int, default=None)
    parser.add_argument("--clip-end", type=int, default=None)
    parser.add_argument("--oc-start", type=int, default=None)
    parser.add_argument("--oc-end", type=int, default=None)
    parser.add_argument("--detect-cache", default="data/cache/sidewalk_demo_dets.json")
    parser.add_argument("--staged", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    video_path = resolve_video(args.url, Path(args.video) if args.video else None)
    cfg = apply_cfg(load_config(args.config))
    cfg.setdefault("lumen", {})
    cfg["lumen"].update({
        "use_exit_zone": True,
        "latent_enter_frames": 2,
        "latent_max_frames": 70,
        "target_classes": [0],
        "pedestrian_only": True,
    })

    scan_frames = load_video(video_path, args.scan_frames, args.scale)
    if len(scan_frames) < 60:
        raise SystemExit(f"Video too short: {video_path}")

    from lumen.detector.yolo_ultralytics import YOLODetector
    from lumen.types import BBox, Detection

    cache_path = Path(args.detect_cache) if args.detect_cache else None
    all_dets: dict[int, list] = {}
    if cache_path and cache_path.exists():
        import json

        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        for k, items in raw.items():
            all_dets[int(k)] = [
                Detection(BBox(*item["bbox"]), item["conf"], item["class_id"]) for item in items
            ]
        print(f"Loaded detections from {cache_path}")
    else:
        detector = YOLODetector(cfg)
        for i, frame in enumerate(tqdm(scan_frames, desc="Detect")):
            all_dets[i] = [d for d in detector.detect_frame(frame) if d.class_id in DETECT_CLASSES]
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                str(k): [{"bbox": d.bbox.as_xyxy(), "conf": d.confidence, "class_id": d.class_id} for d in v]
                for k, v in all_dets.items()
            }
            cache_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    n_total = len(scan_frames)
    fh, fw = scan_frames[0].shape[:2]
    manual = args.clip_start is not None and args.clip_end is not None and args.target_track_id is not None
    use_sidewalk_default = not manual and args.target_track_id is None and "sidewalk" in video_path.stem.lower()

    if manual:
        if args.oc_start is None or args.oc_end is None:
            raise SystemExit("Manual clip requires --oc-start and --oc-end.")
        clip = clip_from_manual(
            all_dets, n_total, args.target_track_id,
            args.clip_start, args.clip_end, args.oc_start, args.oc_end,
        )
    elif use_sidewalk_default:
        dc = DEFAULT_CLIP
        clip = build_tan_coat_clip(
            all_dets, dc["start"], min(n_total, dc["end"]),
            dc["oc_start"], dc["oc_end"], frame_w=float(fw),
        )
        print(f"Tan-coat lock | occlusions {clip.occlusion_windows} | {clip.end - clip.start} frames")
    else:
        clip = scan_video_for_pedestrian_clip(all_dets, n_total, prefer_track=args.target_track_id)
        if clip is None:
            clip = find_women_group_clip(all_dets, n_total)
        if clip is None:
            dc = DEFAULT_CLIP
            clip = build_tan_coat_clip(
                all_dets, dc["start"], min(n_total, dc["end"]),
                dc["oc_start"], dc["oc_end"], frame_w=float(fw),
            )

    frames = scan_frames[clip.start : clip.end]
    num = len(frames)
    raw_dets = {i: all_dets[clip.start + i] for i in range(num)}
    anchor_path = clip.anchor_path
    oc_start, oc_end = clip.oc_start, clip.oc_end
    occlusion_windows = clip.occlusion_windows or [(oc_start, oc_end)]

    masked_dets = mask_subject_windows(raw_dets, anchor_path, occlusion_windows, thresh=0.20)
    bus_dets = {i: [d for d in raw_dets[i] if d.class_id in OCCLUDER_CLASSES] for i in range(num)}

    engine = DemoComparisonEngine(
        cfg=cfg, target_classes=[0], anchor_path=anchor_path,
        oc_start=oc_start, oc_end=oc_end, occlusion_windows=occlusion_windows,
        lock_until_frame=occlusion_windows[0][0] if occlusion_windows else oc_start,
    )
    engine.build(masked_dets, bus_dets, raw_dets=raw_dets)

    beat_sm = StableBeat(hold_frames=4)
    lost_sm = StickyFlag(on_frames=2, off_frames=3)
    ghost_sm = StickyFlag(on_frames=2, off_frames=3)
    smooth = SmoothBBox(alpha=0.40)
    silhouette = SubjectSilhouette()

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    split_w = fw * 2 + 8
    canvas_h = fh + 72

    writers: dict[str, cv2.VideoWriter] = {}
    for name, path in [
        ("main", args.output),
        ("raw", str(out_dir / "VIDEO3_RAW_ONLY.mp4")),
        ("split", str(out_dir / "VIDEO3_PERSIST_SPLIT.mp4")),
    ]:
        writers[name] = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (split_w, canvas_h)
        )

    phase_log: list[str] = []
    manifest_frames: list[dict] = []
    ghost_trail: list[tuple[int, int]] = []

    # Pre-render all split frames for staged + split export
    split_frames: list[np.ndarray] = []
    for i in tqdm(range(num), desc="Render"):
        split, beat, ghost_trail, meta = render_frame_bundle(
            i, num, frames, raw_dets, anchor_path, occlusion_windows,
            engine, masked_dets, bus_dets, beat_sm, lost_sm, ghost_sm,
            smooth, silhouette, ghost_trail,
        )
        split_frames.append(split)
        phase_log.append(beat.name)
        manifest_frames.append(meta)

    act1_end = min(ACT1_FRAMES, num)
    act1_last = act1_end - 1

    if args.staged:
        intro = compose_real_intro(
            frames[min(num // 3, num - 1)],
            "PERSIST-AI: Street Scene",
            "Left = PERSIST-AI | Right = Raw YOLO | Tan-coat woman = TARGET",
        )
        for _ in range(12):
            writers["main"].write(
                compose_crowd_frame(intro, RealBeat.VISIBLE, [], None, layout="split", frame_idx=0, total_frames=num)
            )

        # Act 1: raw fullscreen
        for i in range(act1_end):
            raw_frame = compose_raw_fullscreen(
                frames[i], raw_dets[i], RealBeat.VISIBLE,
                "VIDEO 3 - STREET | Raw detection only",
                ACT1_CAPTION, i, num, split_w,
            )
            writers["main"].write(raw_frame)
            writers["raw"].write(raw_frame)

        # Act 2: activate button
        frozen = frames[act1_last]
        for t in range(ACT2_FRAMES):
            pulse = 0.5 + 0.5 * math.sin(t / ACT2_FRAMES * math.pi * 2)
            btn = compose_activate_button_frame(
                frozen, pulse, split_w, "VIDEO 3 - STREET | Activate PERSIST-AI",
            )
            writers["main"].write(btn)

        # Act 3: split comparison
        for i in range(num):
            writers["main"].write(split_frames[i])
            writers["split"].write(split_frames[i])
    else:
        for i in range(num):
            writers["main"].write(split_frames[i])
            writers["split"].write(split_frames[i])
            raw_frame = compose_raw_fullscreen(
                frames[i], raw_dets[i], RealBeat.VISIBLE,
                "VIDEO 3 - STREET", ACT1_CAPTION, i, num, split_w,
            )
            writers["raw"].write(raw_frame)

    for w in writers.values():
        w.release()

    manifest = {
        "source": str(video_path),
        "youtube": args.url,
        "clip": [clip.start, clip.end],
        "occlusion_windows": occlusion_windows,
        "occlusion_primary": [oc_start, oc_end],
        "frames": num,
        "fps": args.fps,
        "staged": args.staged,
        "act1_frames": act1_end,
        "act2_frames": ACT2_FRAMES,
        "split_start_frame": act1_end + ACT2_FRAMES if args.staged else 0,
        "phases": phase_log,
        "frame_meta": manifest_frames,
        "assets": {
            "main": args.output,
            "raw_only": str(out_dir / "VIDEO3_RAW_ONLY.mp4"),
            "lumen_split": str(out_dir / "VIDEO3_PERSIST_SPLIT.mp4"),
        },
    }
    save_json(out_dir / "VIDEO3_manifest.json", manifest)
    save_json(Path(args.output).with_suffix(".json"), manifest)
    print(f"Done: {args.output} | raw: VIDEO3_RAW_ONLY.mp4 | split: VIDEO3_PERSIST_SPLIT.mp4")


if __name__ == "__main__":
    main()
