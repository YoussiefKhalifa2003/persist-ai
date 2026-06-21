"""Download and register Video 5/6 curated PERSIST-AI scenes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.data.selectable_target import build_selectable_target_clip, candidate_tracking_quality
from lumen.detector.yolo_ultralytics import YOLODetector
from lumen.types import BBox, Detection
from lumen.utils.io import load_config, save_json
from lumen.web.interactive_demo import render_selected_target


SCENE_CONFIG = Path("configs/interactive_scenes.json")
SUPPORTED_CLASSES = [0, 1, 2, 3, 5, 7]
INGEST_MODEL = "yolov8x.pt"
INGEST_IMGSZ = 1280
INGEST_CONF = 0.10
PARTIAL_SAVE_EVERY = 25


@dataclass(frozen=True)
class IngestSpec:
    number: int
    scene_id: str
    name: str
    youtube_url: str
    source_video: Path
    preview_video: Path
    known_good_split: Path
    detection_cache: Path


SPECS = [
    IngestSpec(
        number=5,
        scene_id="video5-static-street",
        name="Video 5 - Static Street Scene",
        youtube_url="https://www.youtube.com/watch?v=sz8BO2_wW0k",
        source_video=Path("data/raw/youtube/video5_static_street.mp4"),
        preview_video=Path("results/demo_videos/VIDEO5_RAW_PREVIEW.mp4"),
        known_good_split=Path("results/demo_videos/VIDEO5_PERSIST_SPLIT.mp4"),
        detection_cache=Path("data/cache/video5_static_street_dets.json"),
    ),
    IngestSpec(
        number=6,
        scene_id="video6-static-street",
        name="Video 6 - Static Street Scene",
        youtube_url="https://www.youtube.com/watch?v=3FXUw98rrUY",
        source_video=Path("data/raw/youtube/video6_static_street.mp4"),
        preview_video=Path("results/demo_videos/VIDEO6_RAW_PREVIEW.mp4"),
        known_good_split=Path("results/demo_videos/VIDEO6_PERSIST_SPLIT.mp4"),
        detection_cache=Path("data/cache/video6_static_street_dets.json"),
    ),
]


def _yt_dlp() -> Path:
    exe = Path(sys.executable).with_name("yt-dlp.exe")
    if exe.exists():
        return exe
    raise RuntimeError("yt-dlp is not installed in the active Python environment.")


def download_video(spec: IngestSpec, overwrite: bool = False) -> None:
    if spec.source_video.exists() and spec.source_video.stat().st_size > 100_000 and not overwrite:
        return
    spec.source_video.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and spec.source_video.exists():
        spec.source_video.unlink()
    subprocess.run(
        [
            str(_yt_dlp()),
            "-f",
            "bv*[height<=1080][ext=mp4][vcodec^=avc1]/bv*[height<=720][ext=mp4][vcodec^=avc1]/b[height<=1080][ext=mp4]/b[height<=720][ext=mp4]/18",
            "--force-overwrites",
            "--merge-output-format",
            "mp4",
            "-o",
            str(spec.source_video),
            spec.youtube_url,
        ],
        check=True,
    )


def video_meta(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video metadata: {path}")
    return frames, fps, width, height


def best_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _load_detection_cache(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    normalized: dict[str, list[dict]] = {}
    for frame, items in raw.items():
        try:
            frame_key = str(int(frame))
        except (TypeError, ValueError):
            continue
        if isinstance(items, list):
            normalized[frame_key] = items
    return normalized


def detect_video_json(spec: IngestSpec, max_frames: int | None = None, overwrite: bool = False) -> None:
    dets_map = {} if overwrite else _load_detection_cache(spec.detection_cache)
    if max_frames is not None and all(str(i) in dets_map for i in range(max_frames)):
        return
    cfg = load_config("configs/default.yaml")
    cfg["device"] = best_torch_device()
    det_cfg = dict(cfg.get("detector", {}))
    det_cfg["model"] = INGEST_MODEL
    det_cfg["imgsz"] = INGEST_IMGSZ
    det_cfg["classes"] = SUPPORTED_CLASSES
    det_cfg["conf"] = min(float(det_cfg.get("conf", 0.25)), INGEST_CONF)
    det_cfg["iou"] = max(float(det_cfg.get("iou", 0.45)), 0.50)
    det_cfg["half"] = cfg["device"].startswith("cuda")
    cfg["detector"] = det_cfg
    detector = YOLODetector(cfg)

    cap = cv2.VideoCapture(str(spec.source_video))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break
        if str(idx) not in dets_map:
            dets = detector.detect_frame(frame)
            dets_map[str(idx)] = [
                {"bbox": d.bbox.as_xyxy(), "conf": d.confidence, "class_id": d.class_id}
                for d in dets
                if d.class_id in SUPPORTED_CLASSES
            ]
            if idx and idx % PARTIAL_SAVE_EVERY == 0:
                save_json(spec.detection_cache, dets_map)
                print(f"   cached detections through frame {idx} on {cfg['device']}")
        idx += 1
    cap.release()
    save_json(spec.detection_cache, dets_map)


def load_dets(path: Path) -> dict[int, list[Detection]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(frame): [Detection(BBox(*item["bbox"]), item["conf"], item["class_id"]) for item in items]
        for frame, items in raw.items()
    }


def window_score(dets: dict[int, list[Detection]], start: int, end: int) -> float:
    people = vehicles = low_miss = crowd_frames = 0
    prev_people = 0
    for i in range(start, end):
        frame_dets = dets.get(i, [])
        person_count = sum(1 for d in frame_dets if d.class_id == 0 and d.bbox.h >= 24)
        vehicle_count = sum(1 for d in frame_dets if d.class_id in {1, 2, 3, 5, 7})
        people += min(person_count, 10)
        vehicles += min(vehicle_count, 5)
        if person_count >= 3:
            crowd_frames += 1
        if prev_people and person_count < prev_people:
            low_miss += min(prev_people - person_count, 4)
        prev_people = person_count
    length = max(1, end - start)
    return people * 1.8 + vehicles * 1.2 + crowd_frames * 0.7 + low_miss * 2.2 - length * 0.02


def choose_clip(dets: dict[int, list[Detection]], total_frames: int, fps: float) -> tuple[int, int]:
    window = max(45, int(round(fps * 15)))
    step = max(15, int(round(fps * 2)))
    if total_frames <= window:
        return 0, total_frames
    starts = range(0, total_frames - window + 1, step)
    best_start = max(starts, key=lambda s: window_score(dets, s, s + window))
    return best_start, min(total_frames, best_start + window)


def _rough_candidate_score(det: Detection, frame_h: int) -> float:
    score = det.confidence * 25.0
    if det.class_id == 0:
        size = det.bbox.h / max(1.0, float(frame_h))
        score += 30.0
        if size > 0.62:
            score -= 90.0
        elif size < 0.055:
            score -= 28.0
        else:
            score += 48.0 * (1.0 - min(1.0, abs(size - 0.22) / 0.28))
    else:
        score += min(det.bbox.area / 9000.0, 1.0) * 35.0
    return score


def choose_candidate(
    dets: dict[int, list[Detection]],
    start: int,
    end: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int]:
    clip_len = max(1, end - start)
    midpoint = (start + end) // 2
    step = max(3, clip_len // 72)
    search_frames = list(range(start, end, step))
    if midpoint not in search_frames:
        search_frames.append(midpoint)
    search_frames = sorted(set(search_frames), key=lambda i: abs(i - midpoint))
    quality_bonus = {"high": 65.0, "degraded": 25.0, "low": -20.0}
    best: tuple[float, int, int] | None = None
    for frame in search_frames:
        frame_candidates = [
            (idx, det)
            for idx, det in enumerate(dets.get(frame, []))
            if det.class_id in SUPPORTED_CLASSES
        ]
        frame_candidates.sort(key=lambda item: _rough_candidate_score(item[1], frame_h), reverse=True)
        for idx, det in frame_candidates[:8]:
            try:
                clip = build_selectable_target_clip(
                    dets,
                    start,
                    end,
                    frame,
                    idx,
                    float(frame_w),
                    min_visible_frames=1,
                )
            except ValueError:
                continue
            rendered_len = clip.end - clip.start
            visible = len(clip.visible_frames)
            ghost_frames = sum(win_end - win_start for win_start, win_end in clip.occlusion_windows)
            visible_ratio = visible / max(1, rendered_len)
            midpoint_bonus = max(0.0, 1.0 - abs(frame - midpoint) / max(1.0, clip_len / 2.0)) * 18.0
            early_exit_penalty = max(0.0, clip_len * 0.55 - rendered_len) * 1.25
            long_prediction_penalty = max(0, ghost_frames - max(18, visible)) * 1.8
            score = (
                visible * 5.0
                + rendered_len * 0.7
                + visible_ratio * 55.0
                + midpoint_bonus
                + quality_bonus.get(candidate_tracking_quality(det), -20.0)
                - ghost_frames * 0.55
                - early_exit_penalty
                - long_prediction_penalty
            )
            if det.class_id == 0:
                size = det.bbox.h / max(1.0, float(frame_h))
                score += 45.0
                if size > 0.62:
                    score -= 140.0
                elif size < 0.055:
                    score -= 30.0
            if clip.tracking_quality == "low":
                score -= 55.0
            elif clip.tracking_quality == "degraded":
                score -= 20.0
            if best is None or score > best[0]:
                best = (score, frame, idx)
    if best is None:
        raise RuntimeError("No supported candidates found in selected clip.")
    return best[1], best[2]


def write_preview(spec: IngestSpec, clip_start: int, clip_end: int, fps: float) -> None:
    spec.preview_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(spec.source_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read preview frame from {spec.source_video}")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(spec.preview_video), cv2.VideoWriter_fourcc(*"mp4v"), int(round(fps)), (w, h))
    writer.write(frame)
    for _ in range(clip_start + 1, clip_end):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()


def write_contact_sheet(spec: IngestSpec, clip_start: int, clip_end: int) -> Path:
    out = Path("results/ingestion") / f"VIDEO{spec.number}_contact_sheet.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(spec.source_video))
    indices = [int(clip_start + (clip_end - clip_start - 1) * t / 8) for t in range(9)]
    thumbs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (360, 202), interpolation=cv2.INTER_AREA)
        cv2.putText(frame, f"f{idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(frame)
    cap.release()
    if not thumbs:
        raise RuntimeError("Could not build contact sheet.")
    while len(thumbs) % 3:
        thumbs.append(thumbs[-1].copy() * 0)
    import numpy as np

    sheet = np.vstack([np.hstack(thumbs[i : i + 3]) for i in range(0, len(thumbs), 3)])
    cv2.imwrite(str(out), sheet)
    return out


def upsert_scene(spec: IngestSpec, clip_start: int, clip_end: int, fps: float, candidate_frame: int) -> None:
    scenes = json.loads(SCENE_CONFIG.read_text(encoding="utf-8"))
    scene = {
        "scene_id": spec.scene_id,
        "name": spec.name,
        "description": "Static-camera street scene for broad PERSIST-AI validation.",
        "youtube_url": spec.youtube_url,
        "source_video": str(spec.source_video).replace("\\", "/"),
        "preview_video": str(spec.preview_video).replace("\\", "/"),
        "known_good_split": str(spec.known_good_split).replace("\\", "/"),
        "detection_cache": str(spec.detection_cache).replace("\\", "/"),
        "output_dir": "results/interactive_jobs",
        "clip_start": clip_start,
        "clip_end": clip_end,
        "scale": 1.0,
        "fps": int(round(fps)),
        "recommended_frame": max(0, candidate_frame - clip_start),
        "selectable_frame_range": [0, max(0, clip_end - clip_start - 1)],
        "supported_classes": SUPPORTED_CLASSES,
        "quality_notes": "Auto-selected 15s segment; valid targets render with confidence/uncertainty states.",
    }
    out = [s for s in scenes if s.get("scene_id") != spec.scene_id]
    out.append(scene)
    save_json(SCENE_CONFIG, out)


def render_known_good(spec: IngestSpec, candidate_frame: int, candidate_idx: int) -> None:
    out_dir = Path("results/interactive_jobs") / f"known_good_video{spec.number}"
    video, _manifest = render_selected_target(
        spec.scene_id,
        f"{candidate_frame}:{candidate_idx}",
        0,
        out_dir,
    )
    spec.known_good_split.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(video, spec.known_good_split)
    shutil.copyfile(_manifest, spec.known_good_split.with_name(f"VIDEO{spec.number}_manifest.json"))


def ingest(spec: IngestSpec, overwrite: bool, max_seconds: int | None) -> None:
    print(f"== VIDEO{spec.number}: download")
    download_video(spec, overwrite=overwrite)
    total_frames, fps, frame_w, frame_h = video_meta(spec.source_video)
    max_frames = min(total_frames, int(fps * max_seconds)) if max_seconds else total_frames
    print(f"== VIDEO{spec.number}: detect {max_frames} frames")
    detect_video_json(spec, max_frames=max_frames, overwrite=overwrite)
    dets = load_dets(spec.detection_cache)
    clip_start, clip_end = choose_clip(dets, min(total_frames, max_frames), fps)
    clip_end = min(clip_end, total_frames)
    candidate_frame, candidate_idx = choose_candidate(dets, clip_start, clip_end, frame_w, frame_h)
    print(f"== VIDEO{spec.number}: clip {clip_start}-{clip_end}, candidate {candidate_frame}:{candidate_idx}")
    write_preview(spec, clip_start, clip_end, fps)
    sheet = write_contact_sheet(spec, clip_start, clip_end)
    upsert_scene(spec, clip_start, clip_end, fps, candidate_frame)
    render_known_good(spec, candidate_frame, candidate_idx)
    print(f"== VIDEO{spec.number}: done | contact sheet {sheet}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["5", "6"], help="Ingest only one scene.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=120)
    args = parser.parse_args()
    specs = [s for s in SPECS if args.only is None or str(s.number) == args.only]
    for spec in specs:
        ingest(spec, overwrite=args.overwrite, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
