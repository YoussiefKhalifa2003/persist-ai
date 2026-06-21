"""Service layer for the local Try PERSIST-AI demo."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2

from lumen.data.selectable_target import build_candidates, build_selectable_target_clip, class_name
from lumen.pipelines.comparison_pipeline import DemoComparisonEngine, LumenVisualState
from lumen.pipelines.persist_occlusion import mask_subject_windows, target_visible_enough
from lumen.types import BBox, Detection
from lumen.utils.io import load_config, save_json
from lumen.viz.crowd_compositor import compose_crowd_frame
from lumen.viz.real_compositor import RealBeat, SmoothBBox, StableBeat, StickyFlag
from lumen.viz.silhouette import SubjectSilhouette

OCCLUDER_CLASSES = {1, 2, 3, 5, 7}
SCENE_CONFIG = Path("configs/interactive_scenes.json")


@dataclass
class RenderJob:
    job_id: str
    scene_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    video_path: str | None = None
    manifest_path: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


JOBS: dict[str, RenderJob] = {}
JOBS_LOCK = threading.Lock()


def _repo_path(path: str | Path) -> Path:
    return Path(path)


def load_scenes(config_path: Path = SCENE_CONFIG) -> list[dict[str, Any]]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return raw


def get_scene(scene_id: str) -> dict[str, Any]:
    for scene in load_scenes():
        if scene["scene_id"] == scene_id:
            return scene
    raise KeyError(f"Unknown scene: {scene_id}")


def public_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": scene["scene_id"],
        "name": scene["name"],
        "description": scene.get("description", ""),
        "preview_video": f"/media/{scene['preview_video']}",
        "known_good_split": f"/media/{scene['known_good_split']}",
        "recommended_frame": scene.get("recommended_frame", 0),
        "selectable_frame_range": scene.get("selectable_frame_range", [0, scene["clip_end"] - scene["clip_start"]]),
        "supported_classes": scene.get("supported_classes", [0, 1, 2, 3, 5, 7]),
        "fps": scene.get("fps", 15),
        "frames": scene["clip_end"] - scene["clip_start"],
    }


def load_detections(scene: dict[str, Any]) -> dict[int, list[Detection]]:
    payload = json.loads(_repo_path(scene["detection_cache"]).read_text(encoding="utf-8"))
    dets: dict[int, list[Detection]] = {}
    for key, items in payload.items():
        dets[int(key)] = [
            Detection(BBox(*item["bbox"]), item["conf"], item["class_id"]) for item in items
        ]
    return dets


def read_scene_frame(scene: dict[str, Any], rel_frame: int) -> tuple[bool, Any]:
    clip_len = scene["clip_end"] - scene["clip_start"]
    if rel_frame < 0 or rel_frame >= clip_len:
        return False, None
    cap = cv2.VideoCapture(str(_repo_path(scene["source_video"])))
    cap.set(cv2.CAP_PROP_POS_FRAMES, scene["clip_start"] + rel_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False, None
    scale = float(scene.get("scale", 1.0))
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return True, frame


def encode_frame_jpeg(scene_id: str, rel_frame: int) -> bytes:
    scene = get_scene(scene_id)
    ok, frame = read_scene_frame(scene, rel_frame)
    if not ok:
        raise ValueError("Frame is outside the scene range.")
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("Could not encode frame.")
    return encoded.tobytes()


def scene_candidates(scene_id: str, rel_frame: int) -> dict[str, Any]:
    scene = get_scene(scene_id)
    start, end = scene["clip_start"], scene["clip_end"]
    if rel_frame < 0 or rel_frame >= end - start:
        raise ValueError("Frame is outside the scene range.")
    dets = load_detections(scene)
    supported = set(scene.get("supported_classes", [0, 1, 2, 3, 5, 7]))
    return {
        "scene_id": scene_id,
        "frame": rel_frame,
        "absolute_frame": start + rel_frame,
        "candidates": build_candidates(dets, start + rel_frame, supported),
    }


def _apply_runtime_cfg(cfg: dict, target_class_id: int) -> dict:
    cfg = dict(cfg)
    lumen_cfg = dict(cfg.get("lumen", {}))
    lumen_cfg.update(
        {
            "use_exit_zone": True,
            "latent_enter_frames": 2,
            "latent_max_frames": 70,
            "target_classes": [target_class_id],
            "pedestrian_only": target_class_id == 0,
        }
    )
    cfg["lumen"] = lumen_cfg
    return cfg


def _beat_for_frame(in_oc: bool, anchor: BBox | None, visible: bool, beat_sm: StableBeat) -> RealBeat:
    if in_oc:
        return beat_sm.update(RealBeat.OCCLUDED)
    if anchor and visible:
        return beat_sm.update(RealBeat.VISIBLE)
    return beat_sm.update(RealBeat.VISIBLE)


def render_selected_target(
    scene_id: str,
    candidate_id: str,
    selection_frame: int,
    output_dir: Path,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[Path, Path]:
    scene = get_scene(scene_id)
    all_dets = load_detections(scene)
    selected_abs, selected_idx = (int(part) for part in candidate_id.split(":", 1))
    full_frames = []
    for i in range(scene["clip_end"] - scene["clip_start"]):
        ok, frame = read_scene_frame(scene, i)
        if not ok:
            break
        full_frames.append(frame)
    if len(full_frames) < 10:
        raise ValueError("Could not read enough frames for this scene.")
    frame_h, frame_w = full_frames[0].shape[:2]

    def frame_provider(abs_frame: int):
        rel = abs_frame - scene["clip_start"]
        if 0 <= rel < len(full_frames):
            return full_frames[rel]
        return None

    clip = build_selectable_target_clip(
        all_dets,
        scene["clip_start"],
        scene["clip_end"],
        selected_abs,
        selected_idx,
        float(frame_w),
        frame_provider=frame_provider,
    )
    frames = full_frames[: clip.end - clip.start]
    if len(frames) < 10:
        raise ValueError("Could not read enough frames for this scene.")

    num = len(frames)
    raw_dets = {i: all_dets.get(clip.start + i, []) for i in range(num)}
    masked_dets = mask_subject_windows(raw_dets, clip.anchor_path, clip.occlusion_windows, thresh=0.20)
    occluder_dets = {i: [d for d in raw_dets[i] if d.class_id in OCCLUDER_CLASSES] for i in range(num)}
    cfg = _apply_runtime_cfg(load_config("configs/default.yaml"), clip.class_id)

    oc_start = clip.occlusion_windows[0][0] if clip.occlusion_windows else num
    oc_end = clip.occlusion_windows[-1][1] if clip.occlusion_windows else num
    engine = DemoComparisonEngine(
        cfg=cfg,
        target_classes=[clip.class_id],
        anchor_path=clip.anchor_path,
        oc_start=oc_start,
        oc_end=oc_end,
        occlusion_windows=clip.occlusion_windows,
        lock_until_frame=oc_start,
        target_class_id=clip.class_id,
    )
    engine.build(masked_dets, occluder_dets, raw_dets=raw_dets)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{scene_id}_{candidate_id.replace(':', '_')}_persist_split.mp4"
    manifest_path = output_dir / f"{scene_id}_{candidate_id.replace(':', '_')}_manifest.json"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(scene.get("fps", 15)),
        (frame_w * 2 + 8, frame_h + 72),
    )
    if not writer.isOpened():
        raise ValueError("Could not open video writer.")

    beat_sm = StableBeat(hold_frames=4)
    lost_sm = StickyFlag(on_frames=2, off_frames=3)
    ghost_sm = StickyFlag(on_frames=2, off_frames=3)
    smooth = SmoothBBox(alpha=0.40)
    silhouette = SubjectSilhouette()
    ghost_trail: list[tuple[int, int]] = []
    frame_meta: list[dict[str, Any]] = []

    label = class_name(clip.class_id).upper()
    for i, frame in enumerate(frames):
        comp = engine.step(i, masked_dets[i], occluder_dets[i])
        anchor = clip.anchor_path.get(i)
        target_state = (
            clip.target_memory.state_by_frame.get(i, "EXITED")
            if clip.target_memory is not None
            else ("VISIBLE" if i in clip.visible_frames else "PREDICTED")
        )
        visible = bool(
            anchor
            and (
                i in clip.visible_frames
                or target_visible_enough(raw_dets[i], anchor, clip.class_id, 0.14)
            )
        )
        in_window = any(start <= i < end for start, end in clip.occlusion_windows)
        in_oc = bool(anchor and not visible and in_window)
        baseline_lost = lost_sm.update(comp.baseline_lost or in_oc)
        lumen_ghost = ghost_sm.update(comp.lumen_ghost and bool(anchor) and in_oc)
        if in_oc:
            baseline_lost = True
            lumen_ghost = True
        lumen_visual = comp.lumen_visual
        if in_oc and anchor:
            lumen_visual = LumenVisualState(
                exit_zones=list(comp.lumen_visual.exit_zones),
                confidence=clip.confidence_by_frame.get(i, comp.lumen_visual.confidence),
                predicted_path=clip.predicted_paths.get(i, []),
                latent_badge=target_state,
            )
        beat = _beat_for_frame(in_oc, anchor, visible, beat_sm)

        if in_oc and anchor:
            lumen_bb = smooth.update(anchor)
            ghost_trail.append((int(lumen_bb.cx), int(lumen_bb.cy)))
        elif anchor and visible:
            lumen_bb = anchor
            smooth.reset(lumen_bb)
            ghost_trail.clear()
            silhouette.update_from_frame(frame, anchor)
        else:
            lumen_bb = None
            ghost_trail.clear()

        if not anchor and not in_oc:
            caption = "Target exited - standard detection only."
        elif in_oc:
            caption = f"{target_state} - PERSIST-AI maintains target memory."
        elif clip.tracking_quality != "high":
            caption = f"DEGRADED - {label} identity memory active with uncertainty."
        else:
            caption = f"LOCKED - {label} identity memory active."

        rendered = compose_crowd_frame(
            frame,
            beat,
            crowd_dets=[d for d in raw_dets[i] if d.class_id == 0],
            raw_dets=raw_dets[i],
            target_anchor=anchor,
            lumen_target_bb=lumen_bb,
            lumen_ghost=lumen_ghost and lumen_bb is not None,
            ghost_trail=ghost_trail if lumen_ghost else None,
            silhouette=silhouette if lumen_ghost else None,
            occluder_dets=occluder_dets[i] if lumen_ghost else None,
            lumen_visual=lumen_visual if lumen_ghost else None,
            frame_idx=i,
            total_frames=num,
            header="TRY PERSIST-AI | PERSIST-AI (left) vs Raw YOLO (right)",
            beat_label=caption,
            target_label="",
            ghost_label="",
            layout="split",
        )
        writer.write(rendered)
        frame_meta.append(
            {
                "frame": i,
                "phase": beat.name,
                "occlusion": in_oc,
                "ghost_drawn": bool(lumen_ghost and lumen_bb is not None),
                "selected_identity_visible": visible,
                "target_state": target_state,
                "confidence": round(clip.confidence_by_frame.get(i, 0.0), 3),
                "prediction_mode": (clip.prediction_mode_by_frame or {}).get(i, "uncertain"),
                "uncertainty_radius": round((clip.uncertainty_radius_by_frame or {}).get(i, 0.0), 3),
                "predicted_path": clip.predicted_paths.get(i, []) if in_oc else [],
                "anchor": anchor.as_xyxy() if anchor else None,
            }
        )
        if progress_cb:
            progress_cb((i + 1) / num)
    writer.release()

    save_json(
        manifest_path,
        {
            "scene_id": scene_id,
            "candidate_id": candidate_id,
            "target_class_id": clip.class_id,
            "target_class_name": class_name(clip.class_id),
            "selection_frame": selection_frame,
            "clip": [clip.start, clip.end],
            "occlusion_windows": clip.occlusion_windows,
            "visible_frames": sorted(clip.visible_frames),
            "tracking_quality": clip.tracking_quality,
            "failure_mode": clip.failure_mode,
            "identity_switch_guard": clip.identity_switch_guard,
            "target_states": [
                frame.get("target_state", "EXITED") for frame in frame_meta
            ],
            "target_memory": {
                "target_id": clip.target_memory.target_id if clip.target_memory else candidate_id,
                "stable_width": round(clip.target_memory.stable_width, 3) if clip.target_memory else None,
                "stable_height": round(clip.target_memory.stable_height, 3) if clip.target_memory else None,
                "aspect_ratio": round(clip.target_memory.aspect_ratio, 4) if clip.target_memory else None,
                "foot_y": round(clip.target_memory.foot_y, 3) if clip.target_memory else None,
            },
            "frames": num,
            "fps": scene.get("fps", 15),
            "frame_meta": frame_meta,
        },
    )
    return video_path, manifest_path


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def start_render_job(scene_id: str, candidate_id: str, selection_frame: int) -> RenderJob:
    scene = get_scene(scene_id)
    job_id = uuid.uuid4().hex[:12]
    job = RenderJob(job_id=job_id, scene_id=scene_id)
    with JOBS_LOCK:
        JOBS[job_id] = job

    def run() -> None:
        try:
            job.status = "running"
            job.message = "Building PERSIST-AI comparison"
            output_dir = _repo_path(scene["output_dir"]) / job_id

            def update_progress(progress: float) -> None:
                job.progress = round(progress, 3)

            video_path, manifest_path = render_selected_target(
                scene_id,
                candidate_id,
                selection_frame,
                output_dir,
                progress_cb=update_progress,
            )
            job.video_path = str(video_path)
            job.manifest_path = str(manifest_path)
            job.progress = 1.0
            job.status = "complete"
            job.message = "Complete"
        except Exception as exc:  # noqa: BLE001 - shown to local demo user
            job.status = "failed"
            job.error = str(exc)
            job.message = str(exc)

    threading.Thread(target=run, daemon=True).start()
    return job


def get_job(job_id: str) -> RenderJob:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise KeyError(f"Unknown job: {job_id}")
        return JOBS[job_id]
