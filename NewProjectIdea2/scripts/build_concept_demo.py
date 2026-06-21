"""
VIDEO 1: Building Blocks — minimal, linear demo.
Ghost walks smoothly through the van; baseline loses id, PERSIST-AI keeps id 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lumen.core.track_manager import TrackManager
from lumen.trackers.baseline_adapter import BaselineTracker
from lumen.types import BBox, Detection, TrackOutput, TrackState
from lumen.utils.io import load_config
from lumen.viz.concept_compositor import ConceptBeat, compose_concept_frame, compose_title_card


def make_clean_scene(w: int, h: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[: int(h * 0.55)] = (200, 180, 160)
    frame[int(h * 0.55) :] = (70, 70, 70)
    cv2.line(frame, (0, int(h * 0.55)), (w, int(h * 0.55)), (100, 100, 100), 2)
    return frame


def person_bbox(px: int, py: int, person_w: int, person_h: int) -> BBox:
    return BBox(px - person_w // 2, py - person_h, px + person_w // 2, py)


def build_storyboard(
    w: int, h: int, van_x: int
) -> tuple[list[np.ndarray], list[list[Detection]], list[bool], list[BBox]]:
    """Three acts: approach (visible), pass through van (hidden), exit (visible)."""
    van_y1, van_y2 = int(h * 0.42), int(h * 0.72)
    van = BBox(van_x, van_y1, van_x + 200, van_y2)
    person_h, person_w = 100, 50
    py = int(h * 0.62)

    segments = [
        (80, int(van.x1) - 25, 30, True),
        (int(van.x1) - 25, int(van.x2) + 25, 40, False),  # smooth pass behind van
        (int(van.x2) + 25, 880, 30, True),
    ]

    frames: list[np.ndarray] = []
    dets_list: list[list[Detection]] = []
    visible_flags: list[bool] = []
    truth_bboxes: list[BBox] = []

    for px_start, px_end, n_frames, visible in segments:
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            px = int(px_start + (px_end - px_start) * t)
            frame = make_clean_scene(w, h)
            cv2.rectangle(frame, (int(van.x1), int(van.y1)), (int(van.x2), int(van.y2)), (40, 40, 100), -1)
            cv2.putText(
                frame, "VAN", (int(van.x1) + 70, int(van.y1) + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            )

            bb = person_bbox(px, py, person_w, person_h)
            if visible:
                cv2.circle(frame, (px, py - person_h // 2), 28, (0, 180, 255), -1)
                cv2.putText(
                    frame, "PERSON", (px - 38, py - person_h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                )

            frames.append(frame)
            dets_list.append([Detection(bb, 0.95, 0, embedding=[1.0] * 32)] if visible else [])
            visible_flags.append(visible)
            truth_bboxes.append(bb)

    return frames, dets_list, visible_flags, truth_bboxes


def infer_beat(visible: bool, baseline_lost: bool, lumen_ghost: bool, baseline_new_id: bool) -> ConceptBeat:
    if baseline_new_id and visible:
        return ConceptBeat.RETURN
    if lumen_ghost and baseline_lost:
        return ConceptBeat.PERSIST_KEEPS
    if not visible and baseline_lost:
        return ConceptBeat.BASELINE_LOST
    if visible:
        return ConceptBeat.VISIBLE
    return ConceptBeat.HIDDEN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/demo_videos/VIDEO1_BUILDING_BLOCKS.mp4")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    cfg.setdefault("lumen", {})
    cfg["lumen"]["latent_enter_frames"] = 1
    cfg["lumen"]["latent_max_frames"] = 80
    cfg["lumen"]["use_exit_zone"] = False

    w, h = 960, 540
    canvas_w = w * 2 + 8
    canvas_h = h + 72

    frames, all_dets, visible_flags, truth_bboxes = build_storyboard(w, h, van_x=380)

    baseline_map = BaselineTracker(cfg, "bytetrack").track_from_detections(
        frames, {i: d for i, d in enumerate(all_dets)}
    )

    lumen = TrackManager(cfg)
    first_baseline_id: int | None = None
    lumen_id: int | None = None
    rendered: list[np.ndarray] = []
    ghost_trail: list[tuple[int, int]] = []
    occluded_started = False

    for i in range(len(frames)):
        dets = all_dets[i]
        l_out = lumen.update(dets)
        b_raw = baseline_map.get(i, [])
        b_out = [TrackOutput(t, b, TrackState.ACTIVE, 1.0) for t, b in b_raw]

        if lumen_id is None and l_out:
            lumen_id = next((t.track_id for t in l_out if not t.is_ghost), l_out[0].track_id)
        if first_baseline_id is None and b_out:
            first_baseline_id = b_out[0].track_id

        visible = visible_flags[i]
        truth = truth_bboxes[i]
        lumen_ghost = not visible or any(t.is_ghost for t in l_out)

        baseline_has = len(b_out) > 0
        baseline_lost = not baseline_has and not visible
        baseline_new_id = (
            visible
            and baseline_has
            and first_baseline_id is not None
            and b_out[0].track_id != first_baseline_id
        )

        if not visible:
            if not occluded_started:
                ghost_trail = []
                occluded_started = True
            ghost_trail.append((int(truth.cx), int(truth.cy)))
        else:
            occluded_started = False

        ghost_bbox = truth if not visible else None
        lumen_is_ghost = not visible

        if baseline_has:
            tid = b_out[0].track_id
            if first_baseline_id is not None and tid == first_baseline_id:
                baseline_label = "ID 1"
            else:
                baseline_label = "ID 2 (NEW)"
            baseline_lost_flag = False
        else:
            baseline_label = None
            baseline_lost_flag = not visible or baseline_lost

        beat = infer_beat(visible, baseline_lost_flag, lumen_is_ghost, baseline_new_id)

        lumen_display = [TrackOutput(lumen_id or 1, truth, TrackState.ACTIVE, 1.0)] if visible and l_out else []

        rendered.append(
            compose_concept_frame(
                frames[i],
                b_out if baseline_has else [],
                lumen_display,
                beat,
                baseline_lost=baseline_lost_flag,
                ghost_bbox=ghost_bbox,
                ghost_trail=ghost_trail if lumen_is_ghost else None,
                baseline_label=baseline_label,
                lumen_label="ID 1 (ghost)" if lumen_is_ghost else "ID 1",
                lumen_is_ghost=lumen_is_ghost,
            )
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (canvas_w, canvas_h))

    intro = compose_title_card(canvas_w, canvas_h, ConceptBeat.INTRO.value)
    outro = compose_title_card(
        canvas_w, canvas_h, ConceptBeat.OUTRO.value,
        "Detection = what is visible. PERSIST-AI = what still exists.",
    )
    for _ in range(24):
        writer.write(intro)
    for canvas in tqdm(rendered, desc="Concept demo"):
        writer.write(canvas)
    for _ in range(30):
        writer.write(outro)

    writer.release()
    total_sec = (24 + len(rendered) + 30) / args.fps
    print(f"VIDEO 1 saved: {out} ({total_sec:.0f}s)")


if __name__ == "__main__":
    main()
