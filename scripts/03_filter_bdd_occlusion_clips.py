"""Filter BDD MOT clips with occlusion events and rank hero clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumen.data.bdd_mot_parser import find_bdd_label_files, parse_bdd_mot_label
from lumen.eval.events import extract_occlusion_events
from lumen.utils.io import ensure_dir, save_json


def score_event(ev, gt_by_frame, has_vehicle: bool = False) -> float:
    score = 0.0
    score += min(ev.gap_frames / 10, 3) * 3
    if has_vehicle:
        score += 3
    if ev.t_end - ev.t_start >= 8:
        score += 2
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", default="data/raw/bdd100k/labels/box_track_20")
    parser.add_argument("--videos-root", default="data/raw/bdd100k/videos/val")
    parser.add_argument("--min-gap", type=int, default=10)
    parser.add_argument("--rank-hero", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output", default="data/manifests/bdd_occlusion_clips.json")
    parser.add_argument("--hero-output", default="data/manifests/hero_clips.json")
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    ensure_dir(Path(args.output).parent)
    all_clips = []
    hero_candidates = []

    for label_path in find_bdd_label_files(labels_root):
        gt = parse_bdd_mot_label(label_path)
        video_id = label_path.stem
        events = extract_occlusion_events(gt, video_id, min_gap=args.min_gap)
        for ev in events:
            clip = {
                "clip_id": f"bdd_{video_id}_{ev.track_id}",
                "video_id": video_id,
                "video_path": str(Path(args.videos_root) / f"{video_id}.mp4"),
                "event": {
                    "track_id": ev.track_id,
                    "t_start": ev.t_start,
                    "t_end": ev.t_end,
                    "gap_frames": ev.gap_frames,
                },
                "score": score_event(ev, gt),
            }
            all_clips.append(clip)
            hero_candidates.append(clip)

    save_json(args.output, all_clips)
    print(f"Wrote {len(all_clips)} occlusion clips to {args.output}")

    if args.rank_hero:
        hero_candidates.sort(key=lambda c: c["score"], reverse=True)
        heroes = []
        for c in hero_candidates[: args.top]:
            ev = c["event"]
            heroes.append(
                {
                    **c,
                    "demo_window": {
                        "start_frame": max(0, ev["t_start"] - 15),
                        "end_frame": ev["t_end"] + 15,
                    },
                    "baseline_lost_id": True,
                    "notes": "auto-ranked BDD occlusion clip",
                }
            )
        save_json(args.hero_output, heroes)
        print(f"Wrote {len(heroes)} hero clips to {args.hero_output}")


if __name__ == "__main__":
    main()
