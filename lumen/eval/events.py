from __future__ import annotations

from pathlib import Path

from lumen.types import OcclusionEvent


def extract_occlusion_events(
    gt_by_frame: dict,
    video_id: str,
    min_gap: int = 10,
    person_only: bool = True,
) -> list[OcclusionEvent]:
    """Extract visibility gap events from per-frame GT."""
    track_timeline: dict[int, dict[int, object]] = {}

    for fidx, frame_gt in gt_by_frame.items():
        for obj in frame_gt.objects:
            cat = obj.get("category", "")
            if person_only and cat not in ("person", "pedestrian", "Person"):
                continue
            tid = obj["track_id"]
            track_timeline.setdefault(tid, {})[fidx] = obj["bbox"]

    events: list[OcclusionEvent] = []
    for tid, timeline in track_timeline.items():
        frames = sorted(timeline.keys())
        if len(frames) < 2:
            continue
        i = 0
        while i < len(frames) - 1:
            t_a = frames[i]
            j = i + 1
            while j < len(frames) and frames[j] == frames[j - 1] + 1:
                j += 1
            if j >= len(frames):
                break
            t_b = frames[j]
            gap = t_b - t_a - 1
            if gap >= min_gap:
                events.append(
                    OcclusionEvent(
                        video_id=video_id,
                        track_id=tid,
                        t_start=t_a,
                        t_end=t_b,
                        gap_frames=gap,
                    )
                )
            i = j
    return events
