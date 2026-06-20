from __future__ import annotations

from lumen.eval.events import OcclusionEvent


def compute_tcuo(
    events: list[OcclusionEvent],
    pred_tracks: dict[int, dict[int, int]],
) -> float:
    """Track Continuity Under Occlusion."""
    if not events:
        return 0.0
    scores = []
    for ev in events:
        tid_pred = pred_tracks.get(ev.video_id, {})
        ids_in_range = set()
        for f in range(ev.t_start, ev.t_end + 1):
            if f in tid_pred:
                ids_in_range.add(tid_pred[f])
        scores.append(1.0 if len(ids_in_range) == 1 and len(ids_in_range) > 0 else 0.0)
    return sum(scores) / len(scores)


def continuity_for_event(
    ev: OcclusionEvent, frame_to_tid: dict[int, int], start_tid: int | None
) -> bool:
    if start_tid is None:
        return False
    for f in range(ev.t_start, ev.t_end + 1):
        if frame_to_tid.get(f) != start_tid:
            return False
    return frame_to_tid.get(ev.t_end) == start_tid
