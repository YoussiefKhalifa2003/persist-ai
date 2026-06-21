from __future__ import annotations

from lumen.eval.events import OcclusionEvent


def compute_orr(
    events: list[OcclusionEvent],
    pred_tracks: dict[int, dict[int, int]],
    tolerance_frames: int = 3,
) -> float:
    """Occlusion Recovery Rate."""
    if not events:
        return 0.0
    scores = []
    for ev in events:
        tid_map = pred_tracks.get(ev.video_id, {})
        start_tid = tid_map.get(ev.t_start)
        if start_tid is None:
            scores.append(0.0)
            continue
        recovered = False
        for f in range(ev.t_end, ev.t_end + tolerance_frames + 1):
            if tid_map.get(f) == start_tid:
                recovered = True
                break
        # Also require continuity through gap
        continuous = all(tid_map.get(f) in (None, start_tid) or tid_map.get(f) == start_tid
                        for f in range(ev.t_start, ev.t_end + 1))
        gap_ok = all(tid_map.get(f) == start_tid for f in range(ev.t_start, ev.t_end + 1) if f in tid_map)
        scores.append(1.0 if recovered and gap_ok else 0.0)
    return sum(scores) / len(scores)
