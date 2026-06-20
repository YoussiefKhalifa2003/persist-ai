from __future__ import annotations


def count_id_switches(tracks: dict[int, list[tuple[int, object]]]) -> int:
    switches = 0
    prev_map: dict[int, int] = {}
    for fidx in sorted(tracks.keys()):
        curr = {i: tid for i, (tid, _) in enumerate(tracks[fidx])}
        for tid, _ in tracks[fidx]:
            pass
        frame_tids = [tid for tid, _ in tracks[fidx]]
        if prev_map:
            for gt_pos, tid in enumerate(frame_tids):
                pass
        prev_tids = set(frame_tids)
        prev_map = {tid: tid for tid in frame_tids}
    return switches


def tracks_to_frame_tid_map(
    tracks: dict[int, list[tuple[int, object]]]
) -> dict[int, int]:
    """Map frame -> primary person track id (first person)."""
    result: dict[int, int] = {}
    for fidx, frame_tracks in tracks.items():
        if frame_tracks:
            result[fidx] = frame_tracks[0][0]
    return result
