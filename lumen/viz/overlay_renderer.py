from __future__ import annotations

import cv2
import numpy as np

from lumen.types import TrackOutput, TrackState


def render_frame(
    frame: np.ndarray,
    tracks: list[TrackOutput],
    title: str = "PERSIST-AI",
    baseline_lost: bool = False,
) -> np.ndarray:
    vis = frame.copy()
    for t in tracks:
        bb = t.bbox
        color = (0, 255, 0) if not t.is_ghost else (255, 255, 0)
        thickness = 2 if not t.is_ghost else 1
        pt1 = (int(bb.x1), int(bb.y1))
        pt2 = (int(bb.x2), int(bb.y2))
        if t.is_ghost:
            for i in range(pt1[0], pt2[0], 10):
                cv2.line(vis, (i, pt1[1]), (min(i + 5, pt2[0]), pt1[1]), color, thickness)
            for i in range(pt1[0], pt2[0], 10):
                cv2.line(vis, (i, pt2[1]), (min(i + 5, pt2[0]), pt2[1]), color, thickness)
        else:
            cv2.rectangle(vis, pt1, pt2, color, thickness)

        label = f"{'GHOST ' if t.is_ghost else ''}#{t.track_id}"
        cv2.putText(
            vis, label, (pt1[0], max(pt1[1] - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

        for zone, _ in t.exit_zones:
            overlay = vis.copy()
            cv2.rectangle(
                overlay,
                (int(zone.x1), int(zone.y1)),
                (int(zone.x2), int(zone.y2)),
                (0, 255, 255),
                -1,
            )
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        if t.predicted_path and len(t.predicted_path) > 1:
            pts = np.array(t.predicted_path, dtype=np.int32)
            for i in range(len(pts) - 1):
                cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (255, 200, 0), 1)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if baseline_lost:
        cv2.putText(vis, "LOST", (vis.shape[1] - 80, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if tracks and tracks[0].is_ghost:
        conf = tracks[0].confidence
        bar_w = int(200 * conf)
        cv2.rectangle(vis, (10, vis.shape[0] - 20), (210, vis.shape[0] - 8), (80, 80, 80), -1)
        cv2.rectangle(vis, (10, vis.shape[0] - 20), (10 + bar_w, vis.shape[0] - 8), (0, int(255 * conf), 255), -1)

    return vis
