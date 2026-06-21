from __future__ import annotations

import numpy as np

from lumen.eval.events import OcclusionEvent
from lumen.utils.geometry import l2_distance


def compute_rle(
    events: list[OcclusionEvent],
    pred_centers: dict[int, dict[int, tuple[float, float]]],
    gt_centers: dict[int, dict[int, tuple[float, float]]],
) -> dict[str, float]:
    """Re-entry Localization Error for successful recoveries."""
    errors = []
    for ev in events:
        pred = pred_centers.get(ev.video_id, {}).get(ev.t_end)
        gt = gt_centers.get(ev.video_id, {}).get(ev.t_end)
        if pred is None or gt is None:
            continue
        errors.append(l2_distance(pred, gt))
    if not errors:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    arr = np.array(errors)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }
