from __future__ import annotations

import numpy as np

from lumen.core.exit_zone import compute_exit_zones, point_in_zone
from lumen.types import BBox, Detection
from lumen.utils.geometry import cosine_similarity, l2_distance


class ReIDAssociator:
    """Spatial + appearance gate for re-association after occlusion."""

    def __init__(
        self,
        cosine_threshold: float = 0.45,
        motion_gate_sigma: float = 3.0,
        use_reid: bool = True,
    ):
        self.cosine_threshold = cosine_threshold
        self.motion_gate_sigma = motion_gate_sigma
        self.use_reid = use_reid

    def try_associate(
        self,
        latent_embedding: list[float] | None,
        predicted_center: tuple[float, float],
        position_uncertainty: float,
        exit_zones: list[tuple[BBox, float]],
        candidates: list[Detection],
    ) -> tuple[Detection | None, float]:
        best_det = None
        best_score = -1.0
        best_rle = float("inf")

        for det in candidates:
            center = (det.bbox.cx, det.bbox.cy)
            in_zone = any(point_in_zone(center, z) for z, _ in exit_zones)
            dist = l2_distance(center, predicted_center)
            motion_ok = dist <= self.motion_gate_sigma * max(position_uncertainty, 20.0)

            if not (in_zone or motion_ok):
                continue

            if self.use_reid and latent_embedding is not None and det.embedding is not None:
                sim = cosine_similarity(
                    np.array(latent_embedding), np.array(det.embedding)
                )
                if sim < self.cosine_threshold:
                    continue
                score = sim
            else:
                score = 1.0 / (1.0 + dist)

            if score > best_score:
                best_score = score
                best_det = det
                best_rle = dist

        return best_det, best_rle
