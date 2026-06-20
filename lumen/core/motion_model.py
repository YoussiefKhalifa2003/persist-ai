from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

from lumen.types import BBox


class MotionModel:
    """Constant-velocity Kalman filter for track center."""

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 4.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.kf.R *= measurement_noise
        self.kf.Q = np.eye(4) * process_noise
        self.kf.P *= 10.0
        self._base_q = process_noise
        self._latent_steps = 0
        self._last_wh: tuple[float, float] = (40.0, 80.0)

    def init_from_bbox(self, bbox: BBox) -> None:
        self.kf.x = np.array([bbox.cx, bbox.cy, 0.0, 0.0], dtype=float)
        self._last_wh = (bbox.w, bbox.h)
        self._latent_steps = 0

    def update(self, bbox: BBox) -> None:
        self.kf.update(np.array([bbox.cx, bbox.cy], dtype=float))
        self._last_wh = (bbox.w, bbox.h)
        self._latent_steps = 0

    def predict(self, latent: bool = False) -> tuple[float, float, float, float]:
        if latent:
            self._latent_steps += 1
            self.kf.Q = np.eye(4) * self._base_q * (1 + 0.15 * self._latent_steps)
        self.kf.predict()
        cx, cy, vx, vy = self.kf.x
        return float(cx), float(cy), float(vx), float(vy)

    def predicted_bbox(self) -> BBox:
        cx, cy, _, _ = self.predict(latent=False)
        w, h = self._last_wh
        return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def position_uncertainty(self) -> float:
        return float(np.sqrt(self.kf.P[0, 0] + self.kf.P[1, 1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.kf.x[2]), float(self.kf.x[3])
