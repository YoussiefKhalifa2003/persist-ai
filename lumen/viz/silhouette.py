"""Subject silhouette cache for ghost rendering during occlusion."""

from __future__ import annotations

import cv2
import numpy as np

from lumen.types import BBox


class SubjectSilhouette:
    """Cache last visible appearance; overlay tinted silhouette at ghost bbox."""

    def __init__(self):
        self.crop: np.ndarray | None = None
        self.mask: np.ndarray | None = None

    def clear(self) -> None:
        self.crop = None
        self.mask = None

    def update_from_frame(self, frame: np.ndarray, bb: BBox) -> None:
        h, w = frame.shape[:2]
        x1 = max(0, int(bb.x1))
        y1 = max(0, int(bb.y1))
        x2 = min(w, int(bb.x2))
        y2 = min(h, int(bb.y2))
        if x2 - x1 < 12 or y2 - y1 < 20:
            return
        crop = frame[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), np.uint8)
        margin_x = max(2, cw // 12)
        margin_y = max(2, ch // 16)
        rect = (margin_x, margin_y, cw - 2 * margin_x, ch - 2 * margin_y)
        if rect[2] <= 0 or rect[3] <= 0:
            return
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        gc_mask = np.zeros((ch, cw), np.uint8)
        try:
            cv2.grabCut(crop, gc_mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            cv2.ellipse(
                mask,
                (cw // 2, ch // 2),
                (cw // 2 - 4, ch // 2 - 4),
                0,
                0,
                360,
                255,
                -1,
            )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.crop = crop
        self.mask = mask

    def draw_ghost(
        self,
        vis: np.ndarray,
        bb: BBox,
        tint: tuple[int, int, int] = (0, 220, 255),
        alpha: float = 0.52,
    ) -> None:
        if self.crop is None or self.mask is None:
            self._draw_fallback_silhouette(vis, bb, tint, alpha)
            return
        h, w = vis.shape[:2]
        x1 = max(0, int(bb.x1))
        y1 = max(0, int(bb.y1))
        x2 = min(w, int(bb.x2))
        y2 = min(h, int(bb.y2))
        tw, th = x2 - x1, y2 - y1
        if tw < 8 or th < 12:
            return
        crop = cv2.resize(self.crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(self.mask, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask_f = (mask.astype(np.float32) / 255.0) * alpha
        region = vis[y1:y2, x1:x2]
        tint_layer = np.full_like(region, tint, dtype=np.uint8)
        for c in range(3):
            region[:, :, c] = (
                region[:, :, c] * (1 - mask_f) + tint_layer[:, :, c] * mask_f
            ).astype(np.uint8)
        vis[y1:y2, x1:x2] = region

    @staticmethod
    def _draw_fallback_silhouette(
        vis: np.ndarray,
        bb: BBox,
        tint: tuple[int, int, int],
        alpha: float,
    ) -> None:
        cx, cy = int(bb.cx), int(bb.cy)
        ax, ay = int(bb.w * 0.42), int(bb.h * 0.48)
        overlay = vis.copy()
        cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, tint, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
