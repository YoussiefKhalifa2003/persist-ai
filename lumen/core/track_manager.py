from __future__ import annotations

from lumen.core.exit_zone import compute_exit_zones
from lumen.core.latent_track import LatentTrack
from lumen.core.motion_model import MotionModel
from lumen.core.occluder_graph import VEHICLE_CLASSES, find_occluder
from lumen.core.reid_associator import ReIDAssociator
from lumen.types import BBox, Detection, LatentTrackState, TrackOutput, TrackState


PERSON_CLASS = 0


class TrackManager:
    """FSM maintaining persistent world state through occlusion."""

    def __init__(self, config: dict):
        lumen_cfg = config.get("lumen", {})
        self.latent_enter_frames = lumen_cfg.get("latent_enter_frames", 2)
        self.latent_max_frames = lumen_cfg.get("latent_max_frames", 45)
        self.confidence_decay_lambda = lumen_cfg.get("confidence_decay_lambda", 0.05)
        self.min_confidence = lumen_cfg.get("min_confidence", 0.15)
        self.exit_zone_margin_px = lumen_cfg.get("exit_zone_margin_px", 25)
        self.pedestrian_only = lumen_cfg.get("pedestrian_only", True)
        self.target_classes: set[int] = set(lumen_cfg.get("target_classes", [PERSON_CLASS]))
        self.use_exit_zone = lumen_cfg.get("use_exit_zone", True)
        self.use_reid = lumen_cfg.get("use_reid", True)
        self.use_decay = lumen_cfg.get("use_decay", True)

        self.tracks: dict[int, LatentTrack] = {}
        self.motion_models: dict[int, MotionModel] = {}
        self.reid = ReIDAssociator(
            cosine_threshold=lumen_cfg.get("reid_cosine_threshold", 0.45),
            motion_gate_sigma=lumen_cfg.get("motion_gate_sigma", 3.0),
            use_reid=self.use_reid,
        )
        self._next_id = 1
        self.rle_log: list[float] = []

    def update(
        self,
        detections: list[Detection],
        vehicle_detections: list[Detection] | None = None,
    ) -> list[TrackOutput]:
        vehicle_detections = vehicle_detections or [
            d for d in detections if d.class_id in VEHICLE_CLASSES
        ]
        person_dets = [d for d in detections if d.class_id in self.target_classes]

        matched_ids: set[int] = set()
        outputs: list[TrackOutput] = []

        # Try re-associate latent tracks first
        for tid, lt in list(self.tracks.items()):
            if lt.state.state != TrackState.LATENT:
                continue
            mm = self.motion_models[tid]
            cx, cy, vx, vy = mm.predict(latent=True)
            if self.use_decay:
                lt.decay_confidence()
            if lt.should_terminate():
                lt.mark_terminated()
                continue

            zones = lt.state.exit_zones
            if not self.use_exit_zone:
                zones = [(BBox(cx - 50, cy - 50, cx + 50, cy + 50), 1.0)]

            det, rle = self.reid.try_associate(
                lt.state.embedding,
                (cx, cy),
                mm.position_uncertainty(),
                [(z, w) for z, w in zones],
                person_dets,
            )
            if det is not None:
                lt.state.bbox = det.bbox
                lt.state.velocity = (vx, vy)
                mm.update(det.bbox)
                lt.mark_recovered()
                lt.mark_active(det.confidence)
                lt.state.embedding = det.embedding
                matched_ids.add(tid)
                self.rle_log.append(rle)
                outputs.append(
                    TrackOutput(
                        track_id=tid,
                        bbox=det.bbox,
                        state=TrackState.RECOVERED,
                        confidence=lt.state.confidence,
                        is_ghost=False,
                    )
                )
                person_dets = [d for d in person_dets if d is not det]
            else:
                ghost = BBox(
                    cx - lt.state.bbox.w / 2,
                    cy - lt.state.bbox.h / 2,
                    cx + lt.state.bbox.w / 2,
                    cy + lt.state.bbox.h / 2,
                )
                lt.state.history.append((cx, cy))
                outputs.append(
                    TrackOutput(
                        track_id=tid,
                        bbox=ghost,
                        state=TrackState.LATENT,
                        confidence=lt.state.confidence,
                        is_ghost=True,
                        exit_zones=lt.state.exit_zones,
                        predicted_path=list(lt.state.history[-15:]),
                        occluder_unknown=lt.state.occluder_unknown,
                    )
                )

        # Match active tracks to detections (greedy IoU)
        active_ids = [
            tid
            for tid, lt in self.tracks.items()
            if lt.state.state in (TrackState.ACTIVE, TrackState.RECOVERED)
            and tid not in matched_ids
        ]
        used_dets: set[int] = set()
        for tid in active_ids:
            lt = self.tracks[tid]
            mm = self.motion_models[tid]
            best_iou = 0.0
            best_det = None
            best_idx = -1
            for i, det in enumerate(person_dets):
                if i in used_dets:
                    continue
                score = self._iou(lt.state.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_det = det
                    best_idx = i
            if best_det is not None and best_iou > 0.1:
                used_dets.add(best_idx)
                lt.state.bbox = best_det.bbox
                mm.update(best_det.bbox)
                lt.mark_active(best_det.confidence)
                lt.state.embedding = best_det.embedding
                matched_ids.add(tid)
                outputs.append(
                    TrackOutput(
                        track_id=tid,
                        bbox=best_det.bbox,
                        state=TrackState.ACTIVE,
                        confidence=lt.state.confidence,
                    )
                )
            else:
                lt.mark_missed()
                if lt.miss_streak >= self.latent_enter_frames:
                    self._enter_latent(tid, lt, vehicle_detections)
                    cx, cy, vx, vy = mm.predict(latent=True)
                    ghost = BBox(
                        cx - lt.state.bbox.w / 2,
                        cy - lt.state.bbox.h / 2,
                        cx + lt.state.bbox.w / 2,
                        cy + lt.state.bbox.h / 2,
                    )
                    outputs.append(
                        TrackOutput(
                            track_id=tid,
                            bbox=ghost,
                            state=TrackState.LATENT,
                            confidence=lt.state.confidence,
                            is_ghost=True,
                            exit_zones=lt.state.exit_zones,
                            occluder_unknown=lt.state.occluder_unknown,
                        )
                    )

        # Init new tracks
        for i, det in enumerate(person_dets):
            if i in used_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            lt = LatentTrack(
                tid,
                self.confidence_decay_lambda,
                self.min_confidence,
                self.latent_max_frames,
            )
            lt.state.bbox = det.bbox
            lt.state.embedding = det.embedding
            self.tracks[tid] = lt
            mm = MotionModel()
            mm.init_from_bbox(det.bbox)
            self.motion_models[tid] = mm
            outputs.append(
                TrackOutput(
                    track_id=tid,
                    bbox=det.bbox,
                    state=TrackState.ACTIVE,
                    confidence=det.confidence,
                )
            )

        return outputs

    def _enter_latent(
        self, tid: int, lt: LatentTrack, vehicle_dets: list[Detection]
    ) -> None:
        lt.enter_latent()
        oc_id, oc_bbox = find_occluder(lt.state.bbox, vehicle_dets)
        lt.state.occluder_id = oc_id
        lt.state.occluder_bbox = oc_bbox
        lt.state.occluder_unknown = oc_bbox is None
        mm = self.motion_models[tid]
        zones = compute_exit_zones(
            (lt.state.bbox.cx, lt.state.bbox.cy),
            mm.velocity,
            oc_bbox,
            self.exit_zone_margin_px if self.use_exit_zone else 50,
        )
        lt.state.exit_zones = zones

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    def get_active_track_ids(self) -> set[int]:
        return {
            tid
            for tid, lt in self.tracks.items()
            if lt.state.state
            in (TrackState.ACTIVE, TrackState.LATENT, TrackState.RECOVERED)
        }
