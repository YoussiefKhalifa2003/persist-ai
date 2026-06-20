from __future__ import annotations

import math

from lumen.types import LatentTrackState, TrackState


class LatentTrack:
    """Persistent state for a single track through occlusion."""

    def __init__(
        self,
        track_id: int,
        confidence_decay_lambda: float = 0.05,
        min_confidence: float = 0.15,
        max_latent_frames: int = 45,
    ):
        self.track_id = track_id
        self.confidence_decay_lambda = confidence_decay_lambda
        self.min_confidence = min_confidence
        self.max_latent_frames = max_latent_frames
        self.state = LatentTrackState(
            track_id=track_id,
            state=TrackState.ACTIVE,
            bbox=None,  # type: ignore[arg-type]
            velocity=(0.0, 0.0),
            confidence=1.0,
            latent_frames=0,
        )
        self.miss_streak = 0

    def mark_active(self, confidence: float = 1.0) -> None:
        self.state.state = TrackState.ACTIVE
        self.state.confidence = confidence
        self.state.latent_frames = 0
        self.miss_streak = 0

    def mark_missed(self) -> None:
        self.miss_streak += 1

    def enter_latent(self) -> None:
        self.state.state = TrackState.LATENT
        self.state.latent_frames = 0

    def decay_confidence(self) -> None:
        self.state.confidence *= math.exp(-self.confidence_decay_lambda)
        self.state.latent_frames += 1

    def should_terminate(self) -> bool:
        if self.state.confidence < self.min_confidence:
            return True
        if self.state.latent_frames >= self.max_latent_frames:
            return True
        return False

    def mark_recovered(self) -> None:
        self.state.state = TrackState.RECOVERED

    def mark_terminated(self) -> None:
        self.state.state = TrackState.TERMINATED
