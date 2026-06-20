from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrackState(str, Enum):
    ACTIVE = "active"
    LATENT = "latent"
    RECOVERED = "recovered"
    TERMINATED = "terminated"


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def as_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_xyxy(cls, box: list[float]) -> "BBox":
        return cls(box[0], box[1], box[2], box[3])


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    embedding: Optional[list[float]] = None


@dataclass
class TrackOutput:
    track_id: int
    bbox: BBox
    state: TrackState
    confidence: float
    is_ghost: bool = False
    exit_zones: list[tuple[BBox, float]] = field(default_factory=list)
    predicted_path: list[tuple[float, float]] = field(default_factory=list)
    occluder_unknown: bool = False


@dataclass
class OcclusionEvent:
    video_id: str
    track_id: int
    t_start: int
    t_end: int
    gap_frames: int
    has_vehicle_occluder: bool = False


@dataclass
class FrameGT:
    frame_idx: int
    objects: list[dict]


@dataclass
class LatentTrackState:
    track_id: int
    state: TrackState
    bbox: BBox
    velocity: tuple[float, float]
    confidence: float
    latent_frames: int
    occluder_id: Optional[int] = None
    occluder_bbox: Optional[BBox] = None
    occluder_unknown: bool = False
    embedding: Optional[list[float]] = None
    exit_zones: list[tuple[BBox, float]] = field(default_factory=list)
    history: list[tuple[float, float]] = field(default_factory=list)
