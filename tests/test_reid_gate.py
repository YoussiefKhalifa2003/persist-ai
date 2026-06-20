import numpy as np
import pytest

from lumen.core.reid_associator import ReIDAssociator
from lumen.types import BBox, Detection


def test_reid_accepts_similar_embedding():
    assoc = ReIDAssociator(cosine_threshold=0.5, motion_gate_sigma=5.0)
    emb = np.random.randn(32).tolist()
    det = Detection(BBox(100, 100, 120, 140), 0.9, 0, embedding=emb)
    zones = [(BBox(90, 90, 130, 150), 1.0)]
    matched, rle = assoc.try_associate(emb, (110, 120), 10.0, zones, [det])
    assert matched is not None


def test_reid_rejects_far_detection():
    assoc = ReIDAssociator(cosine_threshold=0.99, motion_gate_sigma=1.0)
    emb = [1.0] * 32
    det = Detection(BBox(500, 500, 520, 540), 0.9, 0, embedding=[-1.0] * 32)
    zones = [(BBox(90, 90, 130, 150), 1.0)]
    matched, _ = assoc.try_associate(emb, (110, 120), 5.0, zones, [det])
    assert matched is None
