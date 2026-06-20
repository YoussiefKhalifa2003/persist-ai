from lumen.eval.events import OcclusionEvent
from lumen.eval.metrics_tcuo import compute_tcuo


def test_tcuo_perfect_continuity():
    events = [OcclusionEvent("v1", 1, 10, 20, 10)]
    pred = {"v1": {f: 1 for f in range(10, 21)}}
    assert compute_tcuo(events, pred) == 1.0


def test_tcuo_broken_continuity():
    events = [OcclusionEvent("v1", 1, 10, 20, 10)]
    pred = {"v1": {10: 1, 15: 2, 20: 1}}
    assert compute_tcuo(events, pred) == 0.0


def test_tcuo_empty_events():
    assert compute_tcuo([], {}) == 0.0
