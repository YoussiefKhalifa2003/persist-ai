from lumen.core.latent_track import LatentTrack
from lumen.types import BBox, TrackState


def test_active_to_latent():
    lt = LatentTrack(1, max_latent_frames=10)
    lt.state.bbox = BBox(0, 0, 10, 10)
    lt.mark_missed()
    lt.mark_missed()
    lt.enter_latent()
    assert lt.state.state == TrackState.LATENT


def test_confidence_decay_terminates():
    lt = LatentTrack(1, confidence_decay_lambda=0.5, min_confidence=0.15, max_latent_frames=100)
    lt.enter_latent()
    for _ in range(20):
        lt.decay_confidence()
    assert lt.should_terminate()


def test_recovered_state():
    lt = LatentTrack(1)
    lt.mark_recovered()
    assert lt.state.state == TrackState.RECOVERED


def test_miss_streak_reset_on_active():
    lt = LatentTrack(1)
    lt.mark_missed()
    lt.mark_active()
    assert lt.miss_streak == 0
