"""Optional OccluBoost via BoxMOT — falls back to BaselineTracker if unavailable."""

from lumen.trackers.baseline_adapter import BaselineTracker


class OccluBoostBaseline(BaselineTracker):
    def __init__(self, config: dict):
        super().__init__(config, method="occluboost")
