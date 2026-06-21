from __future__ import annotations

from pathlib import Path

from lumen.types import FrameGT


def parse_acdc_list(root: str | Path, condition: str) -> list[Path]:
    """List ACDC image paths for fog/rain/night/snow subsets."""
    root = Path(root)
    if not root.exists():
        return []
    cond_dir = root / condition
    if cond_dir.exists():
        return sorted(cond_dir.rglob("*.png")) + sorted(cond_dir.rglob("*.jpg"))
    return sorted(root.rglob(f"*{condition}*/*.png"))


def acdc_frames_to_gt_stub(paths: list[Path]) -> dict[int, FrameGT]:
    """ACDC v1.5 uses track continuity without occlusion GT — frame enumeration only."""
    return {i: FrameGT(frame_idx=i, objects=[]) for i in range(len(paths))}
