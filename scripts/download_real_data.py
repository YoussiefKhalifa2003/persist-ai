"""Download real public datasets for PERSIST-AI demos (no manual registration)."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def download_mot17_02() -> Path:
    root = Path("data/raw/mot17")
    snapshot_download(
        repo_id="Lekim89/MOT17",
        repo_type="dataset",
        allow_patterns=["ablation/MOT17-02-FRCNN/**"],
        local_dir=str(root),
    )
    seq = root / "ablation" / "MOT17-02-FRCNN"
    n = len(list((seq / "img1").glob("*.jpg")))
    print(f"MOT17-02 ready: {n} frames at {seq}")
    return seq


def download_mot17_11() -> Path:
    """Crowded scene — good for occlusion."""
    root = Path("data/raw/mot17")
    snapshot_download(
        repo_id="Lekim89/MOT17",
        repo_type="dataset",
        allow_patterns=["ablation/MOT17-11-FRCNN/**"],
        local_dir=str(root),
    )
    seq = root / "ablation" / "MOT17-11-FRCNN"
    n = len(list((seq / "img1").glob("*.jpg")))
    print(f"MOT17-11 ready: {n} frames at {seq}")
    return seq


if __name__ == "__main__":
    download_mot17_02()
    print("Optional: uncomment download_mot17_11() for crowded scene")
