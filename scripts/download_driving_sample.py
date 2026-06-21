"""Download public driving/traffic video for VIDEO 3 (no registration)."""

from __future__ import annotations

from pathlib import Path

import httpx


PUBLIC_SAMPLES = [
    {
        "url": "https://raw.githubusercontent.com/Qpu523/High-density-Intersection-Dataset/3e808ab6db80f6a9262a8eb99d99264aee201447/Datashare/1.mp4",
        "filename": "intersection_uav_1.mp4",
        "desc": "HDI drone intersection (dense cars + trucks)",
    },
    {
        "url": "https://eu-central-1.linodeobjects.com/savant-data/demo/leeds_1080p.mp4",
        "filename": "leeds_1080p.mp4",
        "desc": "Savant intersection traffic demo (1080p)",
    },
]


def download_traffic_sample(local_dir: Path | None = None) -> Path:
    local_dir = local_dir or Path("data/raw/driving")
    local_dir.mkdir(parents=True, exist_ok=True)

    for cached in local_dir.glob("*.mp4"):
        if cached.stat().st_size > 500_000:
            print(f"Using cached traffic video: {cached}")
            return cached

    for sample in PUBLIC_SAMPLES:
        dest = local_dir / sample["filename"]
        print(f"Downloading {sample['desc']} ...")
        try:
            with httpx.stream("GET", sample["url"], follow_redirects=True, timeout=120.0) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            if dest.stat().st_size > 500_000:
                print(f"Traffic video ready: {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
                return dest
        except Exception as exc:
            print(f"  failed: {exc}")
            if dest.exists():
                dest.unlink(missing_ok=True)

    raise SystemExit("Could not download a public traffic sample. Place an MP4 in data/raw/driving/")


def download_baton_dashcam(local_dir: Path | None = None) -> Path:
    """Alias — prefers public traffic samples over gated BATON."""
    return download_traffic_sample(local_dir)


if __name__ == "__main__":
    download_traffic_sample()
