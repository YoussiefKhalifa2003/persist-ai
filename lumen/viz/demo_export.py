from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def export_video_ffmpeg(frames_dir: Path, output: Path, fps: int = 30) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output),
    ]
    subprocess.run(cmd, check=False, capture_output=True)
    return output.exists()
