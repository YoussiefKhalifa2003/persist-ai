"""Create synthetic occlusion demo video for smoke tests when datasets unavailable."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main():
    out = Path("data/raw/synthetic/occlusion_demo.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps, n = 640, 480, 30, 150
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    px, py = 50, 300
    vx = 4
    car_x = 200
    for i in range(n):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 220
        cv2.rectangle(frame, (0, 350), (w, h), (100, 100, 100), -1)
        px += vx
        if 40 < i < 100:
            car_x = 250
        else:
            car_x += 2
        cv2.rectangle(frame, (int(car_x), 280), (int(car_x) + 180, 380), (50, 50, 200), -1)
        if not (int(car_x) < px < int(car_x) + 180):
            cv2.circle(frame, (int(px), int(py)), 15, (0, 180, 0), -1)
        cv2.putText(frame, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()
    print(f"Created {out}")


if __name__ == "__main__":
    main()
