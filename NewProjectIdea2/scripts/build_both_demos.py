"""Build VIDEO 1 (building blocks) and VIDEO 2 (real world) back-to-back."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    py = sys.executable
    run([py, "scripts/build_concept_demo.py"])
    run([py, "scripts/build_real_demo.py"])
    run([py, "scripts/build_street_demo.py"])
    print("\nAll three demos:")
    print("  results/demo_videos/VIDEO1_BUILDING_BLOCKS.mp4")
    print("  results/demo_videos/VIDEO2_REAL_WORLD.mp4")
    print("  results/demo_videos/VIDEO3_VEHICLES.mp4  (street: buses + TARGET woman)")


if __name__ == "__main__":
    main()
