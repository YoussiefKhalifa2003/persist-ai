"""Build LinkedIn-grade PERSIST-AI visuals."""

from __future__ import annotations

from pathlib import Path

from lumen.viz.linkedin_compositor import build_assets

OUT_DIR = Path("results/linkedin")


def main() -> None:
    written = build_assets(OUT_DIR)
    print(f"LinkedIn assets written to {OUT_DIR}")
    for name in sorted(written):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
