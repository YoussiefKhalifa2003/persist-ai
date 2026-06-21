"""Validate curated interactive demo scenes before launching the web UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.web.interactive_demo import load_scenes, validate_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate interactive scene assets.")
    parser.add_argument(
        "--config",
        default="configs/interactive_scenes.json",
        help="Scene registry JSON path.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    scenes = load_scenes(Path(args.config))
    report: list[dict] = []
    ok = True
    for scene in scenes:
        result = validate_scene(scene)
        row = {
            "scene_id": scene["scene_id"],
            "name": scene["name"],
            **result,
        }
        report.append(row)
        if not result["ready"]:
            ok = False

    if args.json:
        print(json.dumps({"ok": ok, "scenes": report}, indent=2))
    else:
        for row in report:
            status = "READY" if row["ready"] else "NOT READY"
            print(f"[{status}] {row['scene_id']} — {row['name']}")
            if row.get("resolution"):
                w, h = row["resolution"]
                print(f"  resolution: {w}x{h} | frames: {row.get('frame_count', 0)} | fps: {row.get('fps', 0)}")
            for issue in row.get("issues", []):
                print(f"  - {issue}")
        print()
        if ok:
            print("All scenes ready.")
        else:
            print("Some scenes are missing assets. Run ingest or build scripts, then retry.")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
