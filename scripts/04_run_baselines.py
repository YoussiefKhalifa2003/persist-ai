"""Run all baselines on cached detections."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-cache", default="results/detections_bdd_val.npz")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    for method in ("bytetrack", "botsort", "lumen"):
        print(f"Run: python -m lumen track --method {method} --config {args.config}")
    cache = Path(args.detections_cache)
    if not cache.exists():
        print(f"Cache not found: {cache} — run detect first")


if __name__ == "__main__":
    main()
