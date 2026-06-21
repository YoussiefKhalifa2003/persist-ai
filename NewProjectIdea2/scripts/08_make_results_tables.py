"""Aggregate results tables."""

from pathlib import Path

import pandas as pd


def main():
    tables = Path("results/tables")
    if not tables.exists():
        print("No results yet — run eval first")
        return
    for csv in tables.glob("*.csv"):
        df = pd.read_csv(csv)
        print(f"\n=== {csv.name} ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
