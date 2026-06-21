"""Run the local Try PERSIST-AI web app."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    uvicorn.run("lumen.web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
