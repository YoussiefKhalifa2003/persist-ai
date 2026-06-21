from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
