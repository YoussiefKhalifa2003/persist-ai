from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lumen.web.app import app


def test_scenes_endpoint_lists_curated_scene():
    client = TestClient(app)
    res = client.get("/api/scenes")
    assert res.status_code == 200
    scene_ids = [scene["scene_id"] for scene in res.json()["scenes"]]
    assert "video3-sidewalk" in scene_ids


def test_candidates_endpoint_returns_frame_aligned_boxes():
    client = TestClient(app)
    res = client.get("/api/scenes/video3-sidewalk/candidates?frame=24")
    assert res.status_code == 200
    payload = res.json()
    assert payload["frame"] == 24
    assert payload["absolute_frame"] == 79
    assert payload["candidates"]
    assert {"id", "class_id", "class_name", "bbox"} <= set(payload["candidates"][0])


def test_invalid_scene_returns_404():
    client = TestClient(app)
    assert client.get("/api/scenes/not-real/manifest").status_code == 404
