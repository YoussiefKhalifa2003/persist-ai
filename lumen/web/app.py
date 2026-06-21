"""FastAPI app for the local Try PERSIST-AI demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lumen.web.interactive_demo import (
    encode_frame_jpeg,
    get_job,
    get_scene,
    load_scenes,
    public_scene,
    scene_candidates,
    start_render_job,
)


class RenderRequest(BaseModel):
    scene_id: str
    candidate_id: str
    selection_frame: int


def create_app() -> FastAPI:
    app = FastAPI(title="Try PERSIST-AI")
    viewer_dir = Path("demo/viewer")
    app.mount("/viewer", StaticFiles(directory=viewer_dir, html=True), name="viewer")
    app.mount("/media", StaticFiles(directory="."), name="media")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(viewer_dir / "index.html")

    @app.get("/api/scenes")
    def scenes() -> dict:
        return {"scenes": [public_scene(scene) for scene in load_scenes()]}

    @app.get("/api/scenes/{scene_id}/manifest")
    def scene_manifest(scene_id: str) -> dict:
        try:
            return public_scene(get_scene(scene_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/scenes/{scene_id}/preview")
    def scene_preview(scene_id: str) -> FileResponse:
        try:
            scene = get_scene(scene_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(scene["preview_video"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Preview video not found.")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/scenes/{scene_id}/known-good")
    def scene_known_good(scene_id: str) -> FileResponse:
        try:
            scene = get_scene(scene_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(scene["known_good_split"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Known-good split video not found.")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/scenes/{scene_id}/frame")
    def scene_frame(scene_id: str, index: int = Query(0, ge=0)) -> Response:
        try:
            return Response(content=encode_frame_jpeg(scene_id, index), media_type="image/jpeg")
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/scenes/{scene_id}/candidates")
    def candidates(scene_id: str, frame: int = Query(0, ge=0)) -> dict:
        try:
            return scene_candidates(scene_id, frame)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/render")
    def render(req: RenderRequest) -> dict:
        try:
            job = start_render_job(req.scene_id, req.candidate_id, req.selection_frame)
            return job.__dict__
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        try:
            return get_job(job_id).__dict__
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/manifest")
    def job_manifest(job_id: str) -> dict:
        try:
            job = get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.status != "complete" or not job.manifest_path:
            raise HTTPException(status_code=409, detail="Render is not complete.")
        return __import__("json").loads(Path(job.manifest_path).read_text(encoding="utf-8"))

    @app.get("/api/jobs/{job_id}/frame")
    def job_frame(job_id: str, index: int = Query(0, ge=0)) -> Response:
        try:
            job = get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.status != "complete" or not job.video_path:
            raise HTTPException(status_code=409, detail="Render is not complete.")
        import cv2

        cap = cv2.VideoCapture(job.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="Frame not found.")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode frame.")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str) -> FileResponse:
        try:
            job = get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.status != "complete" or not job.video_path:
            raise HTTPException(status_code=409, detail="Render is not complete.")
        return FileResponse(job.video_path, media_type="video/mp4")

    return app


app = create_app()
