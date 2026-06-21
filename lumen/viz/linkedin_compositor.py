"""Cinematic LinkedIn asset compositor — real demo frames + manifest data."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lumen.brand import BRAND, TAGLINE

OUT_W, OUT_H = 1920, 1080
FPS = 24

# BGR for OpenCV layers
BG = (10, 12, 14)
PANEL = (18, 22, 28)
ACCENT = (118, 220, 0)
CYAN = (255, 220, 58)
WHITE = (246, 242, 232)
MUTED = (155, 146, 128)
GREEN = (118, 220, 0)
AMBER = (255, 214, 0)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _canvas() -> np.ndarray:
    img = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    for y in range(OUT_H):
        t = y / OUT_H
        row = np.array(BG, dtype=np.float32) * (1 - t) + np.array((5, 10, 16), dtype=np.float32) * t
        img[y, :] = row.astype(np.uint8)
    return img


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_text_block(
    canvas: np.ndarray,
    title: str,
    subtitle: str,
    x: int,
    y: int,
    title_size: int = 64,
) -> None:
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((x, y), title, font=_font(title_size, bold=True), fill=(235, 245, 250))
    draw.text((x, y + title_size + 12), subtitle, font=_font(28), fill=(170, 180, 190))
    canvas[:] = _pil_to_bgr(pil)


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - tw) // 2)
    y0 = max(0, (nh - th) // 2)
    return resized[y0 : y0 + th, x0 : x0 + tw]


def _split_panels(video: Path, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        blank = np.zeros((540, 480, 3), dtype=np.uint8)
        return blank, blank
    h, w = frame.shape[:2]
    footer = 72
    content_h = h - footer
    gap = 8
    panel_w = (w - gap) // 2
    left = frame[:content_h, :panel_w]
    right = frame[:content_h, panel_w + gap : panel_w + gap + panel_w]
    return left, right


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _interesting_frame(manifest: dict, default: int = 0) -> int:
    meta = manifest.get("frame_meta") or []
    for frame in meta:
        if frame.get("ghost_drawn") and not frame.get("selected_identity_visible", True):
            return int(frame.get("frame", default))
    for frame in meta:
        if frame.get("occlusion"):
            return int(frame.get("frame", default))
    return default


def _ghost_violations(manifest: dict) -> int:
    meta = manifest.get("frame_meta") or []
    return sum(
        1
        for frame in meta
        if frame.get("ghost_drawn") and frame.get("selected_identity_visible")
    )


def hero_comparison_card(video: Path, manifest: dict) -> np.ndarray:
    img = _canvas()
    _draw_text_block(
        img,
        f"{BRAND}",
        TAGLINE,
        72,
        56,
        title_size=72,
    )
    frame_idx = _interesting_frame(manifest, 60)
    left, right = _split_panels(video, frame_idx)
    left_fit = _fit(left, (820, 620))
    right_fit = _fit(right, (820, 620))
    y0, x1, x2 = 190, 72, 1048
    cv2.rectangle(img, (x1 - 4, y0 - 4), (x1 + left_fit.shape[1] + 4, y0 + left_fit.shape[0] + 4), ACCENT, 2)
    cv2.rectangle(img, (x2 - 4, y0 - 4), (x2 + right_fit.shape[1] + 4, y0 + right_fit.shape[0] + 4), MUTED, 2)
    img[y0 : y0 + left_fit.shape[0], x1 : x1 + left_fit.shape[1]] = left_fit
    img[y0 : y0 + right_fit.shape[0], x2 : x2 + right_fit.shape[1]] = right_fit
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((x1, y0 + left_fit.shape[0] + 18), f"{BRAND} — locked identity", font=_font(26, bold=True), fill=(120, 230, 150))
    draw.text((x2, y0 + right_fit.shape[0] + 18), "Raw YOLO — visible only", font=_font(26), fill=(180, 180, 180))
    return _pil_to_bgr(pil)


def identity_lock_diagram() -> np.ndarray:
    img = _canvas()
    _draw_text_block(img, "Identity memory stack", "Selected target stays the same object through occlusion.", 72, 56)
    layers = [
        ("Detect", "YOLO proposals"),
        ("Lock", "Click-to-lock bbox + appearance"),
        ("Latent", "Ghost prediction + uncertainty"),
        ("Recover", "Re-associate on reappearance"),
    ]
    y = 220
    for i, (title, desc) in enumerate(layers):
        x1, x2 = 220, 1700
        color = ACCENT if i % 2 == 0 else CYAN
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y), (x2, y + 130), PANEL, -1)
        cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)
        cv2.rectangle(img, (x1, y), (x2, y + 130), color, 2)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((x1 + 28, y + 28), title, font=_font(40, bold=True), fill=(240, 245, 250))
        draw.text((x1 + 28, y + 78), desc, font=_font(24), fill=(170, 180, 190))
        img = _pil_to_bgr(pil)
        y += 160
    return img


def trajectory_card(video: Path, manifest: dict) -> np.ndarray:
    img = _canvas()
    _draw_text_block(img, "Trajectory + uncertainty", "Prediction path from real render metadata.", 72, 56)
    frame_idx = _interesting_frame(manifest, 40)
    left, _ = _split_panels(video, frame_idx)
    panel = _fit(left, (1180, 700))
    x0, y0 = 72, 180
    img[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    meta = manifest.get("frame_meta") or []
    target = next((f for f in meta if int(f.get("frame", -1)) == frame_idx), None)
    path = target.get("predicted_path") if target else []
    if path:
        pts = [(x0 + int(x), y0 + int(y)) for x, y in path[:24]]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], CYAN, 3, cv2.LINE_AA)
        for pt in pts[::3]:
            cv2.circle(img, pt, 5, AMBER, -1, cv2.LINE_AA)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    conf = target.get("confidence", 0.0) if target else 0.0
    draw.text((1320, 240), f"Frame {frame_idx}", font=_font(30, bold=True), fill=(240, 245, 250))
    draw.text((1320, 300), f"Confidence {conf:.2f}", font=_font(26), fill=(120, 230, 150))
    draw.text((1320, 360), f"Mode {target.get('prediction_mode', 'n/a') if target else 'n/a'}", font=_font(26), fill=(255, 220, 120))
    return _pil_to_bgr(pil)


def multi_scene_audit(manifests: list[tuple[str, dict]]) -> np.ndarray:
    img = _canvas()
    _draw_text_block(img, "Multi-scene audit", "Videos 3–6 — ghost discipline across curated scenes.", 72, 56)
    y = 190
    for label, manifest in manifests:
        meta = manifest.get("frame_meta") or []
        n = max(1, len(meta))
        x1, x2 = 100, 1820
        cv2.rectangle(img, (x1, y), (x2, y + 58), PANEL, -1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((x1 + 16, y + 14), label, font=_font(24, bold=True), fill=(240, 245, 250))
        img = _pil_to_bgr(pil)
        bar_y = y + 40
        for frame in meta:
            fi = int(frame.get("frame", 0))
            px = x1 + int((x2 - x1) * fi / n)
            if frame.get("ghost_drawn"):
                color = AMBER
            elif frame.get("selected_identity_visible", frame.get("phase") == "VISIBLE"):
                color = GREEN
            else:
                color = MUTED
            cv2.line(img, (px, bar_y), (px, bar_y + 14), color, 2)
        violations = _ghost_violations(manifest)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((x2 - 320, y + 16), f"visible-ghost violations: {violations}", font=_font(20), fill=(120, 230, 150) if violations == 0 else (90, 120, 255))
        img = _pil_to_bgr(pil)
        y += 92
    return img


def metrics_card(manifests: list[tuple[str, dict]]) -> np.ndarray:
    img = _canvas()
    _draw_text_block(img, "Validation summary", "Ghost-while-visible should remain zero on curated scenes.", 72, 56)
    y = 220
    for label, manifest in manifests:
        violations = _ghost_violations(manifest)
        ghost_frames = sum(1 for f in (manifest.get("frame_meta") or []) if f.get("ghost_drawn"))
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((120, y), label, font=_font(34, bold=True), fill=(240, 245, 250))
        draw.text((120, y + 46), f"ghost frames: {ghost_frames}", font=_font(26), fill=(170, 180, 190))
        draw.text(
            (120, y + 82),
            f"visible ghost violations: {violations}",
            font=_font(26),
            fill=(120, 230, 150) if violations == 0 else (90, 120, 255),
        )
        img = _pil_to_bgr(pil)
        y += 150
    return img


def _crop_square(img: np.ndarray, size: int) -> np.ndarray:
    return _fit(img, (size, size))


def _crop_portrait(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return _fit(img, (w, h))


def build_assets(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_specs = [
        ("Video 3", Path("results/demo_videos/VIDEO3_PERSIST_SPLIT.mp4"), Path("results/demo_videos/VIDEO3_manifest.json")),
        ("Video 4", Path("results/demo_videos/VIDEO4_PERSIST_SPLIT.mp4"), Path("results/demo_videos/VIDEO4_manifest.json")),
        ("Video 5", Path("results/demo_videos/VIDEO5_PERSIST_SPLIT.mp4"), Path("results/demo_videos/VIDEO5_manifest.json")),
        ("Video 6", Path("results/demo_videos/VIDEO6_PERSIST_SPLIT.mp4"), Path("results/demo_videos/VIDEO6_manifest.json")),
    ]
    available = [(label, video, manifest) for label, video, manifest in video_specs if video.exists()]
    manifests = [(label, _load_manifest(manifest)) for label, _, manifest in available]
    hero_video = available[-1][1] if available else Path()
    hero_manifest = manifests[-1][1] if manifests else {}

    outputs: dict[str, np.ndarray] = {
        "01_hero_comparison.png": hero_comparison_card(hero_video, hero_manifest) if hero_video.exists() else _canvas(),
        "02_identity_lock_diagram.png": identity_lock_diagram(),
        "03_trajectory_uncertainty.png": trajectory_card(hero_video, hero_manifest) if hero_video.exists() else _canvas(),
        "04_multi_scene_audit.png": multi_scene_audit(manifests) if manifests else _canvas(),
        "05_metrics_card.png": metrics_card(manifests) if manifests else _canvas(),
    }
    written: dict[str, Path] = {}
    for name, img in outputs.items():
        path = out_dir / name
        cv2.imwrite(str(path), img)
        written[name] = path

    hero = outputs["01_hero_comparison.png"]
    cv2.imwrite(str(out_dir / "01_hero_comparison_1080sq.png"), _crop_square(hero, 1080))
    cv2.imwrite(str(out_dir / "01_hero_comparison_1080x1350.png"), _crop_portrait(hero, 1080, 1350))
    written["01_hero_comparison_1080sq.png"] = out_dir / "01_hero_comparison_1080sq.png"
    written["01_hero_comparison_1080x1350.png"] = out_dir / "01_hero_comparison_1080x1350.png"

    teaser_path = out_dir / "PERSIST_AI_LINKEDIN_TEASER.mp4"
    _write_teaser([video for _, video, _ in available], [m for _, m in manifests], teaser_path)
    written["PERSIST_AI_LINKEDIN_TEASER.mp4"] = teaser_path
    return written


def _write_teaser(videos: list[Path], manifests: list[dict], out: Path) -> None:
    if not videos:
        return
    frames: list[np.ndarray] = []
    total = FPS * 9
    for t in range(total):
        phase = t / max(1, total - 1)
        video = videos[min(len(videos) - 1, int(phase * len(videos)))]
        manifest = manifests[min(len(manifests) - 1, int(phase * len(manifests)))] if manifests else {}
        frame_idx = int(_interesting_frame(manifest, 40) + math.sin(phase * math.pi) * 8)
        card = hero_comparison_card(video, manifest)
        frames.append(card)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _ffmpeg_encode(frames, out):
        return
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _ffmpeg_encode(frames: list[np.ndarray], out: Path) -> bool:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i, frame in enumerate(frames):
                cv2.imwrite(str(tmp_path / f"frame_{i:04d}.png"), frame)
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(tmp_path / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(out),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        return out.exists()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
