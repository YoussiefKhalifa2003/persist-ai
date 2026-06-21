"""Build LinkedIn-grade PERSIST-AI visuals with 3D-style technical HUDs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


OUT_DIR = Path("results/linkedin")
W, H = 1920, 1080
FPS = 24

BG = (8, 11, 16)
PANEL = (16, 22, 29)
PANEL_2 = (22, 30, 39)
GRID = (36, 56, 68)
WHITE = (232, 242, 246)
MUTED = (128, 146, 155)
GREEN = (0, 220, 118)
CYAN = (58, 220, 255)
AMBER = (0, 214, 255)
RED = (70, 90, 255)
BLUE = (235, 120, 50)


def _put(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = WHITE,
    thick: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _text_size(text: str, scale: float, thick: int = 1) -> tuple[int, int]:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    return tw, th


def _bg() -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        row = np.array(BG, dtype=np.float32) * (1.0 - t) + np.array((5, 15, 22), dtype=np.float32) * t
        img[y, :] = row
    for x in range(0, W, 64):
        cv2.line(img, (x, 0), (x, H), (13, 23, 30), 1)
    for y in range(0, H, 64):
        cv2.line(img, (0, y), (W, y), (13, 23, 30), 1)
    return img


def _panel(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str | None = None,
    accent: tuple[int, int, int] = CYAN,
) -> None:
    x1, y1, x2, y2 = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), PANEL, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (42, 58, 67), 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x2, y1), accent, 2, cv2.LINE_AA)
    if title:
        _put(img, title.upper(), (x1 + 22, y1 + 38), 0.62, accent, 2)


def _glow_line(
    img: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    overlay = img.copy()
    for scale, alpha in [(8, 0.08), (5, 0.12), (3, 0.18)]:
        cv2.line(overlay, a, b, color, thickness + scale, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        overlay = img.copy()
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    if frame is None or frame.size == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x = (tw - resized.shape[1]) // 2
    y = (th - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _fit_cover(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    if frame is None or frame.size == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = max(tw / w, th / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = max(0, (resized.shape[1] - tw) // 2)
    y = max(0, (resized.shape[0] - th) // 2)
    return resized[y : y + th, x : x + tw].copy()


def _read_frame(video: Path, frame_idx: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    idx = total // 2 if frame_idx is None else min(max(0, frame_idx), max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((540, 960, 3), dtype=np.uint8)
    return frame


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _interesting_frame(manifest: dict, fallback: int = 120) -> int:
    frames = manifest.get("frame_meta", [])
    for row in frames:
        if row.get("ghost_drawn"):
            return int(row.get("frame", fallback))
    for row in frames:
        if row.get("target_state") in {"OCCLUDED", "PREDICTED"}:
            return int(row.get("frame", fallback))
    return fallback


def _project(x: float, z: float, y: float = 0.0) -> tuple[int, int]:
    cam_z = 6.2
    f = 840.0
    px = W * 0.50 + f * x / (z + cam_z)
    py = H * 0.84 - f * (y + 0.18) / (z + cam_z) - z * 24.0
    return int(px), int(py)


def _draw_ground_grid(img: np.ndarray, horizon_y: int = 390) -> None:
    cv2.line(img, (120, horizon_y), (W - 120, horizon_y), (26, 48, 58), 1, cv2.LINE_AA)
    for x in np.linspace(-5.5, 5.5, 12):
        p1 = _project(x, 0.1)
        p2 = _project(x, 11.0)
        _glow_line(img, p1, p2, (16, 70, 86), 1)
    for z in np.linspace(0.5, 10.5, 13):
        pts = np.array([_project(x, z) for x in np.linspace(-5.5, 5.5, 80)], dtype=np.int32)
        cv2.polylines(img, [pts], False, GRID, 1, cv2.LINE_AA)


def _draw_covariance(img: np.ndarray, center: tuple[int, int], idx: int, color: tuple[int, int, int]) -> None:
    rx = 38 + idx * 18
    ry = 12 + idx * 6
    overlay = img.copy()
    cv2.ellipse(overlay, center, (rx, ry), -10, 0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.11, img, 0.89, 0, img)
    cv2.ellipse(img, center, (rx, ry), -10, 0, 360, color, 2, cv2.LINE_AA)


def _draw_prediction_world(img: np.ndarray, phase: float = 1.0) -> None:
    _draw_ground_grid(img)
    path3d = [(-2.7, 1.0), (-2.05, 2.2), (-1.35, 3.4), (-0.62, 4.65), (0.14, 5.8), (0.82, 7.0), (1.34, 8.25)]
    visible = max(2, min(len(path3d), int(2 + phase * (len(path3d) - 1))))
    projected = [_project(x, z) for x, z in path3d[:visible]]
    if len(projected) > 1:
        ribbon_low = []
        ribbon_high = []
        for i, (px, py) in enumerate(projected):
            spread = int(12 + i * 9)
            ribbon_low.append((px, py + spread))
            ribbon_high.append((px, py - spread))
        ribbon = np.array(ribbon_low + list(reversed(ribbon_high)), dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [ribbon], (0, 96, 170), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.36, img, 0.64, 0, img)
        for a, b in zip(projected, projected[1:]):
            _glow_line(img, a, b, CYAN, 3)
    for i, p in enumerate(projected):
        _draw_covariance(img, p, i, AMBER if i > 1 else GREEN)
        cv2.circle(img, p, 7, GREEN if i < 2 else AMBER, -1, cv2.LINE_AA)
        top = (p[0], max(120, p[1] - 180 - i * 16))
        _glow_line(img, p, top, AMBER if i > 1 else GREEN, 1)
        cv2.circle(img, top, 4, AMBER if i > 1 else GREEN, -1, cv2.LINE_AA)

    if projected:
        px, py = projected[min(len(projected) - 1, 2)]
        overlay = img.copy()
        cv2.rectangle(overlay, (px - 34, py - 136), (px + 34, py), AMBER, -1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py - 158), 27, AMBER, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)
        cv2.rectangle(img, (px - 38, py - 146), (px + 38, py), AMBER, 2, cv2.LINE_AA)
        _put(img, "PREDICTED TARGET STATE", (px + 62, py - 86), 0.54, AMBER, 2)

    rng = np.random.default_rng(42)
    for _ in range(160):
        x = float(rng.uniform(-4.5, 4.5))
        z = float(rng.uniform(1.5, 10.2))
        y = float(rng.uniform(0.05, 0.65))
        p = _project(x, z, y)
        if 0 <= p[0] < W and 0 <= p[1] < H:
            cv2.circle(img, p, 1, (48, 132, 145), -1, cv2.LINE_AA)


def _draw_point_cloud(img: np.ndarray, rect: tuple[int, int, int, int], seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = rect
    for _ in range(520):
        x = int(rng.uniform(x1, x2))
        y = int(rng.uniform(y1, y2))
        depth = (y - y1) / max(1, y2 - y1)
        color = (
            int(40 + 180 * depth),
            int(210 - 110 * depth),
            int(255 - 180 * depth),
        )
        cv2.circle(img, (x, y), int(rng.integers(1, 3)), color, -1, cv2.LINE_AA)
    for z in range(7):
        y = y2 - 34 - z * 38
        cv2.line(img, (x1 + 20 + z * 18, y), (x2 - 20 - z * 14, y - 18), (36, 88, 96), 1, cv2.LINE_AA)


def _metric_card(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    value: str,
    detail: str,
    color: tuple[int, int, int],
) -> None:
    _panel(img, rect, None, color)
    x1, y1, x2, _ = rect
    _put(img, title.upper(), (x1 + 20, y1 + 36), 0.48, MUTED, 1)
    _put(img, value, (x1 + 20, y1 + 88), 1.0, color, 2)
    _put(img, detail, (x1 + 20, y1 + 126), 0.48, WHITE, 1)
    cv2.line(img, (x2 - 100, y1 + 106), (x2 - 20, y1 + 40), color, 2, cv2.LINE_AA)
    cv2.circle(img, (x2 - 20, y1 + 40), 5, color, -1, cv2.LINE_AA)


def overview_card(videos: list[Path], manifests: list[Path]) -> np.ndarray:
    img = _bg()
    _put(img, "PERSIST-AI", (70, 100), 1.75, GREEN, 4)
    _put(img, "target-memory tracking under occlusion", (74, 148), 0.82, WHITE, 1)
    _put(img, "Detector observations are fused with motion, appearance, uncertainty, and exit logic.", (74, 190), 0.62, MUTED, 1)

    manifest = _load_manifest(manifests[-1]) if manifests else {}
    frame = _read_frame(videos[-1], _interesting_frame(manifest, 140)) if videos else np.zeros((540, 960, 3), dtype=np.uint8)
    fit = _fit_cover(frame, (1040, 520))
    _panel(img, (70, 245, 1160, 890), "Live comparison frame", GREEN)
    img[315 : 315 + fit.shape[0], 95 : 95 + fit.shape[1]] = fit
    cv2.rectangle(img, (95, 315), (95 + fit.shape[1], 315 + fit.shape[0]), GREEN, 2, cv2.LINE_AA)

    _panel(img, (1220, 245, 1810, 505), "Target-memory stack", CYAN)
    rows = [
        ("LOCKED BBOX", "manual target prompt"),
        ("APPEARANCE", "ReID/color signature"),
        ("MOTION STATE", "Kalman + optical flow"),
        ("OCCLUSION LOGIC", "visible vs blocked"),
    ]
    for i, (name, detail) in enumerate(rows):
        y = 315 + i * 46
        cv2.rectangle(img, (1252, y - 25), (1780, y + 12), PANEL_2, -1, cv2.LINE_AA)
        cv2.rectangle(img, (1252, y - 25), (1780, y + 12), CYAN if i < 2 else GREEN, 1, cv2.LINE_AA)
        _put(img, name, (1270, y), 0.52, WHITE, 1)
        _put(img, detail, (1512, y), 0.43, MUTED, 1)

    _metric_card(img, (1220, 555, 1485, 725), "Identity guard", "ON", "no neighbor snap", GREEN)
    _metric_card(img, (1545, 555, 1810, 725), "Prediction", "3D", "uncertainty cone", AMBER)
    _panel(img, (1220, 780, 1810, 890), "Render contract", AMBER)
    _put(img, "ghost only when target is missing or blocked", (1252, 840), 0.54, WHITE, 1)
    _put(img, "clean exit clears target memory", (1252, 872), 0.54, WHITE, 1)
    return img


def prediction_card() -> np.ndarray:
    img = _bg()
    _put(img, "3D PROBABILISTIC TRAJECTORY", (70, 100), 1.32, WHITE, 3)
    _put(img, "PERSIST-AI predicts a distribution, not fake certainty.", (74, 148), 0.68, MUTED, 1)
    _panel(img, (70, 205, 1235, 940), "Perspective ground-plane forecast", CYAN)
    roi = img[205:940, 70:1235]
    world = np.zeros_like(img)
    _draw_prediction_world(world, 1.0)
    crop = world[120:940, 260:1530]
    crop = _fit(crop, (1125, 665))
    roi[58 : 58 + crop.shape[0], 20 : 20 + crop.shape[1]] = cv2.addWeighted(
        roi[58 : 58 + crop.shape[0], 20 : 20 + crop.shape[1]], 0.15, crop, 0.85, 0
    )
    _draw_point_cloud(img, (132, 292, 1134, 468), seed=33)
    _put(img, "holographic occupancy field", (158, 505), 0.52, CYAN, 1)
    _put(img, "trajectory ribbon is projected onto the sidewalk plane", (158, 535), 0.43, MUTED, 1)
    cv2.rectangle(img, (870, 305), (1130, 455), (5, 12, 16), -1, cv2.LINE_AA)
    cv2.rectangle(img, (870, 305), (1130, 455), CYAN, 1, cv2.LINE_AA)
    for i in range(7):
        y = 430 - i * 17
        cv2.line(img, (900, y), (900 + i * 30, y - 44), (45, 112, 128), 1, cv2.LINE_AA)
    _put(img, "COVARIANCE", (898, 344), 0.42, AMBER, 1)
    _put(img, "velocity damped", (898, 374), 0.38, MUTED, 1)

    _panel(img, (1285, 205, 1810, 940), "Prediction controls", AMBER)
    controls = [
        ("velocity median", "last reliable target motion"),
        ("stop-aware damping", "stationary targets stop drifting"),
        ("optical flow", "local pixel motion inside target"),
        ("covariance", "uncertainty expands over time"),
        ("exit gate", "clear memory at true frame exit"),
    ]
    for i, (name, detail) in enumerate(controls):
        y = 292 + i * 104
        cv2.rectangle(img, (1325, y - 44), (1770, y + 38), (11, 18, 23), -1, cv2.LINE_AA)
        cv2.rectangle(img, (1325, y - 44), (1770, y + 38), AMBER if i >= 2 else CYAN, 1, cv2.LINE_AA)
        _put(img, f"{i + 1:02d}", (1348, y + 8), 0.9, AMBER, 2)
        _put(img, name.upper(), (1415, y - 8), 0.52, WHITE, 1)
        _put(img, detail, (1415, y + 22), 0.43, MUTED, 1)
    return img


def memory_card() -> np.ndarray:
    img = _bg()
    _put(img, "IDENTITY MEMORY PIPELINE", (70, 100), 1.32, WHITE, 3)
    _put(img, "YOLO proposes boxes. PERSIST-AI decides whether a box is still the selected target.", (74, 148), 0.65, MUTED, 1)
    _panel(img, (70, 210, 1810, 930), "Fused target state", GREEN)
    layers = [
        ("01", "YOLO observations", "person / bike / car / bus / truck candidates", CYAN),
        ("02", "Appearance memory", "selected crop signature and ReID fallback", GREEN),
        ("03", "Motion model", "Kalman center, velocity, covariance", AMBER),
        ("04", "Shape constraints", "stable body dimensions and footline", CYAN),
        ("05", "Occlusion classifier", "object / crowd / prediction / exit", GREEN),
    ]
    base_x, base_y = 220, 760
    for i, (idx, name, detail, color) in enumerate(layers):
        x = base_x + i * 270
        y = base_y - i * 90
        pts = np.array([(x, y), (x + 310, y - 62), (x + 445, y + 20), (x + 132, y + 88)], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], PANEL_2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.84, img, 0.16, 0, img)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        _put(img, idx, (x + 34, y + 10), 0.95, color, 2)
        _put(img, name.upper(), (x + 92, y + 2), 0.55, WHITE, 1)
        _put(img, detail, (x + 92, y + 30), 0.42, MUTED, 1)
        if i:
            _glow_line(img, (x - 105, y + 34), (x + 12, y + 8), color, 2)

    _draw_point_cloud(img, (1090, 235, 1735, 535), seed=21)
    _put(img, "semantic point field", (1125, 575), 0.56, CYAN, 1)
    _put(img, "candidate boxes are observations, not identity", (1125, 610), 0.46, MUTED, 1)
    return img


def audit_card(manifests: list[Path]) -> np.ndarray:
    img = _bg()
    _put(img, "FRAME-STATE AUDIT", (70, 100), 1.32, WHITE, 3)
    _put(img, "Every accepted render must obey: visible target = locked, hidden target = predicted, true exit = clear.", (74, 148), 0.62, MUTED, 1)
    _panel(img, (70, 215, 1810, 905), "Render diagnostics", CYAN)
    y0 = 300
    for row, manifest_path in enumerate(manifests[:4]):
        data = _load_manifest(manifest_path)
        frames = data.get("frame_meta", [])
        label = manifest_path.stem.replace("_manifest", "").replace("_", " ")
        y = y0 + row * 135
        _put(img, label.upper(), (110, y), 0.55, WHITE, 1)
        x1, x2 = 430, 1660
        cv2.rectangle(img, (x1, y - 30), (x2, y + 22), (10, 17, 21), -1, cv2.LINE_AA)
        n = max(1, len(frames))
        for i, frame in enumerate(frames):
            state = frame.get("target_state", "EXITED")
            color = GREEN if state == "VISIBLE" else AMBER if state in {"OCCLUDED", "PREDICTED"} else MUTED
            px = x1 + int((x2 - x1) * i / n)
            cv2.line(img, (px, y - 30), (px, y + 22), color, 2)
        cv2.rectangle(img, (x1, y - 30), (x2, y + 22), (58, 76, 86), 1, cv2.LINE_AA)
        ghost_count = sum(1 for frame in frames if frame.get("ghost_drawn"))
        visible_ghost = sum(
            1 for frame in frames if frame.get("ghost_drawn") and frame.get("selected_identity_visible")
        )
        _put(img, f"ghost frames {ghost_count}", (110, y + 42), 0.44, AMBER, 1)
        _put(img, f"visible ghost violations {visible_ghost}", (270, y + 42), 0.44, GREEN if visible_ghost == 0 else RED, 1)
    _put(img, "VISIBLE", (430, 850), 0.52, GREEN, 1)
    _put(img, "OCCLUDED / PREDICTED", (560, 850), 0.52, AMBER, 1)
    _put(img, "EXITED", (830, 850), 0.52, MUTED, 1)
    return img


def comparison_card(videos: list[Path], manifests: list[Path]) -> np.ndarray:
    img = _bg()
    _put(img, "RAW YOLO VS PERSIST-AI", (70, 100), 1.32, WHITE, 3)
    _put(img, "Same frame, different contract: raw detections only vs selected-target memory.", (74, 148), 0.64, MUTED, 1)
    video = videos[-1] if videos else Path()
    manifest = _load_manifest(manifests[-1]) if manifests else {}
    frame = _read_frame(video, _interesting_frame(manifest, 120))
    fit = _fit_cover(frame, (1660, 560))
    _panel(img, (70, 210, 1810, 970), "Comparison render", GREEN)
    img[285 : 285 + fit.shape[0], 130 : 130 + fit.shape[1]] = fit
    cv2.rectangle(img, (130, 285), (130 + fit.shape[1], 285 + fit.shape[0]), GREEN, 2, cv2.LINE_AA)
    _put(img, "PERSIST-AI maintains a locked identity state.", (135, 925), 0.62, GREEN, 2)
    _put(img, "Raw YOLO remains a visible-object detector.", (1050, 925), 0.62, MUTED, 1)
    return img


def _animated_frame(videos: list[Path], manifests: list[Path], t: int, total: int) -> np.ndarray:
    img = _bg()
    phase = (t % total) / max(1, total - 1)
    _put(img, "PERSIST-AI TARGET MEMORY", (70, 96), 1.25, GREEN, 3)
    _put(img, "3D trajectory ribbon + uncertainty field + identity state", (74, 138), 0.62, MUTED, 1)
    _panel(img, (70, 190, 1170, 940), "Prediction space", CYAN)
    world = np.zeros_like(img)
    _draw_prediction_world(world, phase)
    crop = _fit(world[95:970, 220:1580], (1060, 690))
    img[235 : 235 + crop.shape[0], 90 : 90 + crop.shape[1]] = crop

    _panel(img, (1230, 190, 1810, 560), "Source evidence", GREEN)
    if videos:
        manifest = _load_manifest(manifests[-1]) if manifests else {}
        frame_idx = int(_interesting_frame(manifest, 120) + math.sin(phase * math.pi) * 18)
        frame = _fit(_read_frame(videos[-1], frame_idx), (520, 260))
        img[252 : 252 + frame.shape[0], 1260 : 1260 + frame.shape[1]] = frame
    _panel(img, (1230, 615, 1810, 940), "State vector", AMBER)
    bars = [
        ("appearance", 0.86),
        ("motion", 0.74 + 0.12 * math.sin(phase * math.pi)),
        ("occlusion", 0.64 + 0.24 * phase),
        ("confidence", max(0.26, 0.95 - 0.44 * phase)),
    ]
    for i, (name, val) in enumerate(bars):
        y = 690 + i * 56
        _put(img, name.upper(), (1270, y), 0.46, WHITE, 1)
        cv2.rectangle(img, (1460, y - 16), (1740, y + 2), (40, 47, 50), -1, cv2.LINE_AA)
        color = GREEN if val > 0.75 else AMBER
        cv2.rectangle(img, (1460, y - 16), (1460 + int(280 * val), y + 2), color, -1, cv2.LINE_AA)
        _put(img, f"{int(val * 100)}", (1755, y + 2), 0.42, color, 1)
    return img


def write_teaser(videos: list[Path], manifests: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    total = FPS * 7
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for t in range(total):
        writer.write(_animated_frame(videos, manifests, t, total))
    writer.release()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = [
        p
        for p in [
            Path("results/demo_videos/VIDEO3_PERSIST_SPLIT.mp4"),
            Path("results/demo_videos/VIDEO4_PERSIST_SPLIT.mp4"),
            Path("results/demo_videos/VIDEO5_PERSIST_SPLIT.mp4"),
            Path("results/demo_videos/VIDEO6_PERSIST_SPLIT.mp4"),
        ]
        if p.exists()
    ]
    manifests = [
        p
        for p in [
            Path("results/demo_videos/VIDEO3_manifest.json"),
            Path("results/demo_videos/VIDEO4_manifest.json"),
            Path("results/demo_videos/VIDEO5_manifest.json"),
            Path("results/demo_videos/VIDEO6_manifest.json"),
        ]
        if p.exists()
    ]

    outputs = {
        "01_system_overview.png": overview_card(videos, manifests),
        "02_3d_prediction_field.png": prediction_card(),
        "03_identity_memory_stack.png": memory_card(),
        "04_frame_state_audit.png": audit_card(manifests),
        "05_raw_vs_persist_comparison.png": comparison_card(videos, manifests),
    }
    for name, img in outputs.items():
        cv2.imwrite(str(OUT_DIR / name), img)
    write_teaser(videos, manifests, OUT_DIR / "PERSIST_AI_3D_LINKEDIN_TEASER.mp4")
    print(f"LinkedIn assets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
