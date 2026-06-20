# PERSIST-AI — Object Permanence for Autonomous Perception

**Detection tells us what is visible. PERSIST-AI models what still exists.**

PERSIST-AI is an open, reproducible persistent-tracking layer for autonomous perception. When pedestrians or other targets disappear behind occluders (cars, buses, pillars), standard detectors and trackers drop IDs and forget the target. PERSIST-AI maintains **latent world state**: it keeps a **ghost lock** on the locked subject, predicts where they may reappear, and re-associates when they become visible again — while cleanly **stopping** when the subject truly leaves the scene.

The flagship demo is **Video 3 — Street Scene**: a YouTube sidewalk clip where PERSIST-AI tracks the **tan-coat woman** through a van and bus occlusion, shown side-by-side against raw YOLO detections.

---

## Table of contents

1. [Main goal](#main-goal)
2. [What you will see in the demo](#what-you-will-see-in-the-demo)
3. [Quick start (Video 3)](#quick-start-video-3)
4. [Installation](#installation)
5. [Project structure](#project-structure)
6. [How Video 3 works](#how-video-3-works)
7. [Interactive web viewer](#interactive-web-viewer)
8. [All demo build scripts](#all-demo-build-scripts)
9. [CLI reference](#cli-reference)
10. [Architecture](#architecture)
11. [Core modules (Video 3)](#core-modules-video-3)
12. [Configuration](#configuration)
13. [Datasets & evaluation](#datasets--evaluation)
14. [Metrics](#metrics)
15. [Testing](#testing)
16. [GPU vs CPU](#gpu-vs-cpu)
17. [Troubleshooting](#troubleshooting)
18. [Documentation index](#documentation-index)
19. [Citation & license](#citation--license)

---

## Main goal

Build a **credible product demo** that proves object permanence for street perception:

| Panel | Meaning |
|-------|---------|
| **Left — PERSIST-AI** | Locked **TARGET** with ghost persistence through occlusions |
| **Right — Raw YOLO** | Standard detection: every person/vehicle box, no permanence |

**Rules the demo must satisfy:**

1. **Ghost through real occlusions** — When a vehicle blocks the tan-coat woman, PERSIST-AI draws a dashed **TARGET (ghost)** box and keeps the same identity; raw YOLO loses her.
2. **Clean exit** — When she leaves the frame, PERSIST-AI **stops tracking**. No ghost on the bus, no end-of-clip glitch, caption switches to *"Target exited — standard detection only."*
3. **Global logic** — Occlusion windows and exit rules are computed from detections + anchor path, not hardcoded per-frame hacks (see `lumen/pipelines/persist_occlusion.py`).
4. **Staged narrative (optional)** — Raw detection plays first → user presses **Activate PERSIST-AI** → split comparison reveals the difference.

---

## What you will see in the demo

**Video 3 timeline (approximate clip indices):**

| Phase | Frames | Left (PERSIST-AI) | Right (Raw YOLO) |
|-------|--------|-------------------|------------------|
| Visible tracking | Early clip | Solid green **TARGET** on tan-coat woman | All pedestrian + vehicle boxes |
| Van occlusion | ~37–44 | Dashed **TARGET (ghost)** advances on sidewalk | Woman missing from detections |
| Reappear | ~44+ | Solid **TARGET** again | New detection boxes |
| Subject exits | ~76+ | No ghost; gray crowd outlines only | Normal detections |
| Epilogue | ~76–92 | *"Target exited — standard detection only."* | Raw boxes only |

**Exported files** (after building):

| File | Description |
|------|-------------|
| `results/demo_videos/VIDEO3_VEHICLES.mp4` | Full staged demo: Act 1 raw → Act 2 button → Act 3 split |
| `results/demo_videos/VIDEO3_RAW_ONLY.mp4` | Act 1 loop (used by web viewer) |
| `results/demo_videos/VIDEO3_PERSIST_SPLIT.mp4` | Split comparison only (PERSIST-AI left, raw right) |
| `results/demo_videos/VIDEO3_manifest.json` | Per-frame metadata: phase, occlusion, ghost_drawn, anchor_cx |

---

## Quick start (Video 3)

### Prerequisites

- **Python 3.10–3.12** recommended (3.14 works on CPU with auto-fallback to `yolov8n`)
- **ffmpeg** (optional but helpful for video tooling)
- ~2 GB disk for venv + one demo video

### 1. Clone and install

```powershell
git clone https://github.com/YOUR_USERNAME/persist-ai.git
cd persist-ai

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -e ".[dev]"
```

### 2. Get source video + detections

The build script uses **cached YOLO detections** when available (fast, no GPU needed for render):

```powershell
# Option A — If you already have the video locally:
#   Place at data/raw/youtube/sidewalk_demo.mp4

# Option B — Let the script download from YouTube (default URL in build script):
python scripts/build_street_demo.py --video data/raw/youtube/sidewalk_demo.mp4 --scale 0.85
```

On first run without `data/cache/sidewalk_demo_dets.json`, the script runs YOLO and writes that cache. Subsequent runs reuse it.

### 3. Build the street demo

```powershell
python scripts/build_street_demo.py `
  --video data/raw/youtube/sidewalk_demo.mp4 `
  --scale 0.85 `
  --staged `
  --output results/demo_videos/VIDEO3_VEHICLES.mp4
```

**Flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--video` | YouTube download | Local MP4 path |
| `--scale` | `0.85` | Resize factor (speed / memory) |
| `--staged` | on | Act 1 raw → Act 2 button → Act 3 split |
| `--no-staged` | — | Export split comparison only |
| `--fps` | `15` | Output frame rate |
| `--output` | `results/demo_videos/VIDEO3_VEHICLES.mp4` | Main output path |

### 4. Watch

```powershell
# Full staged narrative
start results/demo_videos/VIDEO3_VEHICLES.mp4

# Or split-only comparison
start results/demo_videos/VIDEO3_PERSIST_SPLIT.mp4
```

---

## Installation

### Full dev install

```powershell
pip install -e ".[dev]"
```

Optional extras:

```powershell
pip install -e ".[dev,boxmot,occluroads]"
```

| Extra | Purpose |
|-------|---------|
| `dev` | pytest, ruff |
| `boxmot` | BoT-SORT / stronger baselines |
| `occluroads` | OccluRoads XML parsing |

### Environment variables (optional)

Copy `.env.example` → `.env`:

```powershell
copy .env.example .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `LUMEN_DATA_ROOT` | `data` | Data root (legacy name; package is PERSIST-AI) |
| `LUMEN_RESULTS_ROOT` | `results` | Output root |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |

---

## Project structure

```
persist-ai/
├── README.md                 ← You are here
├── pyproject.toml            ← Package: persist-ai (import path: lumen)
├── LICENSE                   ← MIT
├── CITATION.cff
├── .env.example
├── configs/
│   ├── default.yaml          ← Detector + PERSIST-AI hyperparameters
│   ├── lumen.yaml
│   └── eval.yaml
├── lumen/                    ← Main Python package (historical name)
│   ├── brand.py              ← BRAND = "PERSIST-AI"
│   ├── cli.py                ← persist-ai / lumen CLI
│   ├── types.py              ← BBox, Detection, TrackOutput
│   ├── core/                 ← TrackManager, latent FSM, exit zones, ReID
│   ├── detector/             ← YOLOv8 Ultralytics wrapper
│   ├── trackers/             ← ByteTrack / baseline adapters
│   ├── pipeline/             ← Full LumenPipeline + baselines
│   ├── pipelines/
│   │   ├── comparison_pipeline.py   ← Demo comparison engine
│   │   └── persist_occlusion.py     ← Global occlusion / ghost rules ★
│   ├── data/
│   │   ├── pedestrian_clip_finder.py  ← Tan-coat path + clip builder ★
│   │   ├── mot17_parser.py
│   │   ├── bdd_mot_parser.py
│   │   └── ...
│   ├── viz/
│   │   ├── crowd_compositor.py      ← Split panel renderer ★
│   │   ├── real_compositor.py       ← Beats, SmoothBBox, intro frames
│   │   ├── staged_compositor.py     ← Raw fullscreen + Activate button
│   │   └── silhouette.py            ← Ghost silhouette overlay
│   └── eval/                 ← TCUO, ORR, RLE, DRD metrics
├── scripts/
│   ├── build_street_demo.py  ← Video 3 builder ★
│   ├── build_real_demo.py    ← MOT17 real-footage demo
│   ├── build_vehicle_demo.py
│   ├── build_concept_demo.py
│   ├── download_real_data.py
│   ├── create_synthetic_demo.py
│   └── 01–08_*.py / *.ps1    ← Dataset download + eval pipeline
├── demo/
│   └── viewer/               ← Interactive HTML viewer ★
│       ├── index.html
│       ├── viewer.js
│       └── style.css
├── tests/
│   ├── test_persist_occlusion.py
│   ├── test_demo_overhaul.py
│   └── ...
├── data/
│   └── manifests/            ← Clip manifests (tracked in git)
└── docs/                     ← Deep-dive docs
    ├── METRICS.md
    ├── DATASETS.md
    ├── BASELINES.md
    └── ...
```

★ = most relevant for the Video 3 street demo.

**Note:** The installable package name is `persist-ai`, but Python imports use `lumen` (legacy internal name). Both CLI entry points work:

```powershell
python -m lumen track --help
persist-ai track --help
```

---

## How Video 3 works

End-to-end pipeline inside `scripts/build_street_demo.py`:

```
YouTube MP4
    ↓
YOLO detections (classes: person=0, car=2, bus=5, truck=7)
    ↓  cached → data/cache/sidewalk_demo_dets.json
build_leftmost_woman_path()     ← NN lock on tan-coat woman (cluster + velocity)
    ↓
find_all_occlusion_windows()    ← Auto van + bus intervals
    ↓
snap_ghost_off_vehicles()       ← Full-body ghost box on sidewalk; velocity-based cx
    ↓
finalize_anchor_path()          ← Cancel post-exit windows; trim clip length
    ↓
DemoComparisonEngine            ← Baseline vs PERSIST-AI track state per frame
    ↓
compose_crowd_frame()           ← Left PERSIST-AI panel + right raw panel
    ↓
MP4 + VIDEO3_manifest.json
```

### Ghost vs visible rendering

For each frame `i`:

1. `anchor = anchor_path[i]` — predicted/tracked box for the locked woman.
2. `visible = _person_visible(raw_dets[i], anchor)` — YOLO agrees with anchor.
3. `in_oc = frame_is_persist_latent(i, anchor, raw_dets[i], occlusion_windows)` — should draw ghost?
4. **Solid TARGET** when anchor exists and person is visible.
5. **Dashed TARGET (ghost)** when `in_oc` and anchor exists.
6. **Nothing / idle caption** when anchor is cleared (subject exited).

### Post-exit rule (fixes end-of-clip glitch)

`finalize_anchor_path()` cancels occlusion windows that start **≥ 3 frames after the last YOLO match** on the anchor. This prevents a bus-only window from creating a ghost after the woman has already left — the root cause of the previous “glitch” at frames 95–110.

### Staged Acts (when `--staged`)

| Act | Duration | Content |
|-----|----------|---------|
| **Act 1** | ~45 frames | Fullscreen raw YOLO — *"Standard perception: only what the camera sees."* |
| **Act 2** | ~22 frames | Frozen frame + pulsing **Activate PERSIST-AI** button |
| **Act 3** | Full clip | Split: PERSIST-AI left, raw YOLO right |

---

## Interactive web viewer

After building Video 3:

1. Open `demo/viewer/index.html` in a browser (Chrome / Edge recommended).
2. **Phase 1:** Loops `VIDEO3_RAW_ONLY.mp4` with an **Activate PERSIST-AI** button.
3. **Phase 2:** Plays `VIDEO3_PERSIST_SPLIT.mp4` with panel labels.
4. **Replay from raw** returns to Phase 1.

The viewer expects videos at:

```
results/demo_videos/VIDEO3_RAW_ONLY.mp4
results/demo_videos/VIDEO3_PERSIST_SPLIT.mp4
```

For local file access, serve the repo root if your browser blocks `file://` cross-path video loading:

```powershell
# Python 3
python -m http.server 8080
# Then open http://localhost:8080/demo/viewer/index.html
```

---

## All demo build scripts

| Script | Output | Use case |
|--------|--------|----------|
| `scripts/build_street_demo.py` | `VIDEO3_*.mp4` | **Primary** — YouTube sidewalk, tan-coat woman |
| `scripts/build_real_demo.py` | `PERSIST_AI_REAL_MOT17-*.mp4` | Real MOT17 public-square footage |
| `scripts/build_vehicle_demo.py` | Vehicle occlusion demo | Car/truck occluder focus |
| `scripts/build_concept_demo.py` | Concept / explainer video | High-level product story |
| `scripts/build_both_demos.py` | Multiple demos in one run | Batch export |
| `scripts/create_synthetic_demo.py` | Synthetic shapes video | Unit pipeline smoke test (YOLO won't detect cartoons well) |
| `scripts/download_real_data.py` | MOT17 sample download | Prep for `build_real_demo.py` |

**MOT17 real demo (alternative to Video 3):**

```powershell
python scripts/download_real_data.py
python scripts/build_real_demo.py --num-frames 250 --start-frame 30 --fps 15
# → results/demo_videos/PERSIST_AI_REAL_MOT17-02.mp4
```

---

## CLI reference

```powershell
# Precompute YOLO detections for a video
python -m lumen detect --video path/to/video.mp4 --cache results/detections/video.npz

# Run tracking
python -m lumen track --method bytetrack --video path/to/video.mp4
python -m lumen track --method lumen --video path/to/video.mp4

# MOT17 sequence (after download)
python -m lumen track --method lumen --dataset mot17 --sequence MOT17-02

# Evaluation
python -m lumen eval --dataset bdd --output results/tables/bdd_main.csv
```

| Command | Methods / datasets |
|---------|-------------------|
| `detect` | YOLO cache to NPZ |
| `track` | `bytetrack`, `botsort`, `lumen` |
| `eval` | `mot17`, `bdd`, `acdc` |

---

## Architecture

```
Video frames
    ↓
YOLOv8 (Ultralytics) — detector/yolo_ultralytics.py
    ↓
┌─────────────────────────┬──────────────────────────┐
│ Baseline tracker        │ PERSIST-AI TrackManager    │
│ (ByteTrack / BoT-SORT)  │ (core/track_manager.py)  │
│                         │  • Latent FSM              │
│                         │  • Exit zones              │
│                         │  • ReID gate               │
│                         │  • Motion model            │
└─────────────────────────┴──────────────────────────┘
    ↓
Comparison + compositors (viz/)
    ↓
Demo MP4 + metrics (eval/)
```

**Video 3 shortcut path:** Instead of running the full online TrackManager, the street demo uses a **curated anchor path** (`pedestrian_clip_finder.py`) plus **global occlusion rules** (`persist_occlusion.py`) to produce the same visual story deterministically from cached detections — faster iteration and reproducible exports.

---

## Core modules (Video 3)

### `lumen/pipelines/persist_occlusion.py`

| Function | Role |
|----------|------|
| `find_all_occlusion_windows()` | Detect every interval where anchor is hidden and a vehicle is near |
| `frame_is_persist_latent()` | Per-frame: should PERSIST-AI show ghost state? |
| `finalize_anchor_path()` | Trim clip; cancel windows after subject exit; clear stale anchors |
| `mask_subject_windows()` | Hide target YOLO boxes on baseline during occlusions |

### `lumen/data/pedestrian_clip_finder.py`

| Function | Role |
|----------|------|
| `build_leftmost_woman_path()` | Lock onto tan-coat woman via cluster + nearest-neighbor |
| `snap_ghost_off_vehicles()` | Normalize ghost to sidewalk height + median body size |
| `build_tan_coat_clip()` | Full clip builder: path → windows → snap → finalize |

### `lumen/viz/crowd_compositor.py`

Renders the split canvas:

- **Left:** PERSIST-AI panel — TARGET / ghost, trail, silhouette, confidence bar, exit zones
- **Right:** Raw YOLO — all detection classes
- Footer beat caption + frame counter

---

## Configuration

Main config: `configs/default.yaml`

```yaml
device: cuda:0
detector:
  model: yolov8m.pt
  imgsz: 1280
  conf: 0.25
  classes: [0, 2, 5, 7]   # person, car, bus, truck
lumen:
  latent_enter_frames: 2
  latent_max_frames: 45
  confidence_decay_lambda: 0.05
  exit_zone_margin_px: 25
  reid_cosine_threshold: 0.45
  pedestrian_only: true
```

Video 3 overrides detector to `yolov8n` + CPU automatically when CUDA is unavailable.

---

## Datasets & evaluation

| Dataset | Role | Setup |
|---------|------|-------|
| **YouTube sidewalk** | Hero Video 3 demo | `build_street_demo.py` |
| **MOT17** | Real pedestrian tracking | `scripts/download_mot17.ps1` |
| **BDD100K MOT** | Occlusion clip mining + eval | `scripts/01_download_bdd.ps1` |
| **OccluRoads** | Optional occlusion benchmark | Email access; see `docs/DATASETS.md` |
| **ACDC** | Weather robustness (v1.5) | `scripts/eval_acdc_weather.py` |

Full dataset guide: [docs/DATASETS.md](docs/DATASETS.md)

**BDD eval pipeline:**

```powershell
.\scripts\01_download_bdd.ps1 -Verify
python scripts/03_filter_bdd_occlusion_clips.py --rank-hero --top 5
python -m lumen eval --dataset bdd --output results/tables/bdd_main.csv
```

---

## Metrics

| Metric | Name | Description |
|--------|------|-------------|
| **TCUO** | Track Continuity Under Occlusion | ID / track persistence through hidden intervals |
| **ORR** | Occlusion Recovery Rate | Successful re-acquisition after occlusion |
| **RLE** | Re-entry Localization Error | Pixel error at reappearance |
| **DRD** | Degradation Robustness Drop | Weather / noise robustness (v1.5) |

Details: [docs/METRICS.md](docs/METRICS.md)

---

## Testing

```powershell
pytest tests/ -v
```

Key test files for Video 3 behavior:

| Test file | Covers |
|-----------|--------|
| `tests/test_persist_occlusion.py` | Occlusion windows, latent frames, finalize trim |
| `tests/test_demo_overhaul.py` | Anchor termination, compositor layout, ghost snap |
| `tests/test_exit_zone.py` | Exit zone geometry |
| `tests/test_latent_fsm.py` | Latent state machine |

---

## GPU vs CPU

| Environment | Behavior |
|-------------|----------|
| **CUDA + Python 3.10–3.12** | Full speed; `yolov8m`, half precision |
| **CPU-only / Python 3.14** | Auto-fallback in CLI and demo scripts: `device=cpu`, `yolov8n`, `half=false` |

For GPU setup on Windows:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

---

## Troubleshooting

### Video 3 build is slow

- Ensure `data/cache/sidewalk_demo_dets.json` exists (second run skips YOLO).
- Use `--scale 0.85` or lower.
- Use `--no-staged` if you only need the split MP4.

### Web viewer shows blank video

- Build demos first so `results/demo_videos/VIDEO3_*.mp4` exist.
- Serve over HTTP (`python -m http.server`) instead of opening `file://` directly.

### Ghost appears on bus after woman left

- Rebuild with latest code; `finalize_anchor_path()` should cancel post-exit windows.
- Check `VIDEO3_manifest.json`: frames after exit should have `"ghost_drawn": false`, `"anchor_cx": null`.

### Van pass: ghost disappears

- Confirm occlusion window includes van interval (~37–44) in build log: `occlusions [(37, 44), ...]`.
- Ghost uses dashed full-body box at sidewalk `cy` from `snap_ghost_off_vehicles()`.

### `ModuleNotFoundError: lumen`

```powershell
pip install -e ".[dev]"
```

---

## Documentation index

| Doc | Topic |
|-----|-------|
| [docs/METRICS.md](docs/METRICS.md) | TCUO, ORR, RLE, DRD definitions |
| [docs/DATASETS.md](docs/DATASETS.md) | Download links, folder layout |
| [docs/BASELINES.md](docs/BASELINES.md) | ByteTrack, BoT-SORT, OccludBoost |
| [docs/FAILURE_CASES.md](docs/FAILURE_CASES.md) | Known limitations |
| [docs/LINKEDIN_LAUNCH.md](docs/LINKEDIN_LAUNCH.md) | Launch / messaging notes |

---

## Citation & license

### Citation

```bibtex
@software{persistai2026,
  title  = {PERSIST-AI: Object Permanence for Autonomous Perception},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/persist-ai}
}
```

Also see [CITATION.cff](CITATION.cff) for GitHub's citation UI.

### License

MIT — see [LICENSE](LICENSE).

---

## What gets committed vs ignored

This repo is configured for GitHub with a root `.gitignore`:

| Tracked | Ignored (generated / large) |
|---------|----------------------------|
| Source code (`lumen/`, `scripts/`, `tests/`) | `.venv/`, `__pycache__/` |
| Configs (`configs/`) | `data/raw/` (videos), `data/cache/` (detections) |
| Manifests (`data/manifests/`) | `results/` (MP4 exports) |
| Demo viewer (`demo/viewer/`) | `*.pt`, `*.npz`, `*.log` |
| Docs (`docs/`) | `.env`, IDE folders |

After cloning, run the [Quick start](#quick-start-video-3) to regenerate videos and detection caches locally.
