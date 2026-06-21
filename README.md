# PERSIST-AI — Object Permanence for Autonomous Perception

**Detection tells us what is visible. PERSIST-AI models what still exists.**

PERSIST-AI is an open, reproducible persistent-tracking layer for autonomous perception. When objects disappear behind occluders, standard trackers drop IDs — PERSIST-AI maintains latent world state, predicts re-entry zones, and re-associates on reappearance.

## Watch the real demo (start here)

```powershell
# 1. Download real MOT17 street footage (~300 frames, public square with pedestrians)
python scripts/download_real_data.py

# 2. Build labeled side-by-side demo (~10-20 seconds, plain-English captions)
python scripts/build_real_demo.py --num-frames 250 --start-frame 30 --fps 15

# Open: results/demo_videos/PERSIST_AI_REAL_MOT17-02.mp4
```

**Left panel** = PERSIST-AI (keeps ghost track on locked TARGET). **Right panel** = raw YOLO (all detections, no permanence).  
Bottom banner explains each phase in plain English.

The old synthetic demo was empty because YOLO does not detect cartoon shapes — real MOT17 footage fixes that.

## Video 3 — Street scene (YouTube, staged demo)

```powershell
# Uses cached detections if data/cache/sidewalk_demo_dets.json exists
python scripts/build_street_demo.py --video data/raw/youtube/sidewalk_demo.mp4

# Outputs:
#   results/demo_videos/VIDEO3_VEHICLES.mp4      — staged: raw → Activate PERSIST-AI → split
#   results/demo_videos/VIDEO3_RAW_ONLY.mp4     — Act 1 loop (web viewer)
#   results/demo_videos/VIDEO3_PERSIST_SPLIT.mp4  — PERSIST-AI left vs raw YOLO right
#   results/demo_videos/VIDEO3_manifest.json    — frame metadata for viewer

# Interactive web viewer (open in browser after building):
#   demo/viewer/index.html
```

**Act 1:** fullscreen raw YOLO. **Act 2:** pulsing “Activate PERSIST-AI” button. **Act 3:** PERSIST-AI (left) keeps tan-coat TARGET ghost through bus; raw (right) shows all boxes only.

Use `--no-staged` to skip Acts 1–2 and export split comparison only.

## Try PERSIST-AI — Interactive local web demo

```powershell
python scripts/run_interactive_demo.py
```

Open: http://127.0.0.1:8000

The local app lets a user pick a curated scene, click **Enhance Scene**, select a visible
person or vehicle, and render a PERSIST-AI vs raw YOLO comparison for that target. Custom
video upload is shown as a beta path but is intentionally disabled in v1.

Curated scenes currently include:
- Video 3: street-side vehicle occlusion and clean target exit.
- Video 4: busy sidewalk crowd with pedestrians, bikes, foreground blockers, and edge exits.

## Note on GPU

If PyTorch CPU wheels are installed (common on Python 3.14), PERSIST-AI auto-falls back to `device=cpu` and `yolov8n`. For RTX GPUs, use Python 3.10–3.12 with CUDA-enabled PyTorch.

## Quickstart

```powershell
cd NewProjectIdea2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/create_synthetic_demo.py
python -m lumen track --method bytetrack --video data/raw/synthetic/occlusion_demo.mp4
python -m lumen track --method lumen --video data/raw/synthetic/occlusion_demo.mp4
pytest tests/ -v
```

## MOT17 smoke test

```powershell
.\scripts\download_mot17.ps1
python -m lumen track --method bytetrack --dataset mot17 --sequence MOT17-02
python -m lumen track --method lumen --dataset mot17 --sequence MOT17-02
```

## BDD100K eval

```powershell
.\scripts\01_download_bdd.ps1 -Verify
python scripts/03_filter_bdd_occlusion_clips.py --rank-hero --top 5
python -m lumen eval --dataset bdd --output results/tables/bdd_main.csv
```

## Architecture

```
Video → YOLOv8m → Baseline Tracker / PERSIST-AI TrackManager
                         ↓
              Latent FSM + Exit Zones + ReID Gate
                         ↓
              TCUO / ORR / RLE metrics + demo video
```

## Metrics

| Metric | Description |
|--------|-------------|
| TCUO | Track Continuity Under Occlusion |
| ORR | Occlusion Recovery Rate |
| RLE | Re-entry Localization Error (px) |
| DRD | Degradation Robustness Drop (v1.5) |

See [docs/METRICS.md](docs/METRICS.md).

## Datasets

- **Primary:** MOT17 (dev), BDD100K MOT (eval + hero clips)
- **Optional:** OccluRoads (email access)
- **v1.5:** ACDC weather

See [docs/DATASETS.md](docs/DATASETS.md).

## v1.5 Weather

ACDC fog/rain/night evaluation via `lumen/eval/metrics_drd.py`. Download ACDC and run:

```powershell
python -m lumen eval --dataset acdc --output results/tables/weather_drd.csv
```

## Citation

```bibtex
@software{lumen2026,
  title={PERSIST-AI: Object Permanence for Autonomous Perception},
  year={2026},
  url={https://github.com/yourusername/lumen}
}
```

## License

MIT
