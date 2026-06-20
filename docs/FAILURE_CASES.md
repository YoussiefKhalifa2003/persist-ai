# Failure Cases

- **Wrong re-ID**: Similar pedestrian appears near exit zone — tighten `reid_cosine_threshold`
- **Ghost drift**: Bad velocity — increase Kalman process noise in latent mode
- **No occluder vehicle**: Pedestrian behind pole — `occluder_unknown` flag, directional fallback zone
- **Multiple pedestrians**: ID swap — v1 uses pedestrian-only focus; document limitation

See `results/demo_videos/failure_case_01.mp4` after demo generation.
