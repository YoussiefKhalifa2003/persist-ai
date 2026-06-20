# Baselines

| Method | Implementation |
|--------|----------------|
| ByteTrack | Ultralytics `bytetrack.yaml` |
| BoT-SORT | Ultralytics `botsort.yaml` |
| OccluBoost | BoxMOT optional extra |
| PERSIST-AI | Custom TrackManager FSM |

All methods should use identical YOLOv8m detections when comparing fairly (detection cache).
