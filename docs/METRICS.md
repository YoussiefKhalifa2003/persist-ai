# PERSIST-AI Metrics

## Occlusion Event
Ground-truth track `g` with visibility gap >= 10 frames between `t_start` and `t_end`.

## TCUO — Track Continuity Under Occlusion
Fraction of events where predicted track ID remains constant across `[t_start, t_end]`.

## ORR — Occlusion Recovery Rate
Fraction of events where ID is preserved and re-associated within 3 frames of `t_end`.

## RLE — Re-entry Localization Error
L2 pixel distance between predicted and GT center at reappearance (successful recoveries only).

## DRD — Degradation Robustness Drop (v1.5)
`(TCUO_clear - TCUO_weather) / TCUO_clear` — lower is better.
