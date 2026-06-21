from __future__ import annotations

from pathlib import Path

import pandas as pd

from lumen.eval.events import extract_occlusion_events
from lumen.eval.metrics_orr import compute_orr
from lumen.eval.metrics_rle import compute_rle
from lumen.eval.metrics_tcuo import compute_tcuo
from lumen.utils.io import ensure_dir, load_config, save_json


class EvalRunner:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = load_config(config_path)
        self.eval_cfg = load_config("configs/eval.yaml")

    def run_comparison(
        self,
        events: list,
        method_tracks: dict[str, dict[str, dict[int, int]]],
        gt_centers: dict | None = None,
        pred_centers: dict | None = None,
    ) -> pd.DataFrame:
        rows = []
        for method, video_maps in method_tracks.items():
            tcuo = compute_tcuo(events, video_maps)
            orr = compute_orr(events, video_maps)
            rle = {"mean": float("nan")}
            if gt_centers and pred_centers:
                rle = compute_rle(events, pred_centers.get(method, {}), gt_centers)
            rows.append(
                {
                    "method": method,
                    "TCUO": round(tcuo, 4),
                    "ORR": round(orr, 4),
                    "RLE_mean": round(rle.get("mean", float("nan")), 2),
                }
            )
        return pd.DataFrame(rows)

    def save_results(self, df: pd.DataFrame, output: str | Path) -> None:
        output = Path(output)
        ensure_dir(output.parent)
        df.to_csv(output, index=False)
        save_json(output.with_suffix(".json"), df.to_dict(orient="records"))
