from __future__ import annotations


def compute_drd(tcuo_clear: float, tcuo_weather: float) -> float:
    """Degradation Robustness Drop — lower is better."""
    if tcuo_clear <= 0:
        return 0.0
    return (tcuo_clear - tcuo_weather) / tcuo_clear
