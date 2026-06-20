"""Compositor for VIDEO 3 — vehicle object permanence."""

from __future__ import annotations

from lumen.viz.concept_compositor import (
    ConceptBeat,
    compose_concept_frame,
    compose_title_card,
)

# Re-use concept layout; vehicle-specific copy
VEHICLE_INTRO = "When a car hides behind a truck, normal AI forgets it."
VEHICLE_OUTRO = "Object permanence for autonomous driving."


class VehicleBeat(str):
    VISIBLE = "Both trackers see the vehicle."
    OCCLUDED = "Vehicle hidden - baseline forgets, PERSIST-AI keeps ghost."
    RETURN = "Vehicle back - PERSIST-AI kept id 1."


def vehicle_title_card(w: int, h: int, title: str, sub: str = "") -> __import__("numpy").ndarray:
    return compose_title_card(
        w, h, title, sub or "VIDEO 3 - VEHICLES (car passes behind truck)",
    )


def vehicle_frame(*args, **kwargs):
    kwargs.setdefault("lumen_label", kwargs.get("lumen_label", "ID 1"))
    return compose_concept_frame(*args, **kwargs)
