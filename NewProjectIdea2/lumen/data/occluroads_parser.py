"""Optional OccluRoads XML parser — integrate when access granted."""

from __future__ import annotations

from pathlib import Path

import xmltodict

from lumen.utils.io import save_json


def parse_occluroads_xml(xml_path: Path) -> dict:
    with xml_path.open("r", encoding="utf-8") as f:
        data = xmltodict.parse(f.read())
    return data


def convert_all(annotations_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for xml in annotations_dir.glob("*.xml"):
        save_json(output_dir / f"{xml.stem}.json", parse_occluroads_xml(xml))
