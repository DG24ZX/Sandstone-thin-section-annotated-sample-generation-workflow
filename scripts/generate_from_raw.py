#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinsection_generator.pipeline import RawSliceConfig, generate_from_raw
from thinsection_generator.reconstruction import ParticleLibraryPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired PPL/XPL thin-section images and labels from raw digital-core slices."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    raw_cfg = RawSliceConfig(
        label_raw=cfg["raw"]["label_raw"],
        dissolution_raw=cfg["raw"].get("dissolution_raw"),
        cement_raw=cfg["raw"].get("cement_raw"),
        shape=tuple(cfg["raw"]["shape"]),
        slice_index=int(cfg["raw"].get("slice_index", 0)),
        resize_to=tuple(cfg["raw"]["resize_to"]) if cfg["raw"].get("resize_to") else None,
    )

    texture_folders = cfg.get("texture_folders", {})
    libraries = ParticleLibraryPaths(
        quartz=cfg["particle_libraries"]["quartz"],
        feldspar=cfg["particle_libraries"]["feldspar"],
        lithic=cfg["particle_libraries"]["lithic"],
        xpl_folder=texture_folders.get("xpl"),
        ppl_folder=texture_folders.get("ppl"),
    )

    generate_from_raw(
        raw_cfg,
        libraries,
        output_dir=cfg["output"]["output_dir"],
        sample_name=cfg["output"].get("sample_name", "sample_0001"),
    )
    print("Generation finished. Output folders: xpl/, ppl/, labels/.")


if __name__ == "__main__":
    main()
