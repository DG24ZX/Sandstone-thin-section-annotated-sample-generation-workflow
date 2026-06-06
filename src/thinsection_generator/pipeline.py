from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .enhancement import add_clay_rims, apply_dissolution, smooth_edges_around_mask
from .io import read_raw_slice, save_generated_sample
from .reconstruction import ParticleLibraryPaths, reconstruct_from_instances
from .segmentation import InstanceOffsets, segment_mineral_instances


@dataclass
class RawSliceConfig:
    label_raw: str | Path
    shape: tuple[int, int, int]
    slice_index: int = 0
    resize_to: tuple[int, int] | None = None
    dissolution_raw: str | Path | None = None
    cement_raw: str | Path | None = None


def generate_from_label_slice(
    label_slice: np.ndarray,
    particle_libraries: ParticleLibraryPaths,
    output_dir: str | Path,
    sample_name: str = "sample_0001",
    dissolution_mask: np.ndarray | None = None,
    add_clay: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full generation workflow from a 2D mineral label slice.

    The output is directly saved as a semantic-segmentation dataset sample:
    one XPL image, one PPL image, and one single-channel label mask.
    """
    instance_map = segment_mineral_instances(label_slice)
    xpl_image, ppl_image, label_mask = reconstruct_from_instances(
        instance_map,
        particle_libraries,
        offsets=InstanceOffsets(),
    )
    if dissolution_mask is not None:
        xpl_image, ppl_image = apply_dissolution(xpl_image, ppl_image, dissolution_mask)
        xpl_image, ppl_image, _ = smooth_edges_around_mask(xpl_image, ppl_image, dissolution_mask)
    if add_clay:
        xpl_image, ppl_image, _ = add_clay_rims(xpl_image, ppl_image, instance_map)
    save_generated_sample(output_dir, xpl_image, ppl_image, label_mask, sample_name=sample_name)
    return xpl_image, ppl_image, label_mask


def generate_from_raw(
    raw_cfg: RawSliceConfig,
    particle_libraries: ParticleLibraryPaths,
    output_dir: str | Path,
    sample_name: str = "sample_0001",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a raw slice and run the full thin-section generation workflow."""
    label_slice = read_raw_slice(
        raw_cfg.label_raw,
        shape=raw_cfg.shape,
        slice_index=raw_cfg.slice_index,
        resize_to=raw_cfg.resize_to,
    )
    dissolution_mask = None
    if raw_cfg.dissolution_raw is not None:
        fdd = read_raw_slice(
            raw_cfg.dissolution_raw,
            shape=raw_cfg.shape,
            slice_index=raw_cfg.slice_index,
            resize_to=raw_cfg.resize_to,
        )
        dissolution_mask = (fdd == 0) & (label_slice == 2)
    return generate_from_label_slice(
        label_slice,
        particle_libraries,
        output_dir,
        sample_name=sample_name,
        dissolution_mask=dissolution_mask,
    )
