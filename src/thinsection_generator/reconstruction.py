from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .enhancement import fill_pores
from .segmentation import InstanceOffsets, build_class_label, generate_instance_masks, instance_to_class
from .texture import get_matched_texture_pair, rotate_crop_resize_pair


@dataclass
class ParticleLibraryPaths:
    quartz: str | Path
    feldspar: str | Path
    lithic: str | Path
    xpl_folder: str | None = None
    ppl_folder: str | None = None

    def for_class(self, class_id: int) -> str | Path:
        if class_id == 1:
            return self.quartz
        if class_id == 2:
            return self.feldspar
        if class_id == 3:
            return self.lithic
        raise ValueError(f"No particle library for class_id={class_id}.")


def reconstruct_from_instances(
    instance_map: np.ndarray,
    particle_libraries: ParticleLibraryPaths,
    offsets: InstanceOffsets = InstanceOffsets(),
    width_tol: float = 0.1,
    ratio_tol: float = 0.1,
    temperature: float = 0.5,
    scale_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct paired one-angle XPL/PPL thin-section images from an instance map.

    Returns
    -------
    xpl_image : np.ndarray
        RGB cross-polarized-light image.
    ppl_image : np.ndarray
        RGB plane-polarized-light image.
    label_mask : np.ndarray
        Single-channel semantic label mask with values 0/1/2/3.
    """
    height, width = instance_map.shape[:2]
    xpl_image = np.zeros((height, width, 3), dtype=np.uint8)
    ppl_image = np.zeros((height, width, 3), dtype=np.uint8)
    label_mask = build_class_label(instance_map, offsets=offsets)
    instance_masks = generate_instance_masks(instance_map)

    pore_mask = instance_masks.get(0, (label_mask == 0).astype(np.uint8) * 255) == 255
    xpl_image, ppl_image = fill_pores(xpl_image, ppl_image, pore_mask)

    for instance_id, mask in instance_masks.items():
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        class_id = instance_to_class(instance_id, offsets=offsets)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        library_root = particle_libraries.for_class(class_id)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1:
                continue
            xpl_texture, ppl_texture = get_matched_texture_pair(
                library_root,
                w,
                h,
                width_tol=width_tol,
                ratio_tol=ratio_tol,
                temperature=temperature,
                xpl_folder=particle_libraries.xpl_folder,
                ppl_folder=particle_libraries.ppl_folder,
            )
            xpl_patch, ppl_patch = rotate_crop_resize_pair(
                xpl_texture,
                ppl_texture,
                w,
                h,
                scale_factor=scale_factor,
            )
            local_mask = mask[y : y + h, x : x + w] == 255
            target_xpl = xpl_image[y : y + h, x : x + w]
            target_ppl = ppl_image[y : y + h, x : x + w]
            target_xpl[local_mask] = xpl_patch[local_mask]
            target_ppl[local_mask] = ppl_patch[local_mask]
            xpl_image[y : y + h, x : x + w] = target_xpl
            ppl_image[y : y + h, x : x + w] = target_ppl

    return xpl_image, ppl_image, label_mask
