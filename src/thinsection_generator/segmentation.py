from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from skimage import filters, measure, morphology, segmentation


@dataclass(frozen=True)
class InstanceOffsets:
    """Offsets used to separate mineral-instance IDs.

    Quartz instances keep watershed IDs smaller than `feldspar_offset`.
    Feldspar instances are shifted by `feldspar_offset`.
    Lithic-fragment instances are shifted by `lithic_offset`.
    """

    feldspar_offset: int = 500
    lithic_offset: int = 800


def watershed_mask(binary_image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Segment connected mineral regions into instances using watershed."""
    binary_image = binary_image.astype(bool)
    distance = ndi.distance_transform_edt(binary_image)
    distance = filters.gaussian(distance, sigma=sigma)
    local_maxima = morphology.local_maxima(distance)
    markers = measure.label(local_maxima)
    return segmentation.watershed(-distance, markers, mask=binary_image).astype(np.int32)


def segment_mineral_instances(
    label_slice: np.ndarray,
    quartz_id: int = 1,
    feldspar_id: int = 2,
    lithic_id: int = 3,
    quartz_sigma: float = 13.0,
    feldspar_sigma: float = 11.0,
    lithic_sigma: float = 7.0,
    offsets: InstanceOffsets = InstanceOffsets(),
) -> np.ndarray:
    """Generate an instance-labeled map from a mineral-class label slice.

    Expected class values are 0=pore/background, 1=quartz, 2=feldspar,
    and 3=lithic fragments. The returned map uses 0 for pore/background;
    quartz instances use IDs < feldspar_offset, feldspar IDs are shifted by
    feldspar_offset, and lithic IDs are shifted by lithic_offset.
    """
    quartz = watershed_mask(label_slice == quartz_id, sigma=quartz_sigma)
    feldspar = watershed_mask(label_slice == feldspar_id, sigma=feldspar_sigma)
    lithic = watershed_mask(label_slice == lithic_id, sigma=lithic_sigma)

    feldspar[feldspar != 0] += offsets.feldspar_offset
    lithic[lithic != 0] += offsets.lithic_offset
    return quartz + feldspar + lithic


def generate_instance_masks(instance_map: np.ndarray) -> Dict[int, np.ndarray]:
    """Generate one binary mask for each instance ID."""
    masks: Dict[int, np.ndarray] = {}
    for value in np.unique(instance_map).tolist():
        mask = np.zeros_like(instance_map, dtype=np.uint8)
        mask[instance_map == value] = 255
        masks[int(value)] = mask
    return masks


def instance_to_class(instance_id: int, offsets: InstanceOffsets = InstanceOffsets()) -> int:
    """Map an instance ID back to a class label: 0, 1, 2, or 3."""
    if instance_id == 0:
        return 0
    if instance_id < offsets.feldspar_offset:
        return 1
    if instance_id < offsets.lithic_offset:
        return 2
    return 3


def build_class_label(instance_map: np.ndarray, offsets: InstanceOffsets = InstanceOffsets()) -> np.ndarray:
    """Convert an instance map to a 0/1/2/3 semantic label mask."""
    label = np.zeros(instance_map.shape, dtype=np.uint8)
    for value in np.unique(instance_map).tolist():
        label[instance_map == value] = instance_to_class(int(value), offsets)
    return label


def perturb_mask_boundary(mask: np.ndarray, max_offset: int = 3, smooth_iter: int = 1) -> np.ndarray:
    """Randomly perturb the boundary of a binary mask along local normal directions."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask.copy()

    height, width = mask.shape
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        if cnt.shape[0] < 10:
            cv2.drawContours(new_mask, [cnt.reshape(-1, 1, 2)], -1, 255, -1)
            continue
        perturbed = []
        for i, (x, y) in enumerate(cnt):
            x_prev, y_prev = cnt[(i - 1) % len(cnt)]
            x_next, y_next = cnt[(i + 1) % len(cnt)]
            tx, ty = x_next - x_prev, y_next - y_prev
            nx, ny = -ty, tx
            norm = float(np.hypot(nx, ny))
            if norm > 0:
                nx, ny = nx / norm, ny / norm
            delta = np.random.randint(-max_offset, max_offset + 1)
            new_x = int(np.clip(round(x + nx * delta), 0, width - 1))
            new_y = int(np.clip(round(y + ny * delta), 0, height - 1))
            perturbed.append([[new_x, new_y]])
        cv2.drawContours(new_mask, [np.array(perturbed, dtype=np.int32)], -1, 255, -1)

    if smooth_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(smooth_iter):
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
    return new_mask


def smooth_mask_boundary(mask: np.ndarray, sigma: float = 2.0, min_pts: int = 10) -> np.ndarray:
    """Smooth mask contours with a 1D Gaussian filter along contour points."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    height, width = mask.shape
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < min_pts:
            cv2.drawContours(new_mask, [pts.reshape(-1, 1, 2)], -1, 255, -1)
            continue
        xs = gaussian_filter1d(pts[:, 0].astype(np.float32), sigma=sigma, mode="wrap")
        ys = gaussian_filter1d(pts[:, 1].astype(np.float32), sigma=sigma, mode="wrap")
        xs = np.clip(np.round(xs).astype(np.int32), 0, width - 1)
        ys = np.clip(np.round(ys).astype(np.int32), 0, height - 1)
        smooth_pts = np.stack([xs, ys], axis=1)

        filtered = [smooth_pts[0]]
        for p in smooth_pts[1:]:
            if not np.array_equal(p, filtered[-1]):
                filtered.append(p)
        if len(filtered) > 1 and np.array_equal(filtered[0], filtered[-1]):
            filtered.pop()
        filtered = np.array(filtered, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(new_mask, [filtered], -1, 255, -1)
    return new_mask
