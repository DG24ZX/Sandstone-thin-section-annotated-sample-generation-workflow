"""
tools_clean.py

Cleaned utility functions for sandstone thin-section annotated sample generation.

This file was refactored from the original notebook workflow and tools_v2.py.
Only the functions required for the manuscript workflow are kept:

1. Read digital-core raw slices.
2. Segment mineral regions into independent particle instances.
3. Select one XPL texture and one PPL texture from the mineral particle library.
4. Reconstruct one XPL image, one PPL image, and one single-channel label mask.
5. Apply simulation-based enhancement, including dissolution pores, edge smoothing,
   and clay-mineral cementation rims.
6. Save outputs in semantic-segmentation dataset format.

Default output:
    xpl/sample_xxxx.png
    ppl/sample_xxxx.png
    labels/sample_xxxx.png

Label values:
    0 = pore/background
    1 = quartz
    2 = feldspar
    3 = lithic fragments
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from skimage import filters, measure, morphology, segmentation


# ---------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------

def read_raw_slice(
    raw_path: str | Path,
    shape: Tuple[int, int, int],
    slice_index: int = 0,
    dtype=np.uint8,
    resize_to: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Read one 2D slice from a 3D raw digital-core file.

    Parameters
    ----------
    raw_path : str or Path
        Path to the raw file.
    shape : tuple
        Raw volume shape in (z, y, x) order.
    slice_index : int
        Slice index along the z direction.
    dtype : numpy dtype
        Data type of the raw file, usually np.uint8.
    resize_to : tuple or None
        Output size in (width, height). Nearest-neighbor interpolation is used.

    Returns
    -------
    slice_img : np.ndarray
        2D label slice.
    """
    raw_path = Path(raw_path)
    data = np.fromfile(raw_path, dtype=dtype)
    expected_size = int(np.prod(shape))
    if data.size != expected_size:
        raise ValueError(
            f"Raw size mismatch: {raw_path}\n"
            f"Expected {expected_size} elements from shape {shape}, got {data.size}."
        )

    volume = data.reshape(shape)
    if not (0 <= slice_index < shape[0]):
        raise IndexError(f"slice_index {slice_index} is out of range for z={shape[0]}.")

    slice_img = volume[slice_index].copy()

    if resize_to is not None:
        slice_img = cv2.resize(slice_img, resize_to, interpolation=cv2.INTER_NEAREST)

    return slice_img


def save_segmentation_sample(
    output_dir: str | Path,
    sample_name: str,
    xpl_image: np.ndarray,
    ppl_image: np.ndarray,
    label_mask: np.ndarray,
) -> None:
    """
    Save one generated sample for semantic segmentation.

    The images are assumed to be RGB arrays in memory.
    They are saved as PNG files:
        output_dir/xpl/sample_name.png
        output_dir/ppl/sample_name.png
        output_dir/labels/sample_name.png
    """
    output_dir = Path(output_dir)
    xpl_dir = output_dir / "xpl"
    ppl_dir = output_dir / "ppl"
    label_dir = output_dir / "labels"

    xpl_dir.mkdir(parents=True, exist_ok=True)
    ppl_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # RGB -> BGR for cv2.imwrite
    cv2.imwrite(str(xpl_dir / f"{sample_name}.png"), cv2.cvtColor(xpl_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(ppl_dir / f"{sample_name}.png"), cv2.cvtColor(ppl_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(label_dir / f"{sample_name}.png"), label_mask.astype(np.uint8))


# ---------------------------------------------------------------------
# Digital-core preprocessing and particle-instance segmentation
# ---------------------------------------------------------------------

def build_diagenetic_masks(
    rock_slice: np.ndarray,
    fdd_slice: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build masks used in the original notebook workflow.

    Parameters
    ----------
    rock_slice : np.ndarray
        Original digital-core label slice.
    fdd_slice : np.ndarray or None
        Feldspar dissolution/cementation slice. If provided:
            fdd_slice == 4 is treated as cementation mask.
            fdd_slice == 0 and rock_slice == 2 is treated as feldspar dissolution mask.

    Returns
    -------
    rock_base : np.ndarray
        Modified rock slice used for subsequent particle segmentation.
    mask_cem : np.ndarray
        Cementation mask, uint8 0/1.
    mask_por1 : np.ndarray
        Feldspar dissolution mask, uint8 0/1.
    """
    rock_base = rock_slice.copy()

    if fdd_slice is None:
        mask_cem = np.zeros_like(rock_slice, dtype=np.uint8)
        mask_por1 = np.zeros_like(rock_slice, dtype=np.uint8)
        return rock_base, mask_cem, mask_por1

    # Match the original notebook logic.
    mask_cem = (fdd_slice == 4).astype(np.uint8)
    mask_por1 = ((fdd_slice == 0) & (rock_slice == 2)).astype(np.uint8)

    # In the original workflow, cementation regions were merged into the grain framework.
    rock_base[mask_cem == 1] = 1

    return rock_base, mask_cem, mask_por1


def watershedMask(binary_image: np.ndarray, sigma: float = 3) -> np.ndarray:
    """
    Watershed segmentation for one mineral class.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary mask of one mineral class.
    sigma : float
        Gaussian smoothing sigma for the distance transform.

    Returns
    -------
    segmented_image : np.ndarray
        Instance-labeled image. Background is 0.
    """
    binary_image = binary_image.astype(bool)
    distance = ndi.distance_transform_edt(binary_image)
    distance = filters.gaussian(distance, sigma=sigma)
    local_maxi = morphology.local_maxima(distance)
    markers = measure.label(local_maxi)
    segmented_image = segmentation.watershed(-distance, markers, mask=binary_image)
    return segmented_image.astype(np.int32)


def segment_mineral_particles(
    rock_slice: np.ndarray,
    quartz_value: int = 1,
    feldspar_value: int = 2,
    lithic_value: int = 3,
    num1: int = 500,
    num2: int = 800,
    sigma_quartz: float = 13,
    sigma_feldspar: float = 11,
    sigma_lithic: float = 7,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Segment quartz, feldspar, and lithic regions into independent particle instances.

    The instance ID convention follows the original notebook:
        quartz instances:        1, 2, ...
        feldspar instances:      500 + instance_id
        lithic instances:        800 + instance_id

    Returns
    -------
    rock_slice_segmented : np.ndarray
        Instance-labeled digital-core slice.
    segmented_parts : dict
        Individual instance maps for quartz, feldspar, and lithic fragments.
    """
    mask_quartz = rock_slice == quartz_value
    mask_feldspar = rock_slice == feldspar_value
    mask_lithic = rock_slice == lithic_value

    segmented_quartz = watershedMask(mask_quartz, sigma=sigma_quartz)
    segmented_feldspar = watershedMask(mask_feldspar, sigma=sigma_feldspar)
    segmented_lithic = watershedMask(mask_lithic, sigma=sigma_lithic)

    segmented_feldspar[segmented_feldspar != 0] += num1
    segmented_lithic[segmented_lithic != 0] += num2

    rock_slice_segmented = segmented_quartz + segmented_feldspar + segmented_lithic

    return rock_slice_segmented.astype(np.int32), {
        "quartz": segmented_quartz.astype(np.int32),
        "feldspar": segmented_feldspar.astype(np.int32),
        "lithic": segmented_lithic.astype(np.int32),
    }


def generateMasks(rockSlice_segmented: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Generate one binary mask for each particle instance.

    Parameters
    ----------
    rockSlice_segmented : np.ndarray
        Instance-labeled image.

    Returns
    -------
    particle_masks : dict
        {instance_id: binary_mask_uint8_0_255}
    """
    particle_masks = {}
    for value in np.unique(rockSlice_segmented).tolist():
        mask = np.zeros_like(rockSlice_segmented, dtype=np.uint8)
        mask[rockSlice_segmented == value] = 255
        particle_masks[int(value)] = mask
    return particle_masks


def build_label_mask(
    particle_masks: Dict[int, np.ndarray],
    num1: int = 500,
    num2: int = 800,
) -> np.ndarray:
    """
    Build a single-channel semantic label mask from particle masks.

    Label values:
        0 = pore/background
        1 = quartz
        2 = feldspar
        3 = lithic fragments
    """
    first_mask = next(iter(particle_masks.values()))
    label_mask = np.zeros(first_mask.shape, dtype=np.uint8)

    for value, mask in particle_masks.items():
        value = int(value)
        region = mask == 255
        if value == 0:
            label_mask[region] = 0
        elif value < num1:
            label_mask[region] = 1
        elif num1 <= value < num2:
            label_mask[region] = 2
        else:
            label_mask[region] = 3

    return label_mask


# ---------------------------------------------------------------------
# Boundary perturbation and smoothing
# ---------------------------------------------------------------------

def perturb_mask_boundary(mask: np.ndarray, max_offset: int = 3, smooth_iter: int = 1) -> np.ndarray:
    """
    Randomly perturb a binary mask boundary along approximate normal directions.
    This function is kept from the original notebook logic.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    H, W = mask.shape
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return mask.copy()

    new_mask = np.zeros_like(mask, dtype=np.uint8)

    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        num_pts = cnt.shape[0]
        if num_pts < 10:
            cv2.drawContours(new_mask, [cnt.reshape(-1, 1, 2)], -1, 255, -1)
            continue

        perturbed = []
        for i in range(num_pts):
            x, y = cnt[i]
            x_prev, y_prev = cnt[(i - 1) % num_pts]
            x_next, y_next = cnt[(i + 1) % num_pts]

            tx = x_next - x_prev
            ty = y_next - y_prev
            nx, ny = -ty, tx
            norm = np.sqrt(nx * nx + ny * ny)
            if norm != 0:
                nx /= norm
                ny /= norm
            else:
                nx, ny = 0, 0

            d = random.randint(-max_offset, max_offset)
            dx = int(round(nx * d))
            dy = int(round(ny * d))

            new_x = max(0, min(W - 1, x + dx))
            new_y = max(0, min(H - 1, y + dy))
            perturbed.append([[new_x, new_y]])

        perturbed_cnt = np.array(perturbed, dtype=np.int32)
        cv2.drawContours(new_mask, [perturbed_cnt], -1, 255, thickness=-1)

    if smooth_iter > 0:
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(smooth_iter):
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel_small)
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel_small)

    return new_mask


def smooth_mask_boundary(mask: np.ndarray, sigma: float = 2, min_pts: int = 10) -> np.ndarray:
    """
    Smooth the boundary of a binary mask using Gaussian filtering along contour points.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        num_pts = cnt.shape[0]
        if num_pts < min_pts:
            cv2.drawContours(new_mask, [cnt.reshape(-1, 1, 2)], -1, 255, thickness=-1)
            continue

        xs = cnt[:, 0].astype(np.float32)
        ys = cnt[:, 1].astype(np.float32)

        xs_smooth = gaussian_filter1d(xs, sigma=sigma, mode="wrap")
        ys_smooth = gaussian_filter1d(ys, sigma=sigma, mode="wrap")

        xs_int = np.clip(np.round(xs_smooth).astype(np.int32), 0, W - 1)
        ys_int = np.clip(np.round(ys_smooth).astype(np.int32), 0, H - 1)

        new_pts = np.stack([xs_int, ys_int], axis=1)

        filtered = [new_pts[0]]
        for i in range(1, new_pts.shape[0]):
            if not (new_pts[i][0] == new_pts[i - 1][0] and new_pts[i][1] == new_pts[i - 1][1]):
                filtered.append(new_pts[i])

        if len(filtered) > 1 and filtered[0][0] == filtered[-1][0] and filtered[0][1] == filtered[-1][1]:
            filtered.pop()

        filtered = np.array(filtered, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(new_mask, [filtered], -1, 255, thickness=-1)

    return new_mask


def smooth_particle_masks(
    particle_masks: Dict[int, np.ndarray],
    kernel_size: int = 5,
    blur_kernel: int = 11,
    blur_sigma: float = 5,
    perturb: bool = False,
    max_offset: int = 3,
) -> Dict[int, np.ndarray]:
    """
    Apply optional morphological smoothing and boundary perturbation to particle masks.
    This is a compact wrapper around the logic previously written in notebook cells.
    """
    out = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for value, mask in particle_masks.items():
        mask = mask.astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        blur = cv2.GaussianBlur(mask.astype(np.float32), (blur_kernel, blur_kernel), sigmaX=blur_sigma)
        _, mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        if perturb and value != 0:
            mask = perturb_mask_boundary(mask, max_offset=max_offset, smooth_iter=1)
            mask = smooth_mask_boundary(mask, sigma=2)

        out[int(value)] = mask.astype(np.uint8)

    return out


# ---------------------------------------------------------------------
# Texture selection and preprocessing
# ---------------------------------------------------------------------

MEMORY_CACHE: Dict[str, Dict[str, Tuple[int, int]]] = {}


def get_or_build_cache(image_folder: str | Path) -> Dict[str, Tuple[int, int]]:
    """
    Get or build an image-size cache for a texture folder.
    """
    global MEMORY_CACHE

    image_folder = str(Path(image_folder).resolve())
    if image_folder in MEMORY_CACHE:
        return MEMORY_CACHE[image_folder]

    cache_dir = Path(image_folder) / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "image_cache.json"

    disk_cache = {}
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                disk_cache = json.load(f)
        except Exception:
            disk_cache = {}

    memory_cache = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_folder_path = Path(image_folder)

    for img_name in os.listdir(image_folder):
        img_path = image_folder_path / img_name
        if img_path.suffix.lower() not in valid_extensions:
            continue

        try:
            img_mtime = os.path.getmtime(img_path)
        except OSError:
            continue

        if img_name in disk_cache and disk_cache[img_name].get("mtime") == img_mtime:
            memory_cache[img_name] = (
                disk_cache[img_name]["width"],
                disk_cache[img_name]["height"],
            )
        else:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                memory_cache[img_name] = (w, h)
                disk_cache[img_name] = {"width": w, "height": h, "mtime": img_mtime}

    try:
        temp_cache_path = cache_path.with_suffix(".tmp")
        with temp_cache_path.open("w", encoding="utf-8") as f:
            json.dump(disk_cache, f, indent=4, ensure_ascii=False)
        temp_cache_path.replace(cache_path)
    except Exception:
        pass

    MEMORY_CACHE[image_folder] = memory_cache
    return memory_cache


def calculate_score(
    w_img: int,
    h_img: int,
    w_target: int,
    h_target: int,
    width_tol: float,
    ratio_tol: float,
) -> float:
    """
    Geometric matching score. Smaller values indicate better matching.
    """
    if width_tol <= 0 or ratio_tol <= 0:
        raise ValueError("width_tol and ratio_tol must be > 0.")

    width_diff = abs(w_img - w_target) / (width_tol * max(w_target, 1))
    ratio_diff = abs(w_img * h_target - h_img * w_target) / (
        ratio_tol * max(w_target * h_target, 1)
    )
    return float(width_diff + ratio_diff)


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_image(
    image_path: str | Path,
    w_target: int,
    h_target: int,
    width_tol: float = 0.1,
    ratio_tol: float = 0.1,
    temperature: float = 0.5,
    xpl_folder: str = "cross_1",
    ppl_folder: str = "single",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select one matched texture pair from a mineral particle library.

    This cleaned version uses only one XPL image and one PPL image.

    Folder structure:
        image_path/
            cross_1/  or xpl/
            single/   or ppl/

    Returns
    -------
    xpl_img_rgb, ppl_img_rgb : np.ndarray
        Selected XPL and PPL texture images in RGB format.
    """
    image_path = Path(image_path)

    xpl_dir = image_path / xpl_folder
    ppl_dir = image_path / ppl_folder

    if not xpl_dir.exists():
        raise FileNotFoundError(f"XPL folder does not exist: {xpl_dir}")
    if not ppl_dir.exists():
        raise FileNotFoundError(f"PPL folder does not exist: {ppl_dir}")

    cache = get_or_build_cache(xpl_dir)
    if not cache:
        raise FileNotFoundError(f"No texture images found in: {xpl_dir}")

    scores = {}
    for img_name, (w_img, h_img) in cache.items():
        scores[img_name] = calculate_score(w_img, h_img, w_target, h_target, width_tol, ratio_tol)

    weights = {name: math.exp(-score / temperature) for name, score in scores.items()}
    images = list(weights.keys())
    wts = list(weights.values())
    selected_image = random.choices(images, weights=wts, k=1)[0] if sum(wts) > 0 else random.choice(images)

    xpl_path = xpl_dir / selected_image
    ppl_path = ppl_dir / selected_image

    xpl_img = _read_rgb(xpl_path)
    ppl_img = _read_rgb(ppl_path)

    return xpl_img, ppl_img


def largest_rotated_rect(w: int, h: int, angle_rad: float) -> Tuple[int, int]:
    """
    Calculate the largest axis-aligned rectangle inside a rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    if angle_rad == 0:
        return w, h

    angle_rad = angle_rad % (np.pi / 2)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    numerator = w * h
    denominator_w = h * sin_a + w * cos_a
    denominator_h = w * sin_a + h * cos_a

    if denominator_w == 0 or denominator_h == 0:
        return 0, 0

    rect_w = min(numerator / denominator_w, w)
    rect_h = min(numerator / denominator_h, h)

    return int(rect_w), int(rect_h)


def process_and_crop_image(
    fill_images: Iterable[np.ndarray],
    target_width: int,
    target_height: int,
    scale_factor: float = 2.0,
    sharpen_xpl: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same random rotation/cropping to one XPL/PPL texture pair.

    Parameters
    ----------
    fill_images : iterable
        [xpl_img, ppl_img]
    target_width, target_height : int
        Output patch size.
    scale_factor : float
        Upscaling factor before random rotation.
    sharpen_xpl : bool
        If True, sharpen only the first image, which is assumed to be XPL.

    Returns
    -------
    xpl_patch, ppl_patch : np.ndarray
        Processed texture patches.
    """
    fill_images = list(fill_images)
    if len(fill_images) != 2:
        raise ValueError("fill_images must contain exactly two images: [xpl_img, ppl_img].")

    angle = random.uniform(0, 180)
    angle_rad = np.deg2rad(abs(angle))
    processed_images = []

    for idx, img in enumerate(fill_images):
        if img is None:
            raise ValueError("Input image is None.")

        resize_w = int(target_width * scale_factor)
        resize_h = int(target_height * scale_factor)
        img_resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)

        h, w = img_resized.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(
            img_resized,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        crop_w, crop_h = largest_rotated_rect(w, h, angle_rad)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = img_rotated[y1:y1 + crop_h, x1:x1 + crop_w]

        final = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        if sharpen_xpl and idx == 0:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            final = cv2.filter2D(final, -1, kernel)

        processed_images.append(final)

    return processed_images[0], processed_images[1]


# ---------------------------------------------------------------------
# Pore texture generation and thin-section reconstruction
# ---------------------------------------------------------------------

def adjust_brightness_random(color, min_factor: float = 0.7, max_factor: float = 1.3) -> np.ndarray:
    """
    Randomly adjust brightness of an RGB color.
    """
    color = np.array(color, dtype=float)
    brightness_factor = np.random.uniform(min_factor, max_factor)
    adjusted_color = np.clip(color * brightness_factor, 0, 255)
    return adjusted_color.astype(np.uint8)


def generate_ppl_pore_texture(
    img_size: Tuple[int, int],
    bg_color=(199, 220, 199),
    num_dark_spots: int = 2000,
    num_light_spots: int = 4000,
    max_radius: int = 2,
    noise_var: float = 3,
    blur_kernel=(3, 3),
    contrast: float = 1.0,
    brightness: int = 0,
) -> np.ndarray:
    """
    Generate PPL pore texture: light gray-green background with weak spots and noise.
    """
    height, width = img_size
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    for _ in range(num_dark_spots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        axes = (random.randint(1, max_radius), random.randint(1, max_radius))
        angle = random.randint(0, 360)
        cv2.ellipse(img, (x, y), axes, angle, 0, 360, (40, 40, 40), -1)

    for _ in range(num_light_spots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(1, max_radius)
        cv2.circle(img, (x, y), radius, (160, 180, 150), -1)

    sigma = noise_var ** 0.5
    noise = np.random.normal(0, sigma, (height, width, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img = cv2.GaussianBlur(img, blur_kernel, 0)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    return img



def generate_xpl_pore_texture(
    img_size: Tuple[int, int],
    mean_rgb=(0, 0, 0),
    std_rgb=(0, 0, 0),
    num_dark_spots: int = 0,
    num_light_spots: int = 0,
    max_radius: int = 0,
    blur_kernel=(3, 3),
) -> np.ndarray:
    """
    Generate XPL pore texture.

    In the public demonstration workflow, pores/background under XPL are
    represented as pure black. This avoids artificial speckled textures in
    pore regions and in feldspar-dissolution pores.
    """
    h, w = img_size
    return np.zeros((h, w, 3), dtype=np.uint8)

def _get_library_for_instance(
    value: int,
    quartz_dir: str | Path,
    feldspar_dir: str | Path,
    lithic_dir: str | Path,
    num1: int,
    num2: int,
) -> Path:
    if value < num1:
        return Path(quartz_dir)
    if num1 <= value < num2:
        return Path(feldspar_dir)
    return Path(lithic_dir)


def reconstruct_single_xpl_ppl(
    rockSlice_segmented: np.ndarray,
    particle_masks: Dict[int, np.ndarray],
    quartz_dir: str | Path,
    feldspar_dir: str | Path,
    lithic_dir: str | Path,
    num1: int = 500,
    num2: int = 800,
    xpl_folder: str = "cross_1",
    ppl_folder: str = "single",
    width_tol: float = 0.1,
    ratio_tol: float = 0.1,
    temperature: float = 0.5,
    pore_bg_color=(199, 220, 199),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct one XPL image, one PPL image, and one single-channel label mask.

    This function corresponds to the texture filling step in the notebook.
    """
    H, W = rockSlice_segmented.shape[:2]

    thin_xpl = np.zeros((H, W, 3), dtype=np.uint8)
    thin_ppl = np.zeros((H, W, 3), dtype=np.uint8)
    label_mask = build_label_mask(particle_masks, num1=num1, num2=num2)

    # Fill pores/background first.
    if 0 in particle_masks:
        pore_mask = particle_masks[0] == 255
    else:
        pore_mask = label_mask == 0

    ppl_pore_tex = generate_ppl_pore_texture(
        img_size=(H, W),
        bg_color=(
            random.randint(max(0, pore_bg_color[0] - 8), min(255, pore_bg_color[0] + 8)),
            random.randint(max(0, pore_bg_color[1] - 8), min(255, pore_bg_color[1] + 8)),
            random.randint(max(0, pore_bg_color[2] - 8), min(255, pore_bg_color[2] + 8)),
        ),
        num_dark_spots=random.randint(1600, 2400),
        num_light_spots=random.randint(3200, 4800),
        max_radius=random.randint(1, 2),
        noise_var=random.uniform(2.0, 4.0),
        blur_kernel=random.choice([(3, 3), (5, 5)]),
        contrast=random.uniform(0.98, 1.03),
        brightness=random.randint(-3, 3),
    )

    xpl_pore_tex = generate_xpl_pore_texture(
        img_size=(H, W),
        mean_rgb=(random.randint(22, 28), random.randint(23, 29), random.randint(17, 23)),
        std_rgb=(random.randint(4, 6), random.randint(4, 6), random.randint(4, 6)),
        num_dark_spots=random.randint(800, 1300),
        num_light_spots=random.randint(180, 320),
        max_radius=random.randint(1, 2),
        blur_kernel=random.choice([(3, 3), (5, 5)]),
    )

    thin_ppl[pore_mask] = ppl_pore_tex[pore_mask]
    thin_xpl[pore_mask] = xpl_pore_tex[pore_mask]

    # Fill mineral particles.
    for value, mask in particle_masks.items():
        value = int(value)
        if value == 0:
            continue

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        library_dir = _get_library_for_instance(value, quartz_dir, feldspar_dir, lithic_dir, num1, num2)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            xpl_tex, ppl_tex = get_image(
                library_dir,
                w,
                h,
                width_tol=width_tol,
                ratio_tol=ratio_tol,
                temperature=temperature,
                xpl_folder=xpl_folder,
                ppl_folder=ppl_folder,
            )
            xpl_patch, ppl_patch = process_and_crop_image([xpl_tex, ppl_tex], w, h)

            local_mask = mask[y:y + h, x:x + w] == 255

            target_xpl = thin_xpl[y:y + h, x:x + w]
            target_ppl = thin_ppl[y:y + h, x:x + w]

            target_xpl[local_mask] = xpl_patch[local_mask]
            target_ppl[local_mask] = ppl_patch[local_mask]

            thin_xpl[y:y + h, x:x + w] = target_xpl
            thin_ppl[y:y + h, x:x + w] = target_ppl

    return thin_xpl, thin_ppl, label_mask


# ---------------------------------------------------------------------
# Simulation-based enhancement
# ---------------------------------------------------------------------


def apply_dissolution_effect(
    thin_xpl: np.ndarray,
    thin_ppl: np.ndarray,
    dissolution_mask: np.ndarray,
    ppl_base_color=(199, 220, 199),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply feldspar-dissolution pore effect.

    The dissolved feldspar regions are converted to pore/background:
        - XPL: pure black.
        - PPL: the same pore-background color used in the reconstructed PPL image.

    Only a narrow edge transition is smoothed. No random mineral texture is
    filled into the dissolution pores.
    """
    out_xpl = thin_xpl.copy()
    out_ppl = thin_ppl.copy()

    mask = dissolution_mask.astype(bool)
    if not np.any(mask):
        return out_xpl, out_ppl

    # XPL: dissolution pores are optically dark.
    out_xpl[mask] = np.array([0, 0, 0], dtype=np.uint8)

    # PPL: dissolution pores use the pore/background color, not a new texture.
    pore_color = np.array(ppl_base_color, dtype=np.uint8)
    out_ppl[mask] = pore_color

    # Smooth only the immediate dissolution boundary to avoid hard stair-step edges.
    mask_u8 = mask.astype(np.uint8) * 255
    edge = cv2.Canny(mask_u8, 50, 150)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)

    if np.any(edge):
        blur_xpl = cv2.GaussianBlur(out_xpl, (5, 5), 0)
        blur_ppl = cv2.GaussianBlur(out_ppl, (5, 5), 0)
        edge_bool = edge > 0
        out_xpl[edge_bool] = blur_xpl[edge_bool]
        out_ppl[edge_bool] = blur_ppl[edge_bool]

    return out_xpl, out_ppl

def porEdgeSmooth(
    thin_cross_por1: np.ndarray,
    thin_single_por1: np.ndarray,
    rockSlice_por1: np.ndarray,
    crossGaussian=(9, 9),
    singleGaussian=(9, 9),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth edges around pore/dissolution regions.
    Kept from the original tools_v2.py logic.
    """
    mask_edge = np.zeros_like(rockSlice_por1, dtype=np.uint8)
    mask_edge[rockSlice_por1 >= 1] = 255
    edges = cv2.Canny(mask_edge, threshold1=100, threshold2=200)

    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    out_cross = thin_cross_por1.copy()
    out_single = thin_single_por1.copy()

    smoothed_cross = cv2.GaussianBlur(out_cross, crossGaussian, 0)
    out_cross[edges_dilated == 255] = smoothed_cross[edges_dilated == 255]

    smoothed_single = cv2.GaussianBlur(out_single, singleGaussian, 0)
    out_single[edges_dilated == 255] = smoothed_single[edges_dilated == 255]

    return out_cross, out_single, edges_dilated



def addClay(
    thin_cross: np.ndarray,
    thin_single: np.ndarray,
    edges: np.ndarray,
    colorValue=(169, 166, 171),
    edges_max: int = 10,
    fill_probability: float = 0.35,
    color_scale: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add clay-mineral/cementation rims along grain boundaries.

    This revised version generates a continuous, softly blended rim instead
    of isolated salt-and-pepper pixels. The PPL rim is produced by alpha
    blending a clay-like color with the original image. The XPL rim is
    slightly darkened, while pore regions remain black when they have already
    been set to black.
    """
    edge_binary = (edges > 0).astype(np.uint8)
    if not np.any(edge_binary):
        empty = np.zeros_like(edges, dtype=np.uint8)
        return thin_cross.copy(), thin_single.copy(), empty

    # Convert the edge map into a continuous rim band.
    # The original call uses edges_max=13. A quarter of this value gives a
    # moderate rim width suitable for 1200 x 1200 demonstration images.
    rim_width = max(2, int(round(edges_max / 4)))
    kernel_size = rim_width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    rim_mask = cv2.dilate(edge_binary, kernel, iterations=1)
    rim_mask = cv2.morphologyEx(rim_mask, cv2.MORPH_CLOSE, kernel)

    # Soft alpha mask for gradual transition.
    rim_alpha = cv2.GaussianBlur(rim_mask.astype(np.float32), (0, 0), sigmaX=max(1.5, rim_width))
    rim_alpha = rim_alpha / (rim_alpha.max() + 1e-8)

    # Low-frequency variation only. This avoids the previous discrete-dot appearance.
    low_freq_noise = np.random.normal(0, 1, size=edges.shape).astype(np.float32)
    low_freq_noise = cv2.GaussianBlur(low_freq_noise, (0, 0), sigmaX=18)
    low_freq_noise = (low_freq_noise - low_freq_noise.min()) / (
        low_freq_noise.max() - low_freq_noise.min() + 1e-8
    )

    out_cross = thin_cross.astype(np.float32).copy()
    out_single = thin_single.astype(np.float32).copy()

    clay_color = np.clip(np.array(colorValue, dtype=np.float32) * color_scale, 0, 255)

    # PPL: continuous clay/cementation rim by color blending.
    # Strength is intentionally moderate to avoid black, speckled boundaries.
    ppl_strength = (0.30 + 0.12 * low_freq_noise)[..., None] * rim_alpha[..., None]
    out_single = out_single * (1.0 - ppl_strength) + clay_color[None, None, :] * ppl_strength

    # XPL: cementation/rim areas are slightly darkened and kept continuous.
    xpl_strength = (0.20 + 0.10 * low_freq_noise)[..., None] * rim_alpha[..., None]
    out_cross = out_cross * (1.0 - xpl_strength)

    out_cross = np.clip(out_cross, 0, 255).astype(np.uint8)
    out_single = np.clip(out_single, 0, 255).astype(np.uint8)

    # Return a compact binary mask for optional label update.
    clay_mask = (rim_alpha > 0.20).astype(np.uint8)

    return out_cross, out_single, clay_mask

def get_instance_edges(rockSlice_segmented: np.ndarray, threshold1: int = 1, threshold2: int = 2) -> np.ndarray:
    """
    Extract grain-boundary edges from the instance-labeled slice.
    """
    mask_edge = cv2.normalize(rockSlice_segmented, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(mask_edge, threshold1=threshold1, threshold2=threshold2)
    return edges


def apply_simulation_enhancement(
    thin_xpl: np.ndarray,
    thin_ppl: np.ndarray,
    rockSlice_segmented: np.ndarray,
    dissolution_mask: Optional[np.ndarray] = None,
    ppl_pore_color=(199, 220, 199),
    add_clay: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply dissolution, edge smoothing, and clay-rim enhancement.
    """
    out_xpl = thin_xpl.copy()
    out_ppl = thin_ppl.copy()
    masks = {}

    rock_for_smoothing = (rockSlice_segmented > 0).astype(np.uint8)

    if dissolution_mask is not None and np.any(dissolution_mask):
        out_xpl, out_ppl = apply_dissolution_effect(
            out_xpl, out_ppl, dissolution_mask, ppl_base_color=ppl_pore_color
        )
        rock_for_smoothing = rock_for_smoothing.copy()
        rock_for_smoothing[dissolution_mask.astype(bool)] = 0
        out_xpl, out_ppl, dissolution_edges = porEdgeSmooth(
            out_xpl, out_ppl, rock_for_smoothing, crossGaussian=(5, 5), singleGaussian=(5, 5)
        )
        masks["dissolution_edges"] = dissolution_edges

    if add_clay:
        edges = get_instance_edges(rockSlice_segmented)
        out_xpl, out_ppl, clay_mask = addClay(
            out_xpl,
            out_ppl,
            edges,
            colorValue=(169, 166, 171),
            edges_max=13,
            fill_probability=0.35,
            color_scale=0.85,
        )
        masks["clay_mask"] = clay_mask

    return out_xpl, out_ppl, masks
