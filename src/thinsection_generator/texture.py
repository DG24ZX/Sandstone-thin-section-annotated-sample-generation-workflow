from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

_MEMORY_CACHE: dict[str, dict[str, tuple[int, int]]] = {}


def resolve_texture_folder(library_root: str | Path, candidates: Sequence[str]) -> Path:
    """Return the first existing texture subfolder from candidate names."""
    root = Path(library_root)
    for name in candidates:
        folder = root / name
        if folder.exists():
            return folder
    candidate_text = ", ".join(candidates)
    raise FileNotFoundError(f"None of the texture folders exists under {root}: {candidate_text}")


def build_image_size_cache(image_folder: str | Path) -> dict[str, tuple[int, int]]:
    """Return cached image sizes for a folder of particle images."""
    folder = Path(image_folder).resolve()
    key = str(folder)
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]

    cache_dir = folder / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "image_cache.json"
    disk_cache: dict[str, dict[str, float | int]] = {}
    if cache_path.exists():
        try:
            disk_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            disk_cache = {}

    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    memory_cache: dict[str, tuple[int, int]] = {}
    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in valid_extensions:
            continue
        mtime = img_path.stat().st_mtime
        record = disk_cache.get(img_path.name)
        if record and record.get("mtime") == mtime:
            memory_cache[img_path.name] = (int(record["width"]), int(record["height"]))
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        height, width = img.shape[:2]
        memory_cache[img_path.name] = (width, height)
        disk_cache[img_path.name] = {"width": width, "height": height, "mtime": mtime}

    cache_path.write_text(json.dumps(disk_cache, indent=2, ensure_ascii=False), encoding="utf-8")
    _MEMORY_CACHE[key] = memory_cache
    return memory_cache


def calculate_match_score(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
    width_tol: float = 0.1,
    ratio_tol: float = 0.1,
) -> float:
    """Calculate the geometry-matching score between candidate and target particles."""
    if width_tol <= 0 or ratio_tol <= 0:
        raise ValueError("width_tol and ratio_tol must be positive.")
    width_diff = abs(image_width - target_width) / (width_tol * target_width)
    ratio_diff = abs(image_width * target_height - image_height * target_width) / (
        ratio_tol * target_width * target_height
    )
    return float(width_diff + ratio_diff)


def select_texture_name(
    library_root: str | Path,
    target_width: int,
    target_height: int,
    width_tol: float = 0.1,
    ratio_tol: float = 0.1,
    temperature: float = 0.5,
    xpl_folder: str | None = None,
) -> str:
    """Select a particle texture filename by temperature-weighted geometric matching."""
    candidates = [xpl_folder] if xpl_folder else ["xpl", "cross", "cross_1"]
    reference_dir = resolve_texture_folder(library_root, [c for c in candidates if c])
    cache = build_image_size_cache(reference_dir)
    if not cache:
        raise FileNotFoundError(f"No particle images found in {reference_dir}")

    names = list(cache.keys())
    scores = [
        calculate_match_score(w, h, target_width, target_height, width_tol, ratio_tol)
        for w, h in cache.values()
    ]
    weights = [math.exp(-score / temperature) for score in scores]
    return random.choices(names, weights=weights, k=1)[0]


def load_texture_pair(
    library_root: str | Path,
    image_name: str,
    xpl_folder: str | None = None,
    ppl_folder: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one matched XPL image and the corresponding PPL image as RGB arrays."""
    library_root = Path(library_root)
    xpl_dir = resolve_texture_folder(library_root, [xpl_folder] if xpl_folder else ["xpl", "cross", "cross_1"])
    ppl_dir = resolve_texture_folder(library_root, [ppl_folder] if ppl_folder else ["ppl", "single"])

    xpl_path = xpl_dir / image_name
    ppl_path = ppl_dir / image_name
    xpl = cv2.imread(str(xpl_path), cv2.IMREAD_COLOR)
    ppl = cv2.imread(str(ppl_path), cv2.IMREAD_COLOR)
    if xpl is None:
        raise FileNotFoundError(f"Cannot read XPL texture: {xpl_path}")
    if ppl is None:
        raise FileNotFoundError(f"Cannot read PPL texture: {ppl_path}")
    return cv2.cvtColor(xpl, cv2.COLOR_BGR2RGB), cv2.cvtColor(ppl, cv2.COLOR_BGR2RGB)


def get_matched_texture_pair(
    library_root: str | Path,
    target_width: int,
    target_height: int,
    width_tol: float = 0.1,
    ratio_tol: float = 0.1,
    temperature: float = 0.5,
    xpl_folder: str | None = None,
    ppl_folder: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and load a matched XPL/PPL particle texture pair."""
    image_name = select_texture_name(
        library_root,
        target_width,
        target_height,
        width_tol=width_tol,
        ratio_tol=ratio_tol,
        temperature=temperature,
        xpl_folder=xpl_folder,
    )
    return load_texture_pair(library_root, image_name, xpl_folder=xpl_folder, ppl_folder=ppl_folder)


def largest_rotated_rect(width: int, height: int, angle_rad: float) -> tuple[int, int]:
    """Return the largest axis-aligned rectangle inside a rotated rectangle."""
    if width <= 0 or height <= 0:
        return 0, 0
    angle_rad = angle_rad % (np.pi / 2)
    if angle_rad == 0:
        return width, height
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    numerator = width * height
    denominator_w = height * sin_a + width * cos_a
    denominator_h = width * sin_a + height * cos_a
    if denominator_w == 0 or denominator_h == 0:
        return 0, 0
    rect_w = min(numerator / denominator_w, width)
    rect_h = min(numerator / denominator_h, height)
    return int(rect_w), int(rect_h)


def rotate_crop_resize_pair(
    xpl_image: np.ndarray,
    ppl_image: np.ndarray,
    target_width: int,
    target_height: int,
    scale_factor: float = 2.0,
    sharpen_xpl: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random rotation, center crop, and resize to an XPL/PPL pair."""
    angle = random.uniform(0, 180)
    angle_rad = np.deg2rad(abs(angle))
    processed: list[np.ndarray] = []
    for idx, image in enumerate([xpl_image, ppl_image]):
        if image is None:
            raise ValueError("Input texture image is None.")
        resize_w = int(target_width * scale_factor)
        resize_h = int(target_height * scale_factor)
        img_resized = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
        h, w = img_resized.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(
            img_resized,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        crop_w, crop_h = largest_rotated_rect(w, h, angle_rad)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = img_rotated[y1 : y1 + crop_h, x1 : x1 + crop_w]
        final = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        if sharpen_xpl and idx == 0:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            final = cv2.filter2D(final, -1, kernel)
        processed.append(final)
    return processed[0], processed[1]
