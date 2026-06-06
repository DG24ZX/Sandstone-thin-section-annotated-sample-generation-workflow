from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_raw_volume(raw_path: str | Path, shape: Tuple[int, int, int], dtype=np.uint8) -> np.ndarray:
    """Read a raw 3D digital-core volume and reshape it.

    Parameters
    ----------
    raw_path : str or Path
        Path to the raw file.
    shape : tuple of int
        Volume shape in (z, y, x) order.
    dtype : numpy dtype
        Data type used in the raw file.
    """
    raw_path = Path(raw_path)
    arr = np.fromfile(str(raw_path), dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"Raw file size does not match shape. Got {arr.size} values, "
            f"expected {expected} for shape={shape}."
        )
    return arr.reshape(shape)


def read_raw_slice(
    raw_path: str | Path,
    shape: Tuple[int, int, int],
    slice_index: int = 0,
    resize_to: Tuple[int, int] | None = None,
    dtype=np.uint8,
) -> np.ndarray:
    """Read one 2D slice from a raw digital-core volume.

    `resize_to` is given as (width, height), following OpenCV convention.
    Nearest-neighbor interpolation is used to preserve class labels.
    """
    volume = read_raw_volume(raw_path, shape=shape, dtype=dtype)
    if not (0 <= slice_index < volume.shape[0]):
        raise IndexError(f"slice_index={slice_index} is outside volume depth {volume.shape[0]}.")
    slice_img = volume[slice_index].copy()
    if resize_to is not None:
        slice_img = cv2.resize(slice_img, resize_to, interpolation=cv2.INTER_NEAREST)
    return slice_img


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    """Save an RGB image using OpenCV."""
    path = Path(path)
    ensure_dir(path.parent)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def save_label(path: str | Path, label: np.ndarray) -> None:
    """Save a single-channel label image for semantic segmentation."""
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), label.astype(np.uint8))


def save_generated_sample(
    output_dir: str | Path,
    xpl_image: np.ndarray,
    ppl_image: np.ndarray,
    label_mask: np.ndarray,
    sample_name: str = "sample_0001",
    xpl_dir: str = "xpl",
    ppl_dir: str = "ppl",
    label_dir: str = "labels",
    write_manifest: bool = True,
) -> None:
    """Save one generated sample in a semantic-segmentation-friendly layout.

    Output layout:
        output_dir/
        ├── xpl/sample.png
        ├── ppl/sample.png
        ├── labels/sample.png
        └── manifest.csv
    """
    output_dir = Path(output_dir)
    filename = f"{sample_name}.png"
    xpl_path = Path(xpl_dir) / filename
    ppl_path = Path(ppl_dir) / filename
    label_path = Path(label_dir) / filename

    save_rgb(output_dir / xpl_path, xpl_image)
    save_rgb(output_dir / ppl_path, ppl_image)
    save_label(output_dir / label_path, label_mask)

    if write_manifest:
        manifest_path = output_dir / "manifest.csv"
        ensure_dir(manifest_path.parent)
        is_new = not manifest_path.exists()
        with manifest_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["sample_id", "xpl_path", "ppl_path", "label_path"])
            writer.writerow([sample_name, str(xpl_path), str(ppl_path), str(label_path)])
