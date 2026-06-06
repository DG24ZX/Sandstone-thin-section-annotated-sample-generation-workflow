from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

OLD_COLORS = {
    "quartz_old_blue": np.array([0, 0, 255]),
    "feldspar_old_yellow": np.array([255, 255, 0]),
    "lithic_old_red": np.array([255, 0, 0]),
    "uncertain_gray": np.array([128, 128, 128]),
    "background_black": np.array([0, 0, 0]),
}

NEW_COLORS = {
    "quartz_old_blue": np.array([0, 212, 255]),
    "feldspar_old_yellow": np.array([255, 229, 0]),
    "lithic_old_red": np.array([127, 0, 0]),
    "uncertain_gray": np.array([127, 0, 0]),
    "background_black": np.array([0, 0, 0]),
}


def convert_mask_color(input_path: str | Path, output_path: str | Path, tolerance: int = 30) -> None:
    """Convert old RGB annotation colors to the unified display palette."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    out = np.zeros_like(arr)
    matched = np.zeros(arr.shape[:2], dtype=bool)
    for key, old_color in OLD_COLORS.items():
        dist = np.linalg.norm(arr.astype(np.int16) - old_color.astype(np.int16), axis=2)
        mask = dist <= tolerance
        out[mask] = NEW_COLORS[key]
        matched |= mask
    out[~matched] = np.array([0, 0, 0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(output_path)


def batch_convert_masks(src_dir: str | Path, dst_dir: str | Path, tolerance: int = 30) -> None:
    """Batch-convert mask colors in a directory."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for path in tqdm(sorted(src_dir.iterdir())):
        if path.suffix.lower() not in valid_ext:
            continue
        convert_mask_color(path, dst_dir / f"{path.stem}.png", tolerance=tolerance)
