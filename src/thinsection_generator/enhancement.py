from __future__ import annotations

import random
from typing import Sequence

import cv2
import numpy as np


def adjust_brightness_random(color: Sequence[int] | np.ndarray, min_factor: float = 0.7, max_factor: float = 1.3) -> np.ndarray:
    """Randomly adjust the brightness of an RGB color."""
    color_arr = np.asarray(color, dtype=np.float32)
    factor = random.uniform(min_factor, max_factor)
    return np.clip(color_arr * factor, 0, 255).astype(np.uint8)


def generate_ppl_pore_texture(
    image_shape: tuple[int, int],
    bg_color: tuple[int, int, int] = (199, 220, 199),
    num_dark_spots: int = 2000,
    num_light_spots: int = 4000,
    max_radius: int = 2,
    noise_var: float = 3.0,
    blur_kernel: tuple[int, int] = (3, 3),
    contrast: float = 1.0,
    brightness: int = 0,
) -> np.ndarray:
    """Generate a weakly mottled pore texture for PPL images."""
    height, width = image_shape
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    for _ in range(num_dark_spots):
        x, y = random.randrange(width), random.randrange(height)
        axes = (random.randint(1, max_radius), random.randint(1, max_radius))
        cv2.ellipse(img, (x, y), axes, random.randint(0, 360), 0, 360, (40, 40, 40), -1)
    for _ in range(num_light_spots):
        x, y = random.randrange(width), random.randrange(height)
        cv2.circle(img, (x, y), random.randint(1, max_radius), (160, 180, 150), -1)
    sigma = float(noise_var) ** 0.5
    noise = np.random.normal(0, sigma, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, blur_kernel, 0)
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def generate_xpl_pore_texture(
    image_shape: tuple[int, int],
    mean_rgb: tuple[int, int, int] = (25, 26, 20),
    std_rgb: tuple[int, int, int] = (5, 5, 5),
    num_dark_spots: int = 1000,
    num_light_spots: int = 250,
    max_radius: int = 2,
    blur_kernel: tuple[int, int] = (3, 3),
) -> np.ndarray:
    """Generate a near-black pore texture for XPL images."""
    h, w = image_shape
    mean = np.asarray(mean_rgb, dtype=np.float32)
    std = np.asarray(std_rgb, dtype=np.float32)
    img = np.random.normal(loc=mean, scale=std, size=(h, w, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(num_dark_spots):
        x, y = random.randrange(w), random.randrange(h)
        c = random.randint(0, 15)
        axes = (random.randint(1, max_radius), random.randint(1, max_radius))
        cv2.ellipse(img, (x, y), axes, random.randint(0, 360), 0, 360, (c, c, c), -1)
    for _ in range(num_light_spots):
        x, y = random.randrange(w), random.randrange(h)
        c = random.randint(25, 45)
        cv2.circle(img, (x, y), random.randint(1, max_radius), (c, c, c), -1)
    return cv2.GaussianBlur(img, blur_kernel, 0)


def fill_pores(
    xpl_image: np.ndarray,
    ppl_image: np.ndarray,
    pore_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill pore/background regions in paired XPL/PPL images."""
    pore_mask = pore_mask.astype(bool)
    h, w = ppl_image.shape[:2]
    ppl_texture = generate_ppl_pore_texture(
        (h, w),
        bg_color=(random.randint(198, 210), random.randint(216, 228), random.randint(201, 213)),
        num_dark_spots=random.randint(1600, 2400),
        num_light_spots=random.randint(3200, 4800),
        noise_var=random.uniform(2.0, 4.0),
        blur_kernel=random.choice([(3, 3), (5, 5)]),
        contrast=random.uniform(0.98, 1.03),
        brightness=random.randint(-3, 3),
    )
    xpl_texture = generate_xpl_pore_texture((h, w))
    ppl_image[pore_mask] = ppl_texture[pore_mask]
    xpl_image[pore_mask] = xpl_texture[pore_mask]
    return xpl_image, ppl_image


def apply_dissolution(
    xpl_image: np.ndarray,
    ppl_image: np.ndarray,
    dissolution_mask: np.ndarray,
    ppl_base_color: tuple[int, int, int] = (199, 220, 199),
) -> tuple[np.ndarray, np.ndarray]:
    """Render feldspar-dissolution pores in paired XPL/PPL images."""
    mask = dissolution_mask.astype(bool)
    xpl_image[mask] = (0, 0, 0)
    coords = np.where(mask)
    for y, x in zip(*coords):
        ppl_image[y, x] = adjust_brightness_random(np.asarray(ppl_base_color), 0.9, 1.1)
    return xpl_image, ppl_image


def smooth_edges_around_mask(
    xpl_image: np.ndarray,
    ppl_image: np.ndarray,
    mask: np.ndarray,
    xpl_gaussian: tuple[int, int] = (9, 9),
    ppl_gaussian: tuple[int, int] = (9, 9),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth image values around mask boundaries."""
    edge_source = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    edges = cv2.Canny(edge_source, threshold1=100, threshold2=200)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    blurred_xpl = cv2.GaussianBlur(xpl_image, xpl_gaussian, 0)
    xpl_image[edges_dilated == 255] = blurred_xpl[edges_dilated == 255]
    blurred_ppl = cv2.GaussianBlur(ppl_image, ppl_gaussian, 0)
    ppl_image[edges_dilated == 255] = blurred_ppl[edges_dilated == 255]
    return xpl_image, ppl_image, edges_dilated


def add_clay_rims(
    xpl_image: np.ndarray,
    ppl_image: np.ndarray,
    instance_map: np.ndarray,
    clay_color: tuple[int, int, int] = (169, 166, 171),
    rim_width: int = 5,
    fill_probability: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add sparse clay-mineral cementation rims near grain boundaries."""
    edge_source = cv2.normalize(instance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(edge_source, threshold1=1, threshold2=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_width, rim_width))
    rim_mask = cv2.dilate((edges > 0).astype(np.uint8), kernel, iterations=1).astype(bool)
    rim_mask &= np.random.random(rim_mask.shape) <= fill_probability

    smooth_band = cv2.dilate((edges > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    blurred_xpl = cv2.GaussianBlur(xpl_image, (5, 5), 0)
    xpl_image[smooth_band] = blurred_xpl[smooth_band]
    xpl_image[rim_mask] = (30, 30, 30)

    blurred_ppl = cv2.GaussianBlur(ppl_image, (5, 5), 0)
    ppl_image[smooth_band] = blurred_ppl[smooth_band]
    ppl_image[rim_mask] = adjust_brightness_random(np.asarray(clay_color), 0.8, 1.2)
    return xpl_image, ppl_image, rim_mask.astype(np.uint8) * 255
