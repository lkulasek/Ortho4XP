#!/usr/bin/env python3
"""
Orthophoto Color Correction Script (Two-Pass, LAB + Graph-Based)

Corrects color mismatches in Ortho4XP orthophoto JPEGs using a two-level
approach in CIELAB color space for perceptually uniform corrections:

Pass 1 - Intra-image correction (LAB):
    Within each 4096x4096 JPG (composed of 16x16 sub-tiles of 256px each),
    equalize sub-tiles relative to each other using mean+std normalization
    in LAB space. This corrects lightness and color cast differences between
    sub-tiles while preserving natural tonal character.
    Correction is bilinearly interpolated for smooth transitions.

Pass 2 - Inter-image correction (Graph-based seam minimization):
    Builds an adjacency graph of neighboring tiles. For each pair of adjacent
    tiles, samples border pixels in two halves and computes LAB color
    differences at the shared edge. Solves a least-squares system to find
    per-tile affine LAB correction fields (a + b*u + c*v) that minimize
    boundary discontinuities across the entire tile mosaic. The affine
    model allows corrections to vary linearly across each tile, handling
    spatially varying color shifts from atmospheric/sensor gradients.

Edit INPUT_DIR and OUTPUT_DIR below, then run:
    python3 correct_colors.py

The script automatically discovers all subdirectories containing JPG files
under INPUT_DIR and recreates the same directory structure under OUTPUT_DIR.
"""

import os
import sys
import shutil
import time
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# ============================================================
# CONFIGURATION - edit these before running
# ============================================================
INPUT_DIR = "Orthophotos/+50+010/+54+018"
OUTPUT_DIR = "Orthophotos/+50+010/+54+018_corrected"
# Mask directory — land/water masks (white=land, black=water).
# Set to "" to disable mask usage.
MASK_DIR = "Masks/+50+010/+54+018"
# Minimum mask value to consider a pixel as land (0-255).
# Pixels below this in the mask are treated as water and excluded.
LAND_THRESHOLD = 128
# ============================================================

# Correction strength 0.0-1.0 for each pass
INTRA_STRENGTH = 0.9   # sub-tile equalization within each image
INTER_STRENGTH = 0.9   # graph-based seam correction across images

# JPEG output quality 1-100
JPEG_QUALITY = 90
# Number of parallel workers
WORKERS = 4
# Create backup of originals before overwriting (only used if OUTPUT_DIR == INPUT_DIR)
BACKUP = True

# Sub-tile grid dimensions for pass 1 intra-image equalization.
# Multiple scales are applied sequentially (coarse to fine).
# 8 = 512px sub-tiles (large-scale gradient correction)
# 16 = 256px sub-tiles (matches original web tile boundaries)
# 32 = 128px sub-tiles (finer detection of internal color gradients)
# Each value must divide 4096 evenly.
PASS1_GRIDS = [8, 32]
# Ignore sub-tiles that are nearly all black or white (likely empty/cloud)
# These are in LAB L-channel units [0-100]
BLACK_THRESHOLD_L = 6.0
WHITE_THRESHOLD_L = 94.0

# Outlier detection: sub-tiles whose mean deviates more than this many
# standard deviations get reduced correction strength.
OUTLIER_SIGMA = 2.0

# Minimum std to use as divisor (avoid division by near-zero)
MIN_STD = 5.0
# Maximum contrast scale factor
MAX_SCALE = 2.0
MIN_SCALE = 0.5

# Border sampling: how many pixel rows/cols to sample along shared edges.
# Narrower = better represents the actual seam discontinuity.
BORDER_SAMPLE_WIDTH = 16

# Number of segments to split each border into for sampling.
# More segments = better-constrained affine model (3 unknowns per tile).
BORDER_SAMPLE_SEGMENTS = 6

# Regularization weight for the least-squares solver.
# Higher = corrections are smaller (more conservative).
# Lower = corrections more aggressively minimize seams.
REGULARIZATION = 0.03

# Number of pass 2 iterations. Each iteration re-samples borders from
# the previous result and re-solves. More iterations = better convergence.
PASS2_ITERATIONS = 2

# Post-processing: overall saturation and contrast boost.
# 1.0 = no change, >1.0 = increase, <1.0 = decrease.
SATURATION_BOOST = 1.1
CONTRAST_BOOST = 1.1

# Blue cast removal for neutral/earthy tones (dirt, ground, concrete).
# Atmospheric haze shifts these toward blue (negative LAB b channel).
# This corrects pixels with low chroma that have a blue bias.
# 0.0 = disabled, 1.0 = full correction to neutral. Recommended: 0.5-0.8
BLUE_CAST_REMOVAL = 0.6
# Maximum chroma (sqrt(a^2 + b^2)) to consider a pixel as "neutral/earthy".
# Higher = more aggressive (affects more saturated pixels too).
BLUE_CAST_CHROMA_LIMIT = 25.0
# Only correct pixels with negative b (blue). Positive b (yellow) is left alone.
# Lightness range to target (avoid touching very dark shadows or bright highlights)
BLUE_CAST_L_MIN = 15.0
BLUE_CAST_L_MAX = 85.0


# ============================================================
# LAB conversion utilities
# ============================================================

def rgb_to_lab(rgb_uint8):
    """Convert RGB uint8 array to CIELAB float32 array.
    L: [0, 100], a: [-128, 127], b: [-128, 127]
    Uses D65 illuminant."""
    rgb = rgb_uint8.astype(np.float32) / 255.0

    # Linearize sRGB
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # RGB to XYZ (sRGB D65)
    # Using the standard sRGB to XYZ matrix
    r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 white point
    x /= 0.95047
    # y /= 1.00000
    z /= 1.08883

    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3

    def f(t):
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)

    return np.stack([L, a, b_ch], axis=-1)


def lab_to_rgb(lab):
    """Convert CIELAB float32 array to RGB uint8 array."""
    L, a, b_ch = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_ch / 200.0

    epsilon = 0.008856
    kappa = 903.3

    x = np.where(fx ** 3 > epsilon, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(L > kappa * epsilon, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(fz ** 3 > epsilon, fz ** 3, (116.0 * fz - 16.0) / kappa)

    # De-normalize by D65 white point
    x *= 0.95047
    # y *= 1.00000
    z *= 1.08883

    # XYZ to linear RGB
    r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Clip before gamma to avoid NaN in power
    r_lin = np.clip(r_lin, 0, None)
    g_lin = np.clip(g_lin, 0, None)
    b_lin = np.clip(b_lin, 0, None)

    # Apply sRGB gamma
    def gamma(c):
        return np.where(c > 0.0031308, 1.055 * np.power(c, 1.0 / 2.4) - 0.055, 12.92 * c)

    rgb = np.stack([gamma(r_lin), gamma(g_lin), gamma(b_lin)], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


# ============================================================
# Core utilities
# ============================================================

def compute_outlier_strength(tile_mean, ref_mean, ref_std, base_strength):
    """Reduce correction strength for outlier regions whose mean
    deviates significantly from the reference.
    Works in any color space (RGB or LAB channels)."""
    safe_std = np.maximum(ref_std, 1.0)
    z_scores = np.abs(tile_mean - ref_mean) / safe_std
    max_z = np.max(z_scores)
    if max_z <= OUTLIER_SIGMA:
        return base_strength
    else:
        reduction = np.clip((max_z - OUTLIER_SIGMA) / OUTLIER_SIGMA, 0.0, 1.0)
        return base_strength * (1.0 - reduction)


def parse_tile_coords(filename):
    """Extract (row, col) grid coordinates from tile filename.
    Expected format: {row}_{col}_{provider}.jpg"""
    stem = Path(filename).stem  # e.g. "25040_13648_GO216"
    parts = stem.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None


def parse_zl_from_dir(dirname):
    """Extract zoom level from directory name like 'GO2_16' or 'BI_18'.
    Returns integer zoom level, or None."""
    name = Path(dirname).name
    parts = name.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


def detect_mask_zl(mask_dir, til_x, til_y, texture_zl):
    """Auto-detect mask_zl by trying factor = 2^k for k=1,2,3,...
    until a mask file is found. Returns (mask_zl, factor) or (None, None)."""
    for mask_zl in range(texture_zl - 1, max(texture_zl - 5, 12), -1):
        factor = 2 ** (texture_zl - mask_zl)
        m_til_x = (int(til_x / factor) // 16) * 16
        m_til_y = (int(til_y / factor) // 16) * 16
        mask_file = os.path.join(mask_dir, f"{m_til_x}_{m_til_y}.png")
        if os.path.isfile(mask_file):
            return mask_zl, factor
    return None, None


def get_land_mask(mask_dir, til_x, til_y, texture_zl, mask_zl):
    """Load the land mask for a texture tile, cropped to the tile's sub-region.

    Returns a (4096, 4096) uint8 array where 255=land, 0=water,
    or None if no mask is available.
    """
    if not mask_dir or mask_zl is None:
        return None
    factor = 2 ** (texture_zl - mask_zl)
    m_til_x = (int(til_x / factor) // 16) * 16
    m_til_y = (int(til_y / factor) // 16) * 16
    rx = int((til_x - factor * m_til_x) / 16)
    ry = int((til_y - factor * m_til_y) / 16)
    mask_file = os.path.join(mask_dir, f"{m_til_x}_{m_til_y}.png")
    if not os.path.isfile(mask_file):
        return None
    try:
        mask_img = Image.open(mask_file)
        # Crop the sub-region for this texture tile
        crop_size = 4096 // factor
        x0 = rx * crop_size
        y0 = ry * crop_size
        sub = mask_img.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        # Resize to 4096x4096 to match the texture tile
        if crop_size != 4096:
            sub = sub.resize((4096, 4096), Image.NEAREST)
        return np.array(sub, dtype=np.uint8)
    except Exception:
        return None


# ============================================================
# Pass 1: Intra-image sub-tile equalization (LAB)
# ============================================================

def _equalize_at_scale(lab, grid, strength, land_mask):
    """Apply single-scale sub-tile equalization in-place on a LAB image.

    Returns the corrected LAB array (may be a new array or modified in-place).
    """
    h, w = lab.shape[:2]
    subtile_size = h // grid

    # Collect per-sub-tile mean and std in LAB
    tile_means = np.zeros((grid, grid, 3), dtype=np.float32)
    tile_stds = np.zeros((grid, grid, 3), dtype=np.float32)
    valid = np.zeros((grid, grid), dtype=bool)

    for row in range(grid):
        for col in range(grid):
            y0 = row * subtile_size
            x0 = col * subtile_size
            tile = lab[y0:y0 + subtile_size, x0:x0 + subtile_size]

            if land_mask is not None:
                mask_tile = land_mask[y0:y0 + subtile_size,
                                     x0:x0 + subtile_size]
                land_pixels = mask_tile >= LAND_THRESHOLD
                land_count = land_pixels.sum()
                if land_count < 100:
                    continue
                tile_land = tile[land_pixels]
                mean = tile_land.mean(axis=0)
                std = tile_land.std(axis=0)
            else:
                mean = tile.mean(axis=(0, 1))
                std = tile.std(axis=(0, 1))

            if mean[0] > BLACK_THRESHOLD_L and mean[0] < WHITE_THRESHOLD_L:
                tile_means[row, col] = mean
                tile_stds[row, col] = std
                valid[row, col] = True

    valid_count = valid.sum()
    if valid_count < 2:
        return lab

    valid_means = tile_means[valid]
    valid_stds = tile_stds[valid]
    ref_mean = np.median(valid_means, axis=0)
    ref_std = np.median(valid_stds, axis=0)
    ref_std = np.maximum(ref_std, 1.0)

    local_mean = np.mean(valid_means, axis=0)
    local_std_of_means = np.std(valid_means, axis=0)
    local_std_of_means = np.maximum(local_std_of_means, 1.0)

    offsets = np.zeros((grid, grid, 3), dtype=np.float32)
    scales = np.ones((grid, grid, 3), dtype=np.float32)

    for row in range(grid):
        for col in range(grid):
            if not valid[row, col]:
                continue
            eff_str = compute_outlier_strength(
                tile_means[row, col], local_mean, local_std_of_means, strength
            )
            if eff_str < 0.01:
                continue
            for ch in range(3):
                t_mean = tile_means[row, col, ch]
                t_std = max(tile_stds[row, col, ch], MIN_STD)
                r_mean = ref_mean[ch]
                r_std = ref_std[ch]
                scale = np.clip(r_std / t_std, MIN_SCALE, MAX_SCALE)
                blended_scale = 1.0 + (scale - 1.0) * eff_str
                scales[row, col, ch] = blended_scale
                offsets[row, col, ch] = (t_mean + (r_mean - t_mean) * eff_str) \
                                        - t_mean * blended_scale

    # Bilinear interpolation (fully vectorized)
    result_lab = np.empty_like(lab)

    pixel_y = np.arange(h, dtype=np.float32)
    gy = pixel_y / subtile_size - 0.5
    gy = np.clip(gy, 0.0, grid - 1.0)
    gy0 = np.floor(gy).astype(np.int32)
    gy0 = np.minimum(gy0, grid - 2)
    gy1 = gy0 + 1
    wy = gy - gy0.astype(np.float32)

    pixel_x = np.arange(w, dtype=np.float32)
    gx = pixel_x / subtile_size - 0.5
    gx = np.clip(gx, 0.0, grid - 1.0)
    gx0 = np.floor(gx).astype(np.int32)
    gx0 = np.minimum(gx0, grid - 2)
    gx1 = gx0 + 1
    wx = gx - gx0.astype(np.float32)

    gy0_2d = gy0[:, np.newaxis]
    gy1_2d = gy1[:, np.newaxis]
    wy_2d = wy[:, np.newaxis]
    gx0_2d = gx0[np.newaxis, :]
    gx1_2d = gx1[np.newaxis, :]
    wx_2d = wx[np.newaxis, :]

    for ch in range(3):
        s_tl = scales[gy0_2d, gx0_2d, ch]
        s_tr = scales[gy0_2d, gx1_2d, ch]
        s_bl = scales[gy1_2d, gx0_2d, ch]
        s_br = scales[gy1_2d, gx1_2d, ch]
        scale_interp = (s_tl + (s_tr - s_tl) * wx_2d) * (1 - wy_2d) + \
                        (s_bl + (s_br - s_bl) * wx_2d) * wy_2d

        o_tl = offsets[gy0_2d, gx0_2d, ch]
        o_tr = offsets[gy0_2d, gx1_2d, ch]
        o_bl = offsets[gy1_2d, gx0_2d, ch]
        o_br = offsets[gy1_2d, gx1_2d, ch]
        offset_interp = (o_tl + (o_tr - o_tl) * wx_2d) * (1 - wy_2d) + \
                         (o_bl + (o_br - o_bl) * wx_2d) * wy_2d

        result_lab[:, :, ch] = lab[:, :, ch] * scale_interp + offset_interp

    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)
    return result_lab


def pass1_correct_image(image_path, output_path, strength,
                        mask_dir="", mask_zl=None):
    """Equalize sub-tiles within a single image in LAB color space.

    Applies multi-scale equalization (coarse to fine) as configured by
    PASS1_GRIDS. Each scale normalizes sub-tiles toward the image-wide
    median using mean+std normalization with bilinear interpolation.
    Saves as lossless PNG to avoid compression artifacts before pass 2.
    """
    try:
        img = Image.open(image_path)
        arr = np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"  Warning: could not read {image_path}: {e}")
        return None

    h, w = arr.shape[:2]
    if arr.ndim != 3 or arr.shape[2] != 3 or h != 4096 or w != 4096:
        Image.fromarray(arr).save(output_path, "PNG")
        return None

    # Convert entire image to LAB
    lab = rgb_to_lab(arr)

    # Load land mask if available
    land_mask = None
    if mask_dir and mask_zl is not None:
        coords = parse_tile_coords(image_path)
        if coords is not None:
            til_x, til_y = coords
            texture_zl = parse_zl_from_dir(Path(image_path).parent)
            if texture_zl is not None:
                land_mask = get_land_mask(mask_dir, til_x, til_y,
                                         texture_zl, mask_zl)

    # Apply multi-scale equalization (coarse to fine)
    for grid in PASS1_GRIDS:
        lab = _equalize_at_scale(lab, grid, strength, land_mask)

    # Convert back to RGB
    result = lab_to_rgb(lab)

    out_img = Image.fromarray(result)
    out_img.save(output_path, "PNG")

    # Return whole-image LAB mean and std for pass 2 (land pixels only)
    if land_mask is not None:
        land_pixels = land_mask >= LAND_THRESHOLD
        if land_pixels.sum() > 0:
            land_lab = lab[land_pixels]
            image_mean = land_lab.mean(axis=0).astype(np.float32)
            image_std = land_lab.std(axis=0).astype(np.float32)
        else:
            image_mean = lab.mean(axis=(0, 1)).astype(np.float32)
            image_std = lab.std(axis=(0, 1)).astype(np.float32)
    else:
        image_mean = lab.mean(axis=(0, 1)).astype(np.float32)
        image_std = lab.std(axis=(0, 1)).astype(np.float32)
    return image_mean, image_std


def pass1_worker(args):
    """Worker for pass 1 parallel processing."""
    image_path, output_path, strength, mask_dir, mask_zl = args
    try:
        stats = pass1_correct_image(
            str(image_path), output_path, strength,
            mask_dir, mask_zl
        )
        return str(image_path), stats, None
    except Exception as e:
        return str(image_path), None, str(e)


# ============================================================
# Pass 2: Graph-based inter-image seam minimization (LAB)
# ============================================================

def build_adjacency_graph(file_paths):
    """Build adjacency graph from tile filenames.

    Two tiles are neighbors if they share an edge in the tile grid
    (row or col differs by exactly the grid step size within the same
    zoom level directory).

    Returns list of (idx_a, idx_b, edge_type) where edge_type is
    'horizontal' (same row, adjacent cols) or 'vertical' (same col, adjacent rows).
    """
    # Group files by parent directory (zoom level)
    dir_groups = defaultdict(list)
    for idx, fpath in enumerate(file_paths):
        parent = str(Path(fpath).parent)
        dir_groups[parent].append(idx)

    edges = []

    for parent, indices in dir_groups.items():
        # Parse coordinates for tiles in this directory
        coord_to_idx = {}
        for idx in indices:
            fname = Path(file_paths[idx]).name
            coords = parse_tile_coords(fname)
            if coords is not None:
                coord_to_idx[coords] = idx

        if not coord_to_idx:
            continue

        # Determine grid step: smallest difference between adjacent rows/cols
        all_rows = sorted(set(r for r, _ in coord_to_idx.keys()))
        all_cols = sorted(set(c for _, c in coord_to_idx.keys()))

        row_step = None
        for i in range(1, len(all_rows)):
            diff = all_rows[i] - all_rows[i - 1]
            if row_step is None or diff < row_step:
                row_step = diff

        col_step = None
        for i in range(1, len(all_cols)):
            diff = all_cols[i] - all_cols[i - 1]
            if col_step is None or diff < col_step:
                col_step = diff

        if row_step is None:
            row_step = 16
        if col_step is None:
            col_step = 16

        # Find edges
        for (r, c), idx in coord_to_idx.items():
            # Right neighbor
            right = (r, c + col_step)
            if right in coord_to_idx:
                edges.append((idx, coord_to_idx[right], 'horizontal'))
            # Bottom neighbor
            bottom = (r + row_step, c)
            if bottom in coord_to_idx:
                edges.append((idx, coord_to_idx[bottom], 'vertical'))

    return edges


def sample_border_stats(image_path, edge_type, side, land_mask=None):
    """Sample LAB statistics from the border of a tile, split into segments.

    For a horizontal edge (left/right neighbor), the border strip runs
    vertically, so we split it into BORDER_SAMPLE_SEGMENTS segments along rows.
    For a vertical edge (top/bottom neighbor), the border strip runs
    horizontally, so we split it into BORDER_SAMPLE_SEGMENTS segments along cols.

    If land_mask is provided (4096x4096 uint8), only land pixels
    (mask >= LAND_THRESHOLD) are included in the statistics.

    Returns list of BORDER_SAMPLE_SEGMENTS (3,) float32 arrays (segment means),
    or None. Segments ordered from top-to-bottom or left-to-right.
    """
    try:
        img = Image.open(image_path)
    except Exception:
        return None

    w, h = img.size
    if img.mode != 'RGB':
        return None

    bw = BORDER_SAMPLE_WIDTH

    if edge_type == 'horizontal':
        if side == 'right':
            box = (w - bw, 0, w, h)
        else:
            box = (0, 0, bw, h)
    else:
        if side == 'bottom':
            box = (0, h - bw, w, h)
        else:
            box = (0, 0, w, bw)

    strip = np.array(img.crop(box), dtype=np.uint8)
    lab_strip = rgb_to_lab(strip)

    mask_strip = None
    if land_mask is not None:
        mask_strip = land_mask[box[1]:box[3], box[0]:box[2]]

    # Split into BORDER_SAMPLE_SEGMENTS segments along the long axis
    n_seg = BORDER_SAMPLE_SEGMENTS
    if edge_type == 'horizontal':
        # Strip is (h, bw, 3) — split along rows
        seg_len = lab_strip.shape[0] // n_seg
        halves_lab = [lab_strip[i*seg_len:(i+1)*seg_len] for i in range(n_seg)]
        if mask_strip is not None:
            halves_mask = [mask_strip[i*seg_len:(i+1)*seg_len] >= LAND_THRESHOLD
                           for i in range(n_seg)]
        else:
            halves_mask = [None] * n_seg
    else:
        # Strip is (bw, w, 3) — split along cols
        seg_len = lab_strip.shape[1] // n_seg
        halves_lab = [lab_strip[:, i*seg_len:(i+1)*seg_len] for i in range(n_seg)]
        if mask_strip is not None:
            halves_mask = [mask_strip[:, i*seg_len:(i+1)*seg_len] >= LAND_THRESHOLD
                           for i in range(n_seg)]
        else:
            halves_mask = [None] * n_seg

    def masked_mean(lab_block, mask_block):
        """Return (mean, confidence) or (None, 0)."""
        if mask_block is not None:
            n_land = mask_block.sum()
            if n_land < 10:
                return None, 0.0
            pixels = lab_block[mask_block]
            mean = pixels.mean(axis=0).astype(np.float32)
            # Confidence: fraction of land pixels × inverse variance factor.
            # Low variance = more uniform segment = higher confidence.
            total_pixels = mask_block.size
            land_frac = n_land / max(total_pixels, 1)
            variance = pixels.var(axis=0).mean()
            inv_var_factor = 1.0 / (1.0 + variance)
            confidence = land_frac * inv_var_factor
            return mean, float(confidence)
        else:
            pixels = lab_block.reshape(-1, 3)
            mean = pixels.mean(axis=0).astype(np.float32)
            variance = pixels.var(axis=0).mean()
            inv_var_factor = 1.0 / (1.0 + variance)
            return mean, float(inv_var_factor)

    results = [masked_mean(q, m) for q, m in zip(halves_lab, halves_mask)]
    means = [r[0] for r in results]
    weights = [r[1] for r in results]
    if any(m is None for m in means):
        return None

    return means, weights


def solve_graph_corrections(n_tiles, border_equations):
    """Solve least-squares system for per-tile bilinear LAB corrections.

    Each tile i has 4 parameters per LAB channel:
        correction_i(u, v) = a_i + b_i * u + c_i * v + d_i * u * v
    where u ∈ [0,1] is horizontal, v ∈ [0,1] is vertical position.

    border_equations: list of (coeffs_lhs, rhs, weight) tuples.
        coeffs_lhs: list of (tile_idx, a_coeff, b_coeff, c_coeff, d_coeff)
        rhs: (3,) float32 array — the measured LAB gap
        weight: float — confidence weight for this equation

    Returns corrections as (n_tiles, 3, 4) float32 array where
    corrections[i, ch] = [a, b, c, d].
    """
    n_params = n_tiles * 4  # 4 params per tile: a, b, c, d

    if not border_equations:
        return np.zeros((n_tiles, 3, 4), dtype=np.float32)

    corrections = np.zeros((n_tiles, 3, 4), dtype=np.float32)

    for ch in range(3):
        AtA = np.zeros((n_params, n_params), dtype=np.float64)
        Atb = np.zeros(n_params, dtype=np.float64)

        for coeffs_lhs, rhs, weight in border_equations:
            d = float(rhs[ch])
            w = float(weight)
            indices = []
            values = []
            for tile_idx, a_coeff, b_coeff, c_coeff, d_coeff in coeffs_lhs:
                base = tile_idx * 4
                for offset, val in enumerate([a_coeff, b_coeff, c_coeff, d_coeff]):
                    if val != 0.0:
                        indices.append(base + offset)
                        values.append(val)

            for p in range(len(indices)):
                ip = indices[p]
                vp = values[p]
                Atb[ip] += w * vp * d
                for q in range(len(indices)):
                    AtA[ip, indices[q]] += w * vp * values[q]

        np.fill_diagonal(AtA, AtA.diagonal() + REGULARIZATION)

        try:
            params = np.linalg.solve(AtA, Atb)
            for i in range(n_tiles):
                base = i * 4
                corrections[i, ch, 0] = params[base + 0]  # a
                corrections[i, ch, 1] = params[base + 1]  # b
                corrections[i, ch, 2] = params[base + 2]  # c
                corrections[i, ch, 3] = params[base + 3]  # d
        except np.linalg.LinAlgError as e:
            print(f"  Warning: least-squares solve failed for channel "
                  f"{['L', 'a', 'b'][ch]}: {e}. Corrections for this "
                  f"channel will be zero.")

    return corrections


def pass2_apply_correction(image_path, output_path, correction, strength, quality):
    """Apply a bilinear LAB correction field to an image.

    correction: (3, 4) float32 array where correction[ch] = [a, b, c, d]
    defines correction_ch(u,v) = a + b*u + c*v + d*u*v for each LAB channel.
    u ∈ [0,1] horizontal, v ∈ [0,1] vertical.

    Output format is determined by output_path extension (.png or .jpg).
    """
    try:
        img = Image.open(image_path)
        arr = np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"  Warning: could not read {image_path}: {e}")
        return False

    is_jpeg = output_path.lower().endswith(('.jpg', '.jpeg'))

    if arr.ndim != 3 or arr.shape[2] != 3:
        if is_jpeg:
            Image.fromarray(arr).save(output_path, "JPEG", quality=quality)
        else:
            Image.fromarray(arr).save(output_path, "PNG")
        return False

    # Check if correction is negligible and no post-processing needed
    no_postprocess = (SATURATION_BOOST == 1.0 and CONTRAST_BOOST == 1.0
                      and BLUE_CAST_REMOVAL == 0.0)
    if np.max(np.abs(correction)) < 0.1 and (no_postprocess or not is_jpeg):
        if is_jpeg:
            Image.fromarray(arr).save(output_path, "JPEG", quality=quality)
        else:
            Image.fromarray(arr).save(output_path, "PNG")
        return False

    h, w = arr.shape[:2]
    lab = rgb_to_lab(arr)

    # Build correction field
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    for ch in range(3):
        a, b, c, d = correction[ch]
        field = (a + b * uu + c * vv + d * uu * vv) * strength
        lab[:, :, ch] += field

    # Post-processing: only on final JPEG output
    if is_jpeg:
        # Blue cast removal first — before saturation boost so chroma
        # thresholds reflect the natural (unboosted) color values.
        if BLUE_CAST_REMOVAL > 0.0:
            L_ch = lab[:, :, 0]
            a_ch = lab[:, :, 1]
            b_ch = lab[:, :, 2]
            chroma = np.sqrt(a_ch ** 2 + b_ch ** 2)
            # Target: low-chroma pixels with negative b (blue) in mid-lightness
            mask = ((b_ch < 0)
                    & (chroma < BLUE_CAST_CHROMA_LIMIT)
                    & (L_ch > BLUE_CAST_L_MIN)
                    & (L_ch < BLUE_CAST_L_MAX))
            # Correction strength ramps down as chroma approaches the limit
            # so the effect fades smoothly rather than creating hard edges
            blend = np.where(mask,
                             BLUE_CAST_REMOVAL * (1.0 - chroma / BLUE_CAST_CHROMA_LIMIT),
                             0.0).astype(np.float32)
            # Nudge b channel toward 0 (neutral)
            lab[:, :, 2] = b_ch * (1.0 - blend)

        if CONTRAST_BOOST != 1.0:
            mean_L = lab[:, :, 0].mean()
            lab[:, :, 0] = (lab[:, :, 0] - mean_L) * CONTRAST_BOOST + mean_L
        if SATURATION_BOOST != 1.0:
            lab[:, :, 1] *= SATURATION_BOOST
            lab[:, :, 2] *= SATURATION_BOOST

    # Clamp LAB channels to valid ranges
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)
    lab[:, :, 1] = np.clip(lab[:, :, 1], -128, 127)
    lab[:, :, 2] = np.clip(lab[:, :, 2], -128, 127)

    result = lab_to_rgb(lab)
    out_img = Image.fromarray(result)
    if is_jpeg:
        out_img.save(output_path, "JPEG", quality=quality)
    else:
        out_img.save(output_path, "PNG")
    return True


def pass2_worker(args):
    """Worker for pass 2 parallel processing."""
    image_path, output_path, correction, strength, quality = args
    try:
        corrected = pass2_apply_correction(
            str(image_path), output_path, correction, strength, quality
        )
        return str(Path(image_path).name), corrected, None
    except Exception as e:
        return str(Path(image_path).name), False, str(e)


# ============================================================
# Main orchestration
# ============================================================

def run_parallel(tasks, worker_fn, label, total):
    """Run tasks in parallel with progress reporting."""
    done = 0
    errors = 0
    results = []

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(worker_fn, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            name, _, err = result
            if err:
                print(f"  Error: {name}: {err}")
                errors += 1
            if done % 20 == 0 or done == total:
                print(f"  {label}: {done}/{total}")

    return results, errors


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: {INPUT_DIR} is not a directory")
        sys.exit(1)

    # Discover all subdirectories containing JPG files
    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)

    all_files = []  # (input_path, intermediate_png_path, final_jpg_path)
    subdirs_found = 0

    for subdir in sorted(input_root.rglob("*")):
        if not subdir.is_dir():
            continue
        jpg_files = sorted(subdir.glob("*.jpg"))
        if not jpg_files:
            continue

        rel = subdir.relative_to(input_root)
        out_subdir = output_root / rel
        os.makedirs(out_subdir, exist_ok=True)

        subdirs_found += 1
        print(f"Found {len(jpg_files)} images in {subdir}")

        if BACKUP and str(output_root) == str(input_root):
            backup_dir = str(subdir) + "_backup"
            if not os.path.exists(backup_dir):
                print(f"  Creating backup in {backup_dir}...")
                shutil.copytree(str(subdir), backup_dir)
                print("  Backup complete.")
            else:
                print(f"  Backup already exists at {backup_dir}, skipping.")

        for fpath in jpg_files:
            final_output = str(out_subdir / fpath.name)
            # Intermediate pass 1 output uses PNG for lossless quality
            intermediate = str(out_subdir / (fpath.stem + "_p1.png"))
            all_files.append((fpath, intermediate, final_output))

    # Also check root directory itself for JPGs
    root_jpgs = sorted(input_root.glob("*.jpg"))
    if root_jpgs:
        os.makedirs(output_root, exist_ok=True)
        print(f"Found {len(root_jpgs)} images in {input_root}")
        if BACKUP and str(output_root) == str(input_root):
            backup_dir = str(input_root) + "_backup"
            if not os.path.exists(backup_dir):
                print(f"  Creating backup in {backup_dir}...")
                shutil.copytree(str(input_root), backup_dir)
                print("  Backup complete.")
        for fpath in root_jpgs:
            final_output = str(output_root / fpath.name)
            intermediate = str(output_root / (fpath.stem + "_p1.png"))
            all_files.append((fpath, intermediate, final_output))

    if not all_files:
        print(f"No .jpg files found under {INPUT_DIR}")
        sys.exit(1)

    total = len(all_files)
    print(f"\nTotal: {total} images across {subdirs_found} subdirectories")

    # ------------------------------------------------------------------
    # Detect mask_zl if masks are available
    # ------------------------------------------------------------------
    mask_zl = None
    if MASK_DIR and os.path.isdir(MASK_DIR):
        # Try to detect mask_zl from the first tile in each subdirectory
        for fpath, _, _ in all_files:
            coords = parse_tile_coords(str(fpath))
            texture_zl = parse_zl_from_dir(fpath.parent)
            if coords is not None and texture_zl is not None:
                til_x, til_y = coords
                detected_zl, factor = detect_mask_zl(
                    MASK_DIR, til_x, til_y, texture_zl
                )
                if detected_zl is not None:
                    mask_zl = detected_zl
                    print(f"Detected mask_zl={mask_zl} "
                          f"(factor={factor} for ZL{texture_zl})")
                    break
        if mask_zl is None:
            print("Warning: MASK_DIR set but could not detect mask_zl. "
                  "Masks will not be used.")
    else:
        if MASK_DIR:
            print(f"Warning: MASK_DIR '{MASK_DIR}' not found. "
                  "Masks will not be used.")

    # ------------------------------------------------------------------
    # Pass 1: Intra-image sub-tile equalization (LAB)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PASS 1: Intra-image sub-tile equalization in LAB (strength={INTRA_STRENGTH})")
    print(f"{'='*60}")

    pass1_tasks = []
    for fpath, intermediate, final_output in all_files:
        pass1_tasks.append((fpath, intermediate, INTRA_STRENGTH,
                            MASK_DIR, mask_zl))

    t0 = time.time()
    pass1_results, pass1_errors = run_parallel(
        pass1_tasks, pass1_worker, "Pass 1", total
    )
    t1 = time.time()

    # Collect per-image stats (LAB mean/std) for diagnostics
    image_stats = {}
    for fpath_str, stats, err in pass1_results:
        if stats is not None and err is None:
            image_stats[fpath_str] = stats  # (mean, std) in LAB

    print(f"\nPass 1 complete: {len(image_stats)}/{total} images "
          f"corrected in {t1 - t0:.1f}s")
    if pass1_errors:
        print(f"  {pass1_errors} errors occurred.")

    # ------------------------------------------------------------------
    # Pass 2: Iterative graph-based inter-image seam minimization (LAB)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PASS 2: Graph-based seam minimization in LAB "
          f"(strength={INTER_STRENGTH}, iterations={PASS2_ITERATIONS})")
    print(f"{'='*60}")

    # Build adjacency graph from pass 1 intermediate file paths
    intermediate_paths = [inter for _, inter, _ in all_files]
    print(f"Building adjacency graph for {len(intermediate_paths)} tiles...")
    edges = build_adjacency_graph(intermediate_paths)
    print(f"  Found {len(edges)} edges (neighbor pairs)")

    if not edges:
        print("No adjacency edges found. Skipping pass 2.")
        print("Converting intermediate files to final JPEG...")
        for _, intermediate, final_output in all_files:
            if os.path.exists(intermediate):
                img = Image.open(intermediate)
                img.save(final_output, "JPEG", quality=JPEG_QUALITY)
                os.remove(intermediate)
        print(f"\nTotal time: {t1 - t0:.1f}s")
        print("Done.")
        return

    # Cache land masks for border sampling
    mask_cache = {}

    def get_tile_mask(tile_path):
        if mask_zl is None or not MASK_DIR:
            return None
        if tile_path in mask_cache:
            return mask_cache[tile_path]
        coords = parse_tile_coords(tile_path)
        texture_zl = parse_zl_from_dir(Path(tile_path).parent)
        mask = None
        if coords is not None and texture_zl is not None:
            til_x, til_y = coords
            mask = get_land_mask(MASK_DIR, til_x, til_y, texture_zl, mask_zl)
        mask_cache[tile_path] = mask
        return mask

    # Segment center positions (normalized 0..1 along the border)
    n_seg = BORDER_SAMPLE_SEGMENTS
    H_POS = [(i + 0.5) / n_seg for i in range(n_seg)]

    # Cross-ZL anchoring data (computed once from pass 1 stats)
    input_to_intermediate = {str(inp): inter for inp, inter, _ in all_files}
    path_to_lab_mean = {}
    for fpath_str, stats, err in pass1_results:
        if stats is not None and err is None:
            lab_mean, _ = stats
            outp = input_to_intermediate.get(fpath_str)
            if outp is not None:
                path_to_lab_mean[outp] = lab_mean

    dir_to_indices = defaultdict(list)
    for idx, outp in enumerate(intermediate_paths):
        parent = str(Path(outp).parent)
        dir_to_indices[parent].append(idx)

    # The current source paths for sampling (starts as intermediate PNGs)
    current_paths = list(intermediate_paths)

    t2 = time.time()

    for iteration in range(PASS2_ITERATIONS):
        is_final = (iteration == PASS2_ITERATIONS - 1)
        print(f"\n--- Iteration {iteration + 1}/{PASS2_ITERATIONS} "
              f"{'(final)' if is_final else ''} ---")

        # Sample border LAB statistics and build equations
        print("  Sampling border colors...")
        border_equations = []
        skipped = 0

        for idx, (i, j, edge_type) in enumerate(edges):
            path_i = current_paths[i]
            path_j = current_paths[j]
            mask_i = get_tile_mask(intermediate_paths[i])
            mask_j = get_tile_mask(intermediate_paths[j])

            if edge_type == 'horizontal':
                stats_i = sample_border_stats(path_i, 'horizontal', 'right', mask_i)
                stats_j = sample_border_stats(path_j, 'horizontal', 'left', mask_j)
            else:
                stats_i = sample_border_stats(path_i, 'vertical', 'bottom', mask_i)
                stats_j = sample_border_stats(path_j, 'vertical', 'top', mask_j)

            if stats_i is not None and stats_j is not None:
                means_i, weights_i = stats_i
                means_j, weights_j = stats_j
                if edge_type == 'horizontal':
                    for k in range(n_seg):
                        v = H_POS[k]
                        gap = means_j[k] - means_i[k]
                        # Bilinear: tile i at u=1, tile j at u=0
                        coeffs = [
                            (i, 1.0, 1.0, v, v),
                            (j, -1.0, 0.0, -v, 0.0),
                        ]
                        w = min(weights_i[k], weights_j[k])
                        border_equations.append((coeffs, gap, w))
                else:
                    for k in range(n_seg):
                        u = H_POS[k]
                        gap = means_j[k] - means_i[k]
                        # Bilinear: tile i at v=1, tile j at v=0
                        coeffs = [
                            (i, 1.0, u, 1.0, u),
                            (j, -1.0, -u, 0.0, 0.0),
                        ]
                        w = min(weights_i[k], weights_j[k])
                        border_equations.append((coeffs, gap, w))
            else:
                skipped += 1

        print(f"  {len(border_equations)} equations ({skipped} edges skipped)")

        # Solve
        print("  Solving least-squares...")
        corrections = solve_graph_corrections(len(current_paths),
                                              border_equations)

        # Cross-directory anchoring (only on first iteration)
        if iteration == 0 and len(dir_to_indices) > 1 and path_to_lab_mean:
            def center_correction(idx):
                return np.array([
                    corrections[idx, ch, 0] + corrections[idx, ch, 1] * 0.5 +
                    corrections[idx, ch, 2] * 0.5 + corrections[idx, ch, 3] * 0.25
                    for ch in range(3)
                ], dtype=np.float32)

            all_corrected_means = []
            for idx, outp in enumerate(intermediate_paths):
                if outp in path_to_lab_mean:
                    all_corrected_means.append(
                        path_to_lab_mean[outp] + center_correction(idx))
            if all_corrected_means:
                global_lab_mean = np.mean(all_corrected_means, axis=0)
                for parent, indices in dir_to_indices.items():
                    dir_means = []
                    for idx in indices:
                        outp = intermediate_paths[idx]
                        if outp in path_to_lab_mean:
                            dir_means.append(
                                path_to_lab_mean[outp] + center_correction(idx))
                    if dir_means:
                        dir_mean = np.mean(dir_means, axis=0)
                        shift = global_lab_mean - dir_mean
                        for idx in indices:
                            for ch in range(3):
                                corrections[idx, ch, 0] += shift[ch]
                        dir_name = Path(parent).name
                        print(f"  Cross-ZL anchor shift for {dir_name}: "
                              f"dL={shift[0]:.2f}, da={shift[1]:.2f}, "
                              f"db={shift[2]:.2f}")

        # Report statistics
        center_corrs = np.array([
            [corrections[i, ch, 0] + corrections[i, ch, 1] * 0.5 +
             corrections[i, ch, 2] * 0.5 + corrections[i, ch, 3] * 0.25
             for ch in range(3)]
            for i in range(len(current_paths))
        ])
        max_corr = np.max(np.abs(center_corrs), axis=0)
        mean_corr = np.mean(np.abs(center_corrs), axis=0)
        print(f"  Max  center correction (L, a, b): "
              f"({max_corr[0]:.2f}, {max_corr[1]:.2f}, {max_corr[2]:.2f})")
        print(f"  Mean center correction (L, a, b): "
              f"({mean_corr[0]:.2f}, {mean_corr[1]:.2f}, {mean_corr[2]:.2f})")

        # Apply corrections
        print("  Applying corrections...")
        pass2_tasks = []
        if is_final:
            # Final iteration: write JPEG to final output paths
            for idx, (_, intermediate, final_output) in enumerate(all_files):
                src = current_paths[idx]
                if os.path.exists(src):
                    pass2_tasks.append((Path(src), final_output,
                                        corrections[idx], INTER_STRENGTH,
                                        JPEG_QUALITY))
        else:
            # Non-final iteration: write PNG, overwriting intermediates
            for idx in range(len(all_files)):
                src = current_paths[idx]
                dst = intermediate_paths[idx]
                if os.path.exists(src):
                    pass2_tasks.append((Path(src), dst,
                                        corrections[idx], INTER_STRENGTH,
                                        JPEG_QUALITY))

        pass2_results, pass2_errors = run_parallel(
            pass2_tasks, pass2_worker,
            f"Pass 2 iter {iteration + 1}", len(pass2_tasks)
        )
        if pass2_errors:
            print(f"  {pass2_errors} errors occurred.")

        # For next iteration, source from intermediate paths
        current_paths = list(intermediate_paths)

    t3 = time.time()

    # Clean up intermediate PNG files
    print("Cleaning up intermediate files...")
    for _, intermediate, _ in all_files:
        if os.path.exists(intermediate):
            os.remove(intermediate)

    print(f"\nPass 2 complete: {total} images corrected in {t3 - t2:.1f}s")
    print(f"\nTotal time: {t3 - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
