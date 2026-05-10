#!/usr/bin/env python3
"""
Verify Polish NMT GeoTIFF files against Copernicus GLO-30 DEM.

For each *_PL_NMT.tif file found in INPUT_DIR (recursively), downloads
the corresponding Copernicus GLO-30 tile, resamples it to match the NMT
grid, and compares elevation values pixel by pixel.

Pixels where the absolute difference exceeds OUTLIER_THRESHOLD_M are
counted as outliers. A per-file and overall summary is printed.

Requirements:
    - GDAL Python bindings (osgeo.gdal)
    - numpy
"""

import os
import sys
import re
import glob

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL Python bindings required.")
    sys.exit(1)

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
INPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Elevation_data",
)

# Pixels differing by more than this (meters) are outliers
OUTLIER_THRESHOLD_M = 30.0

# Print per-file histogram of differences
SHOW_HISTOGRAM = True
# ───────────────────────────────────────────────────────────────────────


def find_nmt_files(input_dir):
    """Find all *_PL_NMT.tif files recursively."""
    pattern = os.path.join(input_dir, "**", "*_PL_NMT.tif")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def parse_lat_lon_from_filename(filepath):
    """Extract lat/lon from filename like N51E020_PL_NMT.tif."""
    basename = os.path.basename(filepath)
    m = re.match(r"([NS])(\d+)([EW])(\d+)_PL_NMT\.tif", basename)
    if not m:
        return None, None
    lat = int(m.group(2))
    if m.group(1) == "S":
        lat = -lat
    lon = int(m.group(4))
    if m.group(3) == "W":
        lon = -lon
    return lat, lon


def get_copernicus_tile(lat, lon, target_w, target_h, target_gt):
    """Download and resample Copernicus GLO-30 DEM to match NMT grid.

    Returns a numpy array (target_h x target_w) or None on failure.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    tile_name = "Copernicus_DSM_COG_10_{}{:02d}_00_{}{:03d}_00_DEM".format(
        ns, abs(lat), ew, abs(lon)
    )
    url = "https://copernicus-dem-30m.s3.amazonaws.com/{}/{}.tif".format(
        tile_name, tile_name
    )

    try:
        vsi_path = "/vsicurl/" + url
        ref_ds = gdal.Open(vsi_path)
        if ref_ds is None:
            return None

        # Warp to match NMT grid exactly
        out_bounds = (
            target_gt[0],
            target_gt[3] + target_h * target_gt[5],
            target_gt[0] + target_w * target_gt[1],
            target_gt[3],
        )
        warped_path = "/vsimem/cop_verify_{}{}.tif".format(lat, lon)
        warp_opts = gdal.WarpOptions(
            dstSRS="EPSG:4326",
            outputBounds=out_bounds,
            width=target_w,
            height=target_h,
            resampleAlg="bilinear",
            format="GTiff",
        )
        warped = gdal.Warp(warped_path, ref_ds, options=warp_opts)
        ref_ds = None

        if warped is None:
            return None

        arr = warped.GetRasterBand(1).ReadAsArray().astype(np.float32)
        nodata = warped.GetRasterBand(1).GetNoDataValue()
        warped = None
        gdal.Unlink(warped_path)

        if nodata is not None:
            arr[arr == nodata] = np.nan

        return arr

    except Exception as e:
        print("    Copernicus download failed: {}".format(e))
        return None


def verify_file(filepath):
    """Compare a single NMT file against Copernicus. Returns stats dict."""
    lat, lon = parse_lat_lon_from_filename(filepath)
    if lat is None:
        print("  SKIP: Cannot parse lat/lon from {}".format(filepath))
        return None

    print("\n  Verifying {} (lat={}, lon={})".format(
        os.path.basename(filepath), lat, lon
    ))

    ds = gdal.Open(filepath)
    if ds is None:
        print("    ERROR: Cannot open file")
        return None

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    nmt_arr = band.ReadAsArray().astype(np.float32)
    w, h = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    ds = None

    # Mark nodata as NaN
    if nodata is not None:
        nmt_arr[nmt_arr == nodata] = np.nan

    # Get Copernicus reference
    print("    Downloading Copernicus GLO-30 reference...")
    cop_arr = get_copernicus_tile(lat, lon, w, h, gt)
    if cop_arr is None:
        print("    ERROR: Could not get Copernicus reference tile")
        return None

    # Compute differences only where both have valid data
    valid_mask = np.isfinite(nmt_arr) & np.isfinite(cop_arr)
    valid_count = int(valid_mask.sum())
    total_pixels = nmt_arr.size
    nmt_nodata_count = int(np.isnan(nmt_arr).sum())

    if valid_count == 0:
        print("    ERROR: No overlapping valid pixels")
        return None

    diff = nmt_arr[valid_mask] - cop_arr[valid_mask]
    abs_diff = np.abs(diff)

    outlier_mask = abs_diff > OUTLIER_THRESHOLD_M
    outlier_count = int(outlier_mask.sum())
    outlier_pct = 100.0 * outlier_count / valid_count

    stats = {
        "file": os.path.basename(filepath),
        "lat": lat,
        "lon": lon,
        "total_pixels": total_pixels,
        "nmt_nodata": nmt_nodata_count,
        "valid_compared": valid_count,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "std_diff": float(np.std(diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "min_diff": float(np.min(diff)),
        "max_diff": float(np.max(diff)),
        "outliers": outlier_count,
        "outlier_pct": outlier_pct,
    }

    # Print results
    print("    Size:           {}x{} ({} pixels)".format(w, h, total_pixels))
    print("    NMT nodata:     {} ({:.1f}%)".format(
        nmt_nodata_count, 100.0 * nmt_nodata_count / total_pixels
    ))
    print("    Compared:       {} pixels".format(valid_count))
    print("    Mean diff:      {:.2f} m (NMT - Copernicus)".format(stats["mean_diff"]))
    print("    Median diff:    {:.2f} m".format(stats["median_diff"]))
    print("    Std dev:        {:.2f} m".format(stats["std_diff"]))
    print("    Range:          {:.2f} to {:.2f} m".format(stats["min_diff"], stats["max_diff"]))
    print("    Outliers:       {} ({:.2f}%) [threshold: {} m]".format(
        outlier_count, outlier_pct, OUTLIER_THRESHOLD_M
    ))

    if SHOW_HISTOGRAM:
        # Simple text histogram of differences
        bins = [-np.inf, -100, -50, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 50, 100, np.inf]
        counts, edges = np.histogram(diff, bins=bins)
        print("    Difference distribution (NMT - Copernicus):")
        max_bar = max(counts) if max(counts) > 0 else 1
        for i in range(len(counts)):
            bar_len = int(40 * counts[i] / max_bar)
            bar = "#" * bar_len
            pct = 100.0 * counts[i] / valid_count
            label_lo = "<" if edges[i] == -np.inf else "{:>5.0f}".format(edges[i])
            label_hi = ">" if edges[i + 1] == np.inf else "{:>5.0f}".format(edges[i + 1])
            print("      {} to {} m: {:>8} ({:>5.1f}%) {}".format(
                label_lo, label_hi, counts[i], pct, bar
            ))

    if outlier_pct > 5.0:
        print("    WARNING: High outlier rate!")
    elif outlier_pct > 1.0:
        print("    NOTE: Moderate outlier rate")
    else:
        print("    OK: Low outlier rate")

    return stats


def main():
    input_dir = INPUT_DIR
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]

    print("Verifying PL NMT files in: {}".format(input_dir))
    print("Outlier threshold: {} m".format(OUTLIER_THRESHOLD_M))
    print("NOTE: NMT is DTM (bare earth), Copernicus is DSM (includes trees/buildings).")
    print("      Expect systematic negative bias (NMT lower than Copernicus) in forested areas.")

    files = find_nmt_files(input_dir)
    if not files:
        print("No *_PL_NMT.tif files found.")
        sys.exit(1)

    print("Found {} files to verify".format(len(files)))

    all_stats = []
    for f in files:
        s = verify_file(f)
        if s is not None:
            all_stats.append(s)

    # Overall summary
    if all_stats:
        print("\n" + "=" * 70)
        print("SUMMARY ({} files verified)".format(len(all_stats)))
        print("=" * 70)
        print("{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "File", "Compared", "Mean(m)", "StdDev(m)", "Outliers", "Out%"
        ))
        print("-" * 70)

        total_outliers = 0
        total_compared = 0
        for s in sorted(all_stats, key=lambda x: (x["lat"], x["lon"])):
            print("{:<20} {:>10} {:>10.2f} {:>10.2f} {:>10} {:>9.2f}%".format(
                s["file"], s["valid_compared"], s["mean_diff"],
                s["std_diff"], s["outliers"], s["outlier_pct"]
            ))
            total_outliers += s["outliers"]
            total_compared += s["valid_compared"]

        print("-" * 70)
        overall_pct = 100.0 * total_outliers / total_compared if total_compared else 0
        print("{:<20} {:>10} {:>10} {:>10} {:>10} {:>9.2f}%".format(
            "TOTAL", total_compared, "", "", total_outliers, overall_pct
        ))


if __name__ == "__main__":
    main()
