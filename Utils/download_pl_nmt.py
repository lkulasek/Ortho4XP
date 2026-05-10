#!/usr/bin/env python3
"""
Download Polish NMT (Numeryczny Model Terenu) elevation data from GUGiK
and produce a GeoTIFF compatible with Ortho4XP.

Usage:
    python3 download_pl_nmt.py <lat> <lon> [--resolution 5] [--year 2024]

Example:
    python3 download_pl_nmt.py 51 20
    python3 download_pl_nmt.py 50 19 --resolution 1 --year 2023

This will create:
    Elevation_data/+50+020/N51E020_PL_NMT.tif

The script queries the GUGiK WFS service to find all NMT sheet tiles
covering the 1x1 degree area, downloads them as ASC files, and merges
them into a single GeoTIFF reprojected to EPSG:4326 for Ortho4XP.

Requirements:
    - GDAL (osgeo.gdal) with Python bindings
    - Internet connection
"""

import argparse
import os
import re
import sys
import time
import tempfile
import shutil
import threading
import ssl
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# GUGiK servers have SSL certificate issues on some platforms (especially Windows).
# Use a non-verifying context for all urllib requests to GUGiK.
_ssl_context = ssl._create_unverified_context()

try:
    import requests as req_lib
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from osgeo import gdal, osr

    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL Python bindings required. Install with:")
    print("  pip install GDAL")
    print("  or: conda install -c conda-forge gdal")
    sys.exit(1)

# Ortho4XP directory structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORTHO4XP_DIR = os.path.dirname(SCRIPT_DIR)
ELEVATION_DIR = os.path.join(ORTHO4XP_DIR, "Elevation_data")

WFS_BASE = (
    "https://mapy.geoportal.gov.pl/wss/service/PZGIK/"
    "NumerycznyModelTerenuEVRF2007/WFS/Skorowidze"
)

# Available years in WFS
AVAILABLE_YEARS = list(range(2018, 2026))


def hem_latlon(lat, lon):
    """Match Ortho4XP naming: N51E020"""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return "{}{}{}{}".format(ns, str(abs(lat)).zfill(2), ew, str(abs(lon)).zfill(3))


def round_latlon(lat, lon):
    """Match Ortho4XP directory grouping: +50+020"""
    from math import floor
    strlatround = "{:+.0f}".format(floor(lat / 10) * 10).zfill(3)
    strlonround = "{:+.0f}".format(floor(lon / 10) * 10).zfill(4)
    return strlatround + strlonround


def output_path(lat, lon):
    rll = round_latlon(lat, lon)
    hll = hem_latlon(lat, lon)
    d = os.path.join(ELEVATION_DIR, rll)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hll + "_PL_NMT.tif")


def query_wfs_sheets(lat, lon, year, min_resolution=None):
    """Query GUGiK WFS for all NMT sheets covering the 1x1 degree tile."""
    layer = "gugik:SkorowidzNMT{}".format(year)
    bbox = "{},{},{},{},EPSG:4326".format(lat, lon, lat + 1, lon + 1)

    all_sheets = []
    start = 0
    page_size = 200

    while True:
        url = (
            "{}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES={}&BBOX={}&COUNT={}&STARTINDEX={}"
        ).format(WFS_BASE, layer, bbox, page_size, start)

        try:
            resp = urllib.request.urlopen(url, timeout=60, context=_ssl_context)
            data = resp.read().decode(errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print("  WFS query failed for year {}: {}".format(year, e))
            break

        urls = re.findall(
            r"<gugik:url_do_pobrania>(.*?)</gugik:url_do_pobrania>", data
        )
        resols = re.findall(
            r"<gugik:char_przestrz>(.*?)</gugik:char_przestrz>", data
        )
        godlos = re.findall(
            r"<gugik:godlo>(.*?)</gugik:godlo>", data
        )

        if not urls:
            break

        for u, r, g in zip(urls, resols, godlos):
            res_m = float(r.replace(" m", "").strip())
            if min_resolution is not None and res_m > min_resolution:
                continue
            all_sheets.append({"url": u, "resolution": res_m, "godlo": g, "year": year})

        start += page_size
        if len(urls) < page_size:
            break

    return all_sheets


_thread_local = threading.local()

def _get_session():
    """Get a per-thread requests.Session for connection reuse."""
    if not hasattr(_thread_local, "session"):
        s = req_lib.Session()
        adapter = req_lib.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=10,
            max_retries=req_lib.adapters.Retry(total=3, backoff_factor=1),
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session


def download_file(url, dest_path, retries=3):
    """Download a single file with retries."""
    if _HAS_REQUESTS:
        session = _get_session()
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception as e:
            print("  FAILED: {} -> {}".format(url, e))
            return False
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(url, timeout=60, context=_ssl_context)
            with open(dest_path, "wb") as f:
                f.write(resp.read())
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print("  FAILED: {} -> {}".format(url, e))
                return False
    return False


def download_sheets(sheets, tmp_dir, max_workers=8):
    """Download all sheet files in parallel, return list of local paths."""
    local_files = []
    total = len(sheets)
    done = 0
    failed = 0
    start_time = time.time()

    def _download(sheet, idx):
        fname = os.path.basename(sheet["url"])
        local_path = os.path.join(tmp_dir, fname)
        ok = download_file(sheet["url"], local_path)
        return local_path if ok else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download, s, i): i for i, s in enumerate(sheets)
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result:
                local_files.append(result)
            else:
                failed += 1

            elapsed = time.time() - start_time
            remaining = total - done
            if done > 0 and elapsed > 0:
                eta = elapsed / done * remaining
                eta_min, eta_sec = divmod(int(eta), 60)
                print(
                    "  Downloaded {}/{} ({} failed)  [ETA {}m{:02d}s]".format(
                        done, total, failed, eta_min, eta_sec
                    ),
                    end="\r",
                )

    print()  # newline after progress
    return local_files


def merge_to_geotiff(asc_files, output_file, lat, lon, target_resolution=None):
    """Merge ASC files into a single GeoTIFF reprojected to EPSG:4326."""
    if not asc_files:
        print("ERROR: No files to merge.")
        return False

    print("  Merging {} files...".format(len(asc_files)))

    # Build VRT from all ASC files (they're in EPSG:2180 / PL-1992)
    tmp_dir = os.path.dirname(asc_files[0])
    vrt_path = os.path.join(tmp_dir, "merged.vrt")

    # Create VRT - all ASC files are in EPSG:2180
    vrt_options = gdal.BuildVRTOptions(resampleAlg="bilinear")
    vrt_ds = gdal.BuildVRT(vrt_path, asc_files, options=vrt_options)
    if vrt_ds is None:
        print("ERROR: Failed to build VRT")
        return False

    # Set the source CRS to EPSG:2180 (PL-1992) for files that lack it
    srs_2180 = osr.SpatialReference()
    srs_2180.ImportFromEPSG(2180)
    vrt_ds.SetProjection(srs_2180.ExportToWkt())
    vrt_ds = None  # close

    # Warp to EPSG:4326, clipping to the 1x1 degree tile
    # Target resolution: ~1 arc-second = ~30m for good quality
    if target_resolution is None:
        target_resolution = 1.0 / 10800  # ~1/3 arc-second (~10m), pixel-center registration

    print("  Reprojecting to EPSG:4326 (resolution: {:.6f} deg)...".format(
        target_resolution
    ))

    warp_options = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        outputBounds=(lon, lat, lon + 1, lat + 1),
        xRes=target_resolution,
        yRes=target_resolution,
        resampleAlg="bilinear",
        format="GTiff",
        outputType=gdal.GDT_Float32,
        srcNodata=-9999,
        dstNodata=-32768,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
        multithread=True,
    )

    result = gdal.Warp(output_file, vrt_path, options=warp_options)
    if result is None:
        print("ERROR: gdalwarp failed")
        return False

    result = None  # close

    # Fill nodata gaps with Copernicus GLO-30 DEM
    _fill_gaps_from_copernicus(output_file, lat, lon)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print("  Output: {} ({:.1f} MB)".format(output_file, size_mb))
    return True


def _fill_gaps_from_copernicus(output_file, lat, lon):
    """Fill nodata pixels in the output GeoTIFF with Copernicus GLO-30 data."""
    import numpy

    ds = gdal.Open(output_file, gdal.GA_Update)
    if ds is None:
        print("  WARNING: Could not open output file for gap-filling")
        return
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray()
    w, h = ds.RasterXSize, ds.RasterYSize

    nodata_mask = (arr == nodata) if nodata is not None else numpy.zeros_like(arr, dtype=bool)
    nodata_count = int(nodata_mask.sum())
    if nodata_count == 0:
        ds = None
        return

    print("  Filling {} nodata pixels from Copernicus GLO-30...".format(nodata_count))

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
            print("  WARNING: Could not open Copernicus DEM, gaps unfilled")
            ds = None
            return

        # Warp Copernicus to match our output grid exactly
        ref_warped_path = "/vsimem/copernicus_warped.tif"
        gt = ds.GetGeoTransform()
        warp_opts = gdal.WarpOptions(
            dstSRS="EPSG:4326",
            outputBounds=(gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3]),
            width=w,
            height=h,
            resampleAlg="bilinear",
            format="GTiff",
        )
        warped = gdal.Warp(ref_warped_path, ref_ds, options=warp_opts)
        ref_ds = None

        if warped is None:
            print("  WARNING: Copernicus warp failed, gaps unfilled")
            ds = None
            return

        ref_arr = warped.GetRasterBand(1).ReadAsArray()
        ref_nodata = warped.GetRasterBand(1).GetNoDataValue()
        warped = None
        gdal.Unlink(ref_warped_path)

        # Only fill where we have nodata AND Copernicus has valid data
        if ref_nodata is not None:
            fill_mask = nodata_mask & (ref_arr != ref_nodata)
        else:
            fill_mask = nodata_mask

        filled_count = int(fill_mask.sum())
        arr[fill_mask] = ref_arr[fill_mask]

        band.WriteArray(arr)
        band.FlushCache()
        ds = None

        remaining = nodata_count - filled_count
        print("  Filled {} pixels from Copernicus ({} remain as nodata — sea/outside)".format(
            filled_count, remaining
        ))

    except Exception as e:
        print("  WARNING: Copernicus gap-fill failed: {}".format(e))
        ds = None


def validate_geotiff(output_file, lat, lon):
    """Validate the output GeoTIFF for completeness and correctness.

    Downloads a reference DEM (Copernicus GLO-30) to distinguish sea/outside
    areas (expected nodata) from missing source tiles (unexpected gaps).
    """
    import numpy

    print("\nValidating {}...".format(output_file))

    ds = gdal.Open(output_file)
    if ds is None:
        print("  ERROR: Cannot open file")
        return False

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    w, h = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    stats = band.ComputeStatistics(False)
    arr = band.ReadAsArray()
    ds = None

    total = arr.size
    nodata_mask = (arr == nodata) if nodata is not None else numpy.zeros_like(arr, dtype=bool)
    nodata_count = int(nodata_mask.sum())
    zero_count = int((arr == 0).sum())
    valid_count = total - nodata_count

    print("  Size:       {}x{} pixels".format(w, h))
    print("  Bounds:     ({:.4f}, {:.4f}) -> ({:.4f}, {:.4f})".format(
        gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3]
    ))
    print("  Pixel size: {:.6f} x {:.6f} deg".format(gt[1], abs(gt[5])))
    print("  Elevation:  min={:.1f}m  max={:.1f}m  mean={:.1f}m  stddev={:.1f}m".format(
        stats[0], stats[1], stats[2], stats[3]
    ))
    print("  Nodata:     {} pixels ({:.1f}%)".format(nodata_count, 100.0 * nodata_count / total))
    print("  Zero:       {} pixels ({:.1f}%)".format(zero_count, 100.0 * zero_count / total))
    print("  Valid:      {} pixels ({:.1f}%)".format(valid_count, 100.0 * valid_count / total))

    if stats[0] == stats[1]:
        print("  WARNING: All pixels have the same value — file may be empty")
        return False
    if w != h:
        print("  NOTE: Non-square raster ({}x{})".format(w, h))

    # If there's nodata, download a reference DEM to check land vs sea
    if nodata_count > 0:
        print("\n  Downloading reference DEM (Copernicus GLO-30) for land/sea check...")
        ref_land = _get_reference_land_mask(lat, lon, w, h)
        if ref_land is not None:
            land_nodata = int((nodata_mask & ref_land).sum())
            sea_nodata = int((nodata_mask & ~ref_land).sum())
            land_total = int(ref_land.sum())
            sea_total = total - land_total

            print("  Reference DEM: {:.1f}% land, {:.1f}% sea".format(
                100.0 * land_total / total, 100.0 * sea_total / total
            ))
            print("  Nodata on land:  {} pixels ({:.1f}% of land)".format(
                land_nodata, 100.0 * land_nodata / land_total if land_total else 0
            ))
            print("  Nodata on sea:   {} pixels (expected)".format(sea_nodata))

            if land_nodata > 0.01 * land_total:
                print("  WARNING: {:.1f}% of land pixels have no data — missing source tiles!".format(
                    100.0 * land_nodata / land_total
                ))
                return False
            else:
                print("  OK: Land coverage is complete ({:.2f}% land nodata)".format(
                    100.0 * land_nodata / land_total if land_total else 0
                ))
                return True
        else:
            print("  Could not download reference DEM, skipping land/sea check")
            if nodata_count > 0.01 * total:
                print("  WARNING: {:.1f}% nodata — cannot determine if sea or missing tiles".format(
                    100.0 * nodata_count / total
                ))
                return False

    print("  OK: Coverage looks complete")
    return True


def _get_reference_land_mask(lat, lon, target_w, target_h):
    """Download Copernicus GLO-30 DEM tile and create a land mask.

    Returns a boolean array (target_h x target_w) where True = land.
    Uses the fact that Copernicus DEM has valid data over land and nodata/0 over ocean.
    """
    import numpy

    # Copernicus GLO-30 on AWS (public, no auth needed)
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    tile_name = "Copernicus_DSM_COG_10_{}{:02d}_00_{}{:03d}_00_DEM".format(
        ns, abs(lat), ew, abs(lon)
    )
    url = "https://copernicus-dem-30m.s3.amazonaws.com/{}/{}.tif".format(
        tile_name, tile_name
    )

    try:
        # Use GDAL's virtual filesystem to read directly from URL
        # (avoids downloading the whole file if not needed)
        vsi_path = "/vsicurl/" + url
        ref_ds = gdal.Open(vsi_path)
        if ref_ds is None:
            print("  Could not open reference DEM from: {}".format(url))
            return None

        ref_band = ref_ds.GetRasterBand(1)
        ref_nodata = ref_band.GetNoDataValue()
        ref_arr = ref_band.ReadAsArray()
        ref_ds = None

        # Land = where reference DEM has valid (non-nodata) data
        if ref_nodata is not None:
            land = ref_arr != ref_nodata
        else:
            # Copernicus DEM uses 0 for sea in some tiles
            land = ref_arr != 0

        # Resize to match our target resolution using GDAL warp
        ref_h, ref_w = land.shape
        if ref_h == target_h and ref_w == target_w:
            return land

        # Use nearest-neighbor sampling via numpy repeat/slice
        row_idx = numpy.linspace(0, ref_h - 1, target_h).astype(int)
        col_idx = numpy.linspace(0, ref_w - 1, target_w).astype(int)
        return land[numpy.ix_(row_idx, col_idx)]

    except Exception as e:
        print("  Reference DEM download failed: {}".format(e))
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download Polish NMT elevation data for Ortho4XP"
    )
    parser.add_argument("lat", type=int, help="Tile latitude (e.g. 51)")
    parser.add_argument("lon", type=int, help="Tile longitude (e.g. 20)")
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Max source resolution in meters (default: all available, typically 5m)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="NMT data year (default: try latest first)",
    )
    parser.add_argument(
        "--target-arcsec",
        type=float,
        default=1.0 / 3,
        help='Target resolution in arc-seconds (default: 1/3" = ~10m)',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Parallel download threads (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )

    args = parser.parse_args()

    out_file = output_path(args.lat, args.lon)
    if os.path.exists(out_file) and not args.force:
        print("Output already exists: {}".format(out_file))
        print("Use --force to overwrite.")
        return

    print(
        "Downloading Polish NMT for tile {}\n".format(
            hem_latlon(args.lat, args.lon)
        )
    )

    # Query WFS for available sheets (all years in parallel)
    years = [args.year] if args.year else sorted(AVAILABLE_YEARS, reverse=True)
    all_sheets = []

    print("  Querying WFS for {} year(s)...".format(len(years)))
    with ThreadPoolExecutor(max_workers=len(years)) as executor:
        futures = {
            executor.submit(query_wfs_sheets, args.lat, args.lon, y, args.resolution): y
            for y in years
        }
        for future in as_completed(futures):
            year = futures[future]
            sheets = future.result()
            if sheets:
                print("  Found {} sheets (year {})".format(len(sheets), year))
                all_sheets.extend(sheets)

    if not all_sheets:
        print("ERROR: No NMT data found for this tile.")
        print("This tile may be outside Poland (49-55N, 14-24E).")
        sys.exit(1)

    # Deduplicate by godlo — keep newest year (years are queried newest-first)
    seen_godlo = {}
    for s in all_sheets:
        g = s["godlo"]
        if g not in seen_godlo or s["year"] > seen_godlo[g]["year"]:
            seen_godlo[g] = s
    all_sheets = list(seen_godlo.values())

    # Also deduplicate by URL (safety)
    seen = set()
    unique_sheets = []
    for s in all_sheets:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique_sheets.append(s)
    all_sheets = unique_sheets

    res_counts = {}
    for s in all_sheets:
        r = s["resolution"]
        res_counts[r] = res_counts.get(r, 0) + 1
    year_counts = {}
    for s in all_sheets:
        y = s["year"]
        year_counts[y] = year_counts.get(y, 0) + 1
    print("  After dedup: {} unique sheets".format(len(all_sheets)))
    print("  Resolution breakdown: {}".format(
        ", ".join(
            "{:.0f}m: {}".format(r, c) for r, c in sorted(res_counts.items())
        )
    ))
    print("  Year breakdown: {}".format(
        ", ".join(
            "{}: {}".format(y, c) for y, c in sorted(year_counts.items(), reverse=True)
        )
    ))

    # Download to temp directory
    tmp_dir = tempfile.mkdtemp(prefix="pl_nmt_")
    try:
        print("\n  Downloading {} files...".format(len(all_sheets)))
        local_files = download_sheets(all_sheets, tmp_dir, args.workers)

        if not local_files:
            print("ERROR: No files were downloaded successfully.")
            sys.exit(1)

        print("  Downloaded {}/{} files\n".format(len(local_files), len(all_sheets)))

        # Merge and reproject
        target_res = args.target_arcsec / 3600.0
        if not merge_to_geotiff(local_files, out_file, args.lat, args.lon, target_res):
            sys.exit(1)

    finally:
        print("  Cleaning up temp files...")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    validate_geotiff(out_file, args.lat, args.lon)

    print(
        "\nDone! To use in Ortho4XP, set custom_dem to:\n"
        '  "Polish NMT (from GUGiK) - Poland, auto-download"\n'
        "Or use the file directly as custom_dem path:\n"
        "  {}".format(out_file)
    )


if __name__ == "__main__":
    main()
