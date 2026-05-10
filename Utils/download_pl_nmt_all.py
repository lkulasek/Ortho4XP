#!/usr/bin/env python3
"""
Download Polish NMT elevation data for ALL tiles covering Poland.

Uses download_pl_nmt.py for each 1x1 degree tile, running multiple
tiles in parallel using subprocess workers.

Poland approximate bounds: lat 49-54, lon 14-24
"""

import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# All 1x1 degree tiles that intersect Poland's land area
POLAND_TILES = [
    (lat, lon)
    for lat in range(49, 55)
    for lon in range(14, 25)
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_SCRIPT = os.path.join(SCRIPT_DIR, "download_pl_nmt.py")

MAX_PARALLEL_TILES = 2


def download_tile(lat_lon):
    lat, lon = lat_lon
    print(f"[START] Tile lat={lat} lon={lon}")
    result = subprocess.run(
        [sys.executable, DOWNLOAD_SCRIPT, str(lat), str(lon)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"[OK] Tile lat={lat} lon={lon}")
    return (lat, lon, result.returncode)


def main():
    print(f"Downloading NMT for {len(POLAND_TILES)} tiles covering Poland")
    print(f"Parallel tiles: {MAX_PARALLEL_TILES}\n")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TILES) as executor:
        futures = {
            executor.submit(download_tile, tile): tile for tile in POLAND_TILES
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Summary
    succeeded = [r for r in results if r[2] == 0]
    failed = [r for r in results if r[2] != 0]

    print(f"\n{'='*60}")
    print(f"DONE: {len(succeeded)}/{len(results)} tiles had data")
    if failed:
        print(f"Skipped {len(failed)} tiles (no data available — outside Poland)")


if __name__ == "__main__":
    main()
