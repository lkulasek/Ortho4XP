"""Simple test to verify directory creation works"""

import os
import tempfile
import shutil
from pathlib import Path

# Create a temporary test directory
test_dir = tempfile.mkdtemp(prefix="test_unpack_")
print(f"Created test directory: {test_dir}")

# Create some fake DSM files with different coordinates
test_files = [
    "ALPSMLC30_N054E026_DSM.tif",
    "ALPSMLC30_N050E010_DSM.tif",
    "ALPSMLC30_N060E020_DSM.tif",
    "ALPSMLC30_N040E030_DSM.tif",
]

print("\nCreating test files:")
for filename in test_files:
    filepath = Path(test_dir) / filename
    filepath.write_text("fake tif content")
    print(f"  Created: {filename}")

# Now run the rename logic
import re

def get_tile_directory(lat_num, lon_num, lat_prefix, lon_prefix):
    """Calculate tile directory name in format +XX+XXX or -XX-XXX."""
    # Apply sign based on N/S and E/W first
    if lat_prefix == 'S':
        lat_num = -lat_num
    if lon_prefix == 'W':
        lon_num = -lon_num

    # Round down to nearest 10 for directory grouping (floor division)
    lat_dir = (lat_num // 10) * 10
    lon_dir = (lon_num // 10) * 10

    # Format as +XX+XXX or -XX-XXX
    lat_sign = '+' if lat_dir >= 0 else '-'
    lon_sign = '+' if lon_dir >= 0 else '-'

    return f"{lat_sign}{abs(lat_dir):02d}{lon_sign}{abs(lon_dir):03d}"

print("\nProcessing files:")
target_path = Path(test_dir)
dem_files = list(target_path.glob("*_DSM.tif"))

for dem_file in dem_files:
    match = re.search(r'[NS](\d+)[EW](\d+)', dem_file.name)
    if match:
        lat_num = int(match.group(1))
        lon_num = int(match.group(2))
        lat_prefix = match.group(0)[0]
        lon_prefix = 'E' if 'E' in match.group(0) else 'W'

        lat = f"{lat_prefix}{lat_num:02d}"
        lon = f"{lon_prefix}{lon_num:03d}"

        tile_dir = get_tile_directory(lat_num, lon_num, lat_prefix, lon_prefix)
        tile_path = target_path / tile_dir

        print(f"  {dem_file.name} -> {tile_dir}/")

        # Create directory
        tile_path.mkdir(exist_ok=True)

        # Move file
        new_name = f"{lat}{lon}.tif"
        new_path = tile_path / new_name
        dem_file.rename(new_path)
        print(f"    Created directory and moved to: {new_path.relative_to(target_path)}")

print("\nFinal directory structure:")
for item in sorted(target_path.rglob("*")):
    if item.is_dir():
        print(f"  [DIR]  {item.relative_to(target_path)}")
    else:
        print(f"  [FILE] {item.relative_to(target_path)}")

# Cleanup
print(f"\nCleaning up test directory: {test_dir}")
shutil.rmtree(test_dir)
print("✓ Test completed successfully!")

