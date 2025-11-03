
import os
import zipfile
import shutil
import re
from pathlib import Path

source_dir = "C:/Users/lukas/Downloads/alosw30"
target_dir = "C:/Users/lukas/Downloads/_unpack"

def get_tile_directory(lat_num, lon_num, lat_prefix, lon_prefix):
    """Calculate tile directory name in format +XX+XXX or -XX-XXX."""
    # Apply sign based on N/S and E/W first
    if lat_prefix == 'S':
        lat_num = -lat_num
    if lon_prefix == 'W':
        lon_num = -lon_num

    # Round down to nearest 10 for directory grouping (floor division)
    # For negative numbers, we want to round towards more negative
    lat_dir = (lat_num // 10) * 10
    lon_dir = (lon_num // 10) * 10

    # Format as +XX+XXX or -XX-XXX
    lat_sign = '+' if lat_dir >= 0 else '-'
    lon_sign = '+' if lon_dir >= 0 else '-'

    return f"{lat_sign}{abs(lat_dir):02d}{lon_sign}{abs(lon_dir):03d}"

def rename_dem_files(dem_files):
    """Rename DEM files from ALPSMLC30_N054E026_DSM.tif to N54E26_ALOS3W30.tif format and organize into tile directories."""

    target_path = Path(target_dir)
    renamed_files = []

    print(f"\nRenaming DEM files and organizing into tile directories...")

    for dem_file in dem_files:
        # Extract coordinates using regex
        # Pattern: ALPSMLC30_N054E026_DSM.tif -> N054E026
        match = re.search(r'[NS](\d+)[EW](\d+)', dem_file.name)

        if match:
            # Get latitude and longitude with proper padding
            # LAT: 2 digits, LON: 3 digits
            lat_num = int(match.group(1))
            lon_num = int(match.group(2))
            lat_prefix = match.group(0)[0]  # N or S
            lon_prefix = 'E' if 'E' in match.group(0) else 'W'

            lat = f"{lat_prefix}{lat_num:02d}"  # N + 2 digits
            lon = f"{lon_prefix}{lon_num:03d}"  # E/W + 3 digits

            # Get tile directory name
            tile_dir = get_tile_directory(lat_num, lon_num, lat_prefix, lon_prefix)
            tile_path = target_path / tile_dir

            # Create tile directory if it doesn't exist
            tile_path.mkdir(exist_ok=True)

            # Create new filename
            new_name = f"{lat}{lon}_ALOS3W30.tif"
            new_path = tile_path / new_name

            # Rename and move file
            if dem_file != new_path:
                # Check if target file already exists
                if new_path.exists():
                    print(f"  Warning: {new_name} already exists in {tile_dir}, replacing...")
                    new_path.unlink()  # Delete existing file

                dem_file.rename(new_path)
                print(f"  {dem_file.name} -> {tile_dir}/{new_name}")
                renamed_files.append(new_path)
            else:
                renamed_files.append(dem_file)
        else:
            print(f"  Warning: Could not extract coordinates from {dem_file.name}, keeping original name")
            renamed_files.append(dem_file)

    return renamed_files

def cleanup_and_organize_dem_files():
    """Recursively find all *_DSM.tif files, organize them into tile directories, delete everything else."""

    target_path = Path(target_dir)

    # Recursively find all *_DSM.tif files
    dem_files = list(target_path.rglob("*_DSM.tif"))

    if not dem_files:
        print(f"No *_DSM.tif files found in {target_dir}")
        return

    print(f"\nFound {len(dem_files)} *_DSM.tif file(s):")
    for dem_file in dem_files:
        print(f"  - {dem_file.name}")

    # Move all DEM files to root temporarily for processing
    print(f"\nPreparing files for organization...")
    temp_files = []
    for dem_file in dem_files:
        if dem_file.parent != target_path:
            new_path = target_path / dem_file.name
            # Handle name conflicts
            counter = 1
            while new_path.exists():
                new_path = target_path / f"{dem_file.stem}_{counter}{dem_file.suffix}"
                counter += 1
            shutil.move(str(dem_file), str(new_path))
            temp_files.append(new_path)
        else:
            temp_files.append(dem_file)

    # Rename and organize files into tile directories
    organized_files = rename_dem_files(temp_files)

    # Delete all files that are not organized DEM files
    print(f"\nDeleting non-DEM files...")
    deleted_files = 0
    for item in target_path.rglob("*"):
        if item.is_file() and item not in organized_files:
            item.unlink()
            deleted_files += 1

    # Delete empty directories (but keep tile directories)
    print(f"Removing empty extraction directories...")
    deleted_dirs = 0
    for item in sorted(target_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if item.is_dir() and item.parent == target_path:
            # Skip tile directories (format: +XX+XXX or -XX-XXX)
            if re.match(r'^[+-]\d{2}[+-]\d{3}$', item.name):
                continue
        if item.is_dir():
            try:
                item.rmdir()
                deleted_dirs += 1
            except OSError:
                pass  # Directory not empty, skip

    print(f"\n✓ Cleanup complete!")
    print(f"  - Organized {len(organized_files)} DEM file(s) into tile directories")
    print(f"  - Deleted {deleted_files} non-DEM file(s)")
    print(f"  - Removed {deleted_dirs} empty director(ies)")

def list_and_unpack_zip_files():
    """List all zip files in source_dir and unpack them into target_dir."""

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get all zip files in source directory
    source_path = Path(source_dir)
    zip_files = list(source_path.glob("*.zip"))

    if not zip_files:
        print(f"No zip files found in {source_dir}")
        return

    print(f"Found {len(zip_files)} zip file(s):")
    for zip_file in zip_files:
        print(f"  - {zip_file.name}")

    print(f"\nUnpacking to {target_dir}...")

    # Unpack each zip file
    for zip_file in zip_files:
        try:
            print(f"\nExtracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"  ✓ Successfully extracted {zip_file.name}")
        except Exception as e:
            print(f"  ✗ Error extracting {zip_file.name}: {e}")

    print(f"\nDone! All files extracted to {target_dir}")

    # Clean up and organize *_DSM.tif files into tile directories
    cleanup_and_organize_dem_files()

if __name__ == "__main__":
    list_and_unpack_zip_files()
