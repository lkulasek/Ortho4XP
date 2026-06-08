import os
import math
import requests
import numpy as np
from PIL import Image
import rasterio

# ==================== CONFIGURATION ====================
AIRPORT_NAME = "EPAR_bircza"
LATITUDE =  49.658317 # Replace with your airport's latitude
LONGITUDE = 22.514223  # Replace with your airport's longitude
ZOOM = 17  # Zoom level (16 = medium, 17 = high, 18 = very high res)
BOX_RADIUS = 4  # How many tiles to grab in each direction from the center


# =======================================================

def lat_lon_to_tile_coords(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile


def tile_coords_to_lat_lon(x, y, zoom):
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def download_and_create_geotiff():
    print(f"Calculating boundaries for {AIRPORT_NAME}...")
    center_x, center_y = lat_lon_to_tile_coords(LATITUDE, LONGITUDE, ZOOM)

    start_x, end_x = center_x - BOX_RADIUS, center_x + BOX_RADIUS
    start_y, end_y = center_y - BOX_RADIUS, center_y + BOX_RADIUS

    total_width = (end_x - start_x + 1) * 256
    total_height = (end_y - start_y + 1) * 256

    # 1. Download raw Mercator tiles
    stitched_image = Image.new('RGB', (total_width, total_height))
    url_template = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    print(f"Downloading {(end_x - start_x + 1) * (end_y - start_y + 1)} grid tiles...")
    for y_idx, y in enumerate(range(start_y, end_y + 1)):
        for x_idx, x in enumerate(range(start_x, end_x + 1)):
            url = url_template.format(x=x, y=y, z=ZOOM)
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code == 200:
                    from io import BytesIO
                    tile_img = Image.open(BytesIO(res.content))
                    stitched_image.paste(tile_img, (x_idx * 256, y_idx * 256))
            except Exception as e:
                print(f"Error fetching tile {x}, {y}: {e}")

    # 2. Get exact edge coordinates
    top_left_lat, top_left_lon = tile_coords_to_lat_lon(start_x, start_y, ZOOM)
    bottom_right_lat, bottom_right_lon = tile_coords_to_lat_lon(end_x + 1, end_y + 1, ZOOM)

    # 3. Un-warp Mercator projection to a flat WGS84 linear grid
    print("Reprojecting imagery to standard WGS84 projection...")
    unwarped_image = Image.new('RGB', (total_width, total_height))
    for new_y in range(total_height):
        frac = new_y / total_height
        current_lat = top_left_lat + frac * (bottom_right_lat - top_left_lat)

        lat_rad = math.radians(current_lat)
        merc_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2.0 ** ZOOM)
        src_y = int((merc_y - start_y) * 256)
        src_y = max(0, min(src_y, stitched_image.height - 1))

        row = stitched_image.crop((0, src_y, total_width, src_y + 1))
        unwarped_image.paste(row, (0, new_y))

    # 4. Convert format for rasterio (expects bands first: channels, height, width)
    img_array = np.array(unwarped_image)
    img_array = np.moveaxis(img_array, -1, 0)

    # 5. Build the GeoTIFF template container
    tiff_filename = f"{AIRPORT_NAME}.tif"
    print(f"Embedding spatial metadata into {tiff_filename}...")

    transform = rasterio.transform.from_bounds(
        top_left_lon,  # West bound
        bottom_right_lat,  # South bound
        bottom_right_lon,  # East bound
        top_left_lat,  # North bound
        total_width,
        total_height
    )

    with rasterio.open(
            tiff_filename, 'w',
            driver='GTiff',
            height=total_height,
            width=total_width,
            count=3,
            dtype=img_array.dtype,
            crs='EPSG:4326',  # True WGS84 global coordinate map used by X-Plane
            transform=transform
    ) as dst:
        dst.write(img_array)

    print(f"\nSuccess! Built a true georeferenced GeoTIFF: {os.path.abspath(tiff_filename)}")


if __name__ == "__main__":
    download_and_create_geotiff()