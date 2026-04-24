"""
=============================================================================
INDUSTRIAL COMPLIANCE MONITORING - DATA DOWNLOAD SCRIPT
=============================================================================
Downloads Sentinel-2 L2A imagery + industrial boundary polygons for the
ML course project (Statement 3).

What this downloads:
  - Sentinel-2 bands (B02, B03, B04, B08, B11, B12, SCL) for Peenya
    Industrial Area, Bengaluru - two time periods (2020 baseline, 2024 recent)
  - Industrial boundary polygon from OpenStreetMap Overpass API

Total download size: ~300-500 MB

Usage:
  pip install -r requirements.txt
  python download_data.py
=============================================================================
"""

import os
import sys
import json
import requests
import warnings
import numpy as np

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# -- Configuration ----------------------------------------------------------

# Peenya Industrial Area, Bengaluru - bounding box [west, south, east, north]
PEENYA_BBOX = [77.48, 13.01, 77.56, 13.07]

# Additional zone: Whitefield / ITPL area (has visible construction growth)
WHITEFIELD_BBOX = [77.72, 12.95, 77.80, 13.01]

# Time periods
T1_START = "2020-01-01"
T1_END = "2020-03-31"
T2_START = "2024-01-01"
T2_END = "2024-03-31"

# Bands to download (10m and 20m resolution)
BANDS_TO_DOWNLOAD = [
    "B02",  # Blue   (10m)
    "B03",  # Green  (10m)
    "B04",  # Red    (10m)
    "B08",  # NIR    (10m)
    "B11",  # SWIR1  (20m)
    "B12",  # SWIR2  (20m)
    "SCL",  # Scene Classification Layer (cloud mask)
]

MAX_CLOUD_COVER = 15  # percent

# Output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
BOUNDARY_DIR = os.path.join(BASE_DIR, "data", "boundaries")

# -- Step 1: Download Industrial Boundaries from OSM ------------------------

def download_osm_boundaries():
    """Download industrial area boundaries from OpenStreetMap Overpass API."""
    print("\n" + "=" * 60)
    print("STEP 1: Downloading Industrial Boundaries from OSM")
    print("=" * 60)

    overpass_url = "https://overpass-api.de/api/interpreter"

    # Queries using bbox-only (no area filter = simpler, more reliable)
    queries = {
        "peenya": '[out:json][timeout:60];(way["landuse"="industrial"](13.01,77.48,13.07,77.56);relation["landuse"="industrial"](13.01,77.48,13.07,77.56););out geom;',
        "whitefield": '[out:json][timeout:60];(way["landuse"="industrial"](12.95,77.72,13.01,77.80);relation["landuse"="industrial"](12.95,77.72,13.01,77.80););out geom;',
    }

    for name, query in queries.items():
        print(f"\n  Querying OSM for {name} industrial boundaries...")
        try:
            # Use POST to avoid URL-too-long errors
            response = requests.post(overpass_url, data={"data": query}, timeout=120)
            response.raise_for_status()
            osm_data = response.json()

            # Convert OSM response to GeoJSON
            geojson = osm_to_geojson(osm_data)
            n_features = len(geojson["features"])
            print(f"  [OK] Found {n_features} industrial polygons for {name}")

            # Save GeoJSON
            output_path = os.path.join(BOUNDARY_DIR, f"{name}_industrial.geojson")
            with open(output_path, "w") as f:
                json.dump(geojson, f, indent=2)
            print(f"  [OK] Saved to {output_path}")

        except Exception as e:
            print(f"  [FAIL] Error downloading {name}: {e}")
            print(f"  --> Creating fallback boundary from bounding box...")
            create_fallback_boundary(name)

    # Also create a combined AOI boundary for each zone
    for name, bbox in [("peenya", PEENYA_BBOX), ("whitefield", WHITEFIELD_BBOX)]:
        aoi_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": f"{name}_aoi", "type": "study_area"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]]
                }
            }]
        }
        aoi_path = os.path.join(BOUNDARY_DIR, f"{name}_aoi.geojson")
        with open(aoi_path, "w") as f:
            json.dump(aoi_geojson, f, indent=2)
        print(f"  [OK] AOI bounding box saved to {aoi_path}")


def osm_to_geojson(osm_data):
    """Convert Overpass API response to GeoJSON FeatureCollection."""
    features = []
    for element in osm_data.get("elements", []):
        if element["type"] == "way" and "geometry" in element:
            coords = [[pt["lon"], pt["lat"]] for pt in element["geometry"]]
            # Close the polygon if needed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            feature = {
                "type": "Feature",
                "properties": {
                    "osm_id": element.get("id"),
                    "name": element.get("tags", {}).get("name", "unnamed"),
                    "landuse": element.get("tags", {}).get("landuse", "industrial"),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
            features.append(feature)
        elif element["type"] == "relation" and "members" in element:
            # Handle multipolygon relations (simplified)
            outer_coords = []
            for member in element.get("members", []):
                if member.get("role") == "outer" and "geometry" in member:
                    coords = [[pt["lon"], pt["lat"]] for pt in member["geometry"]]
                    outer_coords.extend(coords)
            if outer_coords:
                if outer_coords[0] != outer_coords[-1]:
                    outer_coords.append(outer_coords[0])
                feature = {
                    "type": "Feature",
                    "properties": {
                        "osm_id": element.get("id"),
                        "name": element.get("tags", {}).get("name", "unnamed"),
                        "landuse": "industrial",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [outer_coords],
                    },
                }
                features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def create_fallback_boundary(name):
    """Create a simple rectangular boundary as fallback."""
    bbox = PEENYA_BBOX if name == "peenya" else WHITEFIELD_BBOX
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": f"{name}_industrial_zone", "type": "fallback_bbox"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]]
            }
        }]
    }
    output_path = os.path.join(BOUNDARY_DIR, f"{name}_industrial.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  [OK] Fallback boundary saved to {output_path}")


# -- Step 2: Download Sentinel-2 Imagery ------------------------------------

def download_sentinel2():
    """Download Sentinel-2 L2A bands from Microsoft Planetary Computer."""
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Sentinel-2 L2A Imagery")
    print("=" * 60)

    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        print("  [FAIL] Missing packages. Run: pip install pystac-client planetary-computer")
        return False

    # Connect to Planetary Computer STAC API
    print("\n  Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    print("  [OK] Connected")

    zones = {
        "peenya": PEENYA_BBOX,
        "whitefield": WHITEFIELD_BBOX,
    }

    periods = {
        "T1_2020": (T1_START, T1_END),
        "T2_2024": (T2_START, T2_END),
    }

    for zone_name, bbox in zones.items():
        for period_name, (start, end) in periods.items():
            print(f"\n  -- {zone_name} / {period_name} --")
            print(f"  Searching for Sentinel-2 L2A tiles...")
            print(f"  BBOX: {bbox}")
            print(f"  Date range: {start} to {end}")
            print(f"  Max cloud cover: {MAX_CLOUD_COVER}%")

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start}/{end}",
                query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
                sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
                max_items=5,
            )

            items = list(search.items())
            print(f"  [OK] Found {len(items)} scenes with <{MAX_CLOUD_COVER}% cloud cover")

            if not items:
                print(f"  [FAIL] No scenes found! Try increasing MAX_CLOUD_COVER.")
                continue

            # Pick the best (least cloudy) scene
            best_item = items[0]
            cloud_cover = best_item.properties.get("eo:cloud_cover", "N/A")
            date = best_item.properties.get("datetime", "N/A")[:10]
            print(f"  --> Selected: {best_item.id}")
            print(f"      Date: {date}, Cloud cover: {cloud_cover}%")

            # Create output directory
            out_dir = os.path.join(RAW_DIR, zone_name, period_name)
            os.makedirs(out_dir, exist_ok=True)

            # Save metadata
            meta = {
                "item_id": best_item.id,
                "date": date,
                "cloud_cover": cloud_cover,
                "bbox": bbox,
                "zone": zone_name,
                "period": period_name,
            }
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Download each band
            import rasterio
            from rasterio.windows import from_bounds
            from pyproj import Transformer

            for band_name in BANDS_TO_DOWNLOAD:
                band_key = band_name.lower()
                if band_key not in best_item.assets:
                    # Try alternate key names used by Planetary Computer
                    alt_keys = {
                        "b02": "B02", "b03": "B03", "b04": "B04",
                        "b08": "B08", "b11": "B11", "b12": "B12",
                        "scl": "SCL",
                    }
                    band_key = alt_keys.get(band_key, band_key)

                if band_key not in best_item.assets:
                    # Try yet another naming convention
                    alt_keys2 = {
                        "B02": "blue", "B03": "green", "B04": "red",
                        "B08": "nir", "B11": "swir16", "B12": "swir22",
                    }
                    band_key = alt_keys2.get(band_key, band_key)

                if band_key not in best_item.assets:
                    print(f"    [FAIL] Band {band_name} not found in assets")
                    print(f"           Available: {sorted(best_item.assets.keys())}")
                    continue

                asset = best_item.assets[band_key]
                href = asset.href

                out_path = os.path.join(out_dir, f"{band_name}.tif")
                if os.path.exists(out_path):
                    size = os.path.getsize(out_path) / (1024 * 1024)
                    print(f"    [OK] {band_name} already downloaded ({size:.1f} MB), skipping")
                    continue

                print(f"    Downloading {band_name}...", end=" ", flush=True)
                try:
                    with rasterio.open(href) as src:
                        # Reproject bbox from EPSG:4326 (lat/lon) to tile CRS (UTM)
                        tile_crs = src.crs
                        transformer = Transformer.from_crs(
                            "EPSG:4326", tile_crs, always_xy=True
                        )
                        left, bottom = transformer.transform(bbox[0], bbox[1])
                        right, top = transformer.transform(bbox[2], bbox[3])

                        # Calculate the window using reprojected coordinates
                        window = from_bounds(
                            left, bottom, right, top,
                            transform=src.transform,
                        )

                        # Read only the windowed data (not the full tile!)
                        data = src.read(1, window=window)

                        # Calculate the transform for the windowed data
                        win_transform = src.window_transform(window)

                        # Write the clipped data
                        profile = src.profile.copy()
                        profile.update(
                            width=data.shape[1],
                            height=data.shape[0],
                            transform=win_transform,
                            driver="GTiff",
                            compress="deflate",
                        )

                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(data, 1)

                    file_size = os.path.getsize(out_path) / (1024 * 1024)
                    print(f"[OK] ({file_size:.1f} MB)")

                except Exception as e:
                    print(f"[FAIL] Error: {e}")

    return True


# -- Step 3: Verify Downloads -----------------------------------------------

def verify_downloads():
    """Check that all expected files were downloaded."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying Downloads")
    print("=" * 60)

    total_size = 0
    total_files = 0
    missing = []

    for zone in ["peenya", "whitefield"]:
        for period in ["T1_2020", "T2_2024"]:
            dir_path = os.path.join(RAW_DIR, zone, period)
            print(f"\n  {zone}/{period}:")

            if not os.path.exists(dir_path):
                print(f"    [FAIL] Directory missing!")
                missing.append(f"{zone}/{period}")
                continue

            for band in BANDS_TO_DOWNLOAD:
                fpath = os.path.join(dir_path, f"{band}.tif")
                if os.path.exists(fpath):
                    size = os.path.getsize(fpath) / (1024 * 1024)
                    total_size += size
                    total_files += 1
                    print(f"    [OK] {band}.tif ({size:.1f} MB)")
                else:
                    print(f"    [FAIL] {band}.tif MISSING")
                    missing.append(f"{zone}/{period}/{band}.tif")

    # Check boundaries
    print(f"\n  Boundaries:")
    for name in ["peenya", "whitefield"]:
        for suffix in ["_industrial.geojson", "_aoi.geojson"]:
            fpath = os.path.join(BOUNDARY_DIR, f"{name}{suffix}")
            if os.path.exists(fpath):
                print(f"    [OK] {name}{suffix}")
            else:
                print(f"    [FAIL] {name}{suffix} MISSING")
                missing.append(f"boundaries/{name}{suffix}")

    print(f"\n  -- Summary --")
    print(f"  Total files: {total_files}")
    print(f"  Total size:  {total_size:.1f} MB")
    if missing:
        print(f"  Missing:     {len(missing)} files")
        for m in missing:
            print(f"    - {m}")
    else:
        print(f"  [OK] All files present!")

    return len(missing) == 0


# -- Main -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  INDUSTRIAL COMPLIANCE MONITORING - DATA DOWNLOADER")
    print("  ML Course Project (Statement 3)")
    print("=" * 60)
    print(f"\n  Target zones: Peenya, Whitefield (Bengaluru)")
    print(f"  Time periods: T1 (Jan-Mar 2020), T2 (Jan-Mar 2024)")
    print(f"  Bands: {', '.join(BANDS_TO_DOWNLOAD)}")
    print(f"  Output: {RAW_DIR}")

    # Ensure output dirs exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(BOUNDARY_DIR, exist_ok=True)

    # Step 1: Boundaries
    download_osm_boundaries()

    # Step 2: Sentinel-2
    download_sentinel2()

    # Step 3: Verify
    success = verify_downloads()

    if success:
        print("\n" + "=" * 60)
        print("  [OK] ALL DATA DOWNLOADED SUCCESSFULLY")
        print("  Next step: Run the preprocessing pipeline")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  [WARNING] Some files are missing - check errors above")
        print("=" * 60)
