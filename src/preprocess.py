"""
Preprocessing module: Cloud masking, band stacking, resampling.
Reads raw Sentinel-2 bands and produces analysis-ready stacked GeoTIFFs.
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform


def load_band(band_path):
    """Load a single band GeoTIFF and return data + metadata."""
    with rasterio.open(band_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return data, profile, transform, crs, bounds


def create_cloud_mask(scl_path):
    """
    Create a cloud/shadow mask from Scene Classification Layer (SCL).
    
    SCL values:
      0 = No data
      1 = Saturated/defective
      2 = Dark area pixels
      3 = Cloud shadows
      4 = Vegetation
      5 = Not vegetated
      6 = Water
      7 = Unclassified
      8 = Cloud medium probability
      9 = Cloud high probability
      10 = Thin cirrus
      11 = Snow/ice
    
    We mask out: 0, 1, 3, 8, 9, 10 (no data, defective, shadows, clouds)
    """
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
        scl_profile = src.profile.copy()
    
    # Valid pixels: vegetation(4), not-veg(5), water(6), unclassified(7)
    # Also allow dark area (2) and snow (11) as valid
    valid_classes = {2, 4, 5, 6, 7, 11}
    mask = np.isin(scl, list(valid_classes)).astype(np.uint8)
    
    return mask, scl, scl_profile


def resample_to_10m(data_20m, profile_20m, target_shape, target_transform):
    """
    Resample 20m bands (B11, B12) to 10m resolution to match B02-B08.
    Uses bilinear interpolation.
    """
    # Create output array
    resampled = np.zeros(target_shape, dtype=np.float32)
    
    # Simple nearest-neighbor upsampling (2x in each dimension)
    # This is sufficient for our index computation
    from scipy.ndimage import zoom
    zoom_factor_y = target_shape[0] / data_20m.shape[0]
    zoom_factor_x = target_shape[1] / data_20m.shape[1]
    resampled = zoom(data_20m, (zoom_factor_y, zoom_factor_x), order=1)
    
    return resampled.astype(np.float32)


def stack_bands(raw_dir, output_path):
    """
    Stack all bands into a single multi-band GeoTIFF.
    Resamples 20m bands to 10m.
    Applies cloud mask.
    
    Output band order: B02, B03, B04, B08, B11, B12, CloudMask
    """
    bands_10m = ["B02", "B03", "B04", "B08"]
    bands_20m = ["B11", "B12"]
    
    # Load 10m bands to get reference shape/transform
    ref_data, ref_profile, ref_transform, ref_crs, ref_bounds = load_band(
        os.path.join(raw_dir, "B02.tif")
    )
    target_shape = ref_data.shape
    print(f"    Reference shape (10m): {target_shape}")
    
    # Load and stack 10m bands
    stacked = []
    for band_name in bands_10m:
        band_path = os.path.join(raw_dir, f"{band_name}.tif")
        data, _, _, _, _ = load_band(band_path)
        stacked.append(data)
        print(f"    Loaded {band_name}: shape={data.shape}, "
              f"min={data.min():.0f}, max={data.max():.0f}")
    
    # Load and resample 20m bands
    for band_name in bands_20m:
        band_path = os.path.join(raw_dir, f"{band_name}.tif")
        data, profile, _, _, _ = load_band(band_path)
        resampled = resample_to_10m(data, profile, target_shape, ref_transform)
        stacked.append(resampled)
        print(f"    Loaded {band_name}: {data.shape} -> resampled to {resampled.shape}")
    
    # Load cloud mask
    scl_path = os.path.join(raw_dir, "SCL.tif")
    if os.path.exists(scl_path):
        cloud_mask, _, _ = create_cloud_mask(scl_path)
        # Resample SCL mask to 10m (it's 20m native)
        cloud_mask = resample_to_10m(
            cloud_mask.astype(np.float32), None, target_shape, ref_transform
        )
        cloud_mask = (cloud_mask > 0.5).astype(np.float32)
        stacked.append(cloud_mask)
        valid_pct = (cloud_mask.sum() / cloud_mask.size) * 100
        print(f"    Cloud mask: {valid_pct:.1f}% valid pixels")
    
    # Stack into array: (n_bands, height, width)
    stacked_array = np.array(stacked, dtype=np.float32)
    
    # Write output
    out_profile = ref_profile.copy()
    out_profile.update(
        count=stacked_array.shape[0],
        dtype="float32",
        driver="GTiff",
        compress="deflate",
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        for i in range(stacked_array.shape[0]):
            dst.write(stacked_array[i], i + 1)
        # Write band descriptions
        band_names = bands_10m + bands_20m + ["CloudMask"]
        dst.descriptions = tuple(band_names)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    Stacked output: {output_path} ({file_size:.1f} MB)")
    print(f"    Shape: {stacked_array.shape}")
    
    return stacked_array, out_profile


def preprocess_all(base_dir):
    """Run preprocessing for all zones and time periods."""
    raw_base = os.path.join(base_dir, "data", "raw")
    processed_base = os.path.join(base_dir, "data", "processed")
    
    zones = ["peenya", "whitefield"]
    periods = ["T1_2020", "T2_2024"]
    
    results = {}
    
    for zone in zones:
        for period in periods:
            raw_dir = os.path.join(raw_base, zone, period)
            if not os.path.exists(raw_dir):
                print(f"  [SKIP] {zone}/{period} - no raw data")
                continue
            
            print(f"\n  Processing {zone}/{period}...")
            output_path = os.path.join(processed_base, zone, f"{period}_stacked.tif")
            
            try:
                stacked, profile = stack_bands(raw_dir, output_path)
                results[f"{zone}/{period}"] = {
                    "path": output_path,
                    "shape": stacked.shape,
                    "profile": profile,
                }
                print(f"  [OK] {zone}/{period} preprocessed")
            except Exception as e:
                print(f"  [FAIL] {zone}/{period}: {e}")
                import traceback
                traceback.print_exc()
    
    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)
    results = preprocess_all(base_dir)
    print(f"\n  Processed {len(results)} datasets")
