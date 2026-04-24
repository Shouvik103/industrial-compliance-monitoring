"""
Spectral Index Computation module.
Computes NDVI, NDBI, and NBI from stacked Sentinel-2 bands.
"""

import os
import numpy as np
import rasterio
import json


def safe_divide(a, b, fill=0.0):
    """Division with zero-handling."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result.astype(np.float32)


def compute_ndvi(nir, red):
    """
    Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    Range: [-1, 1], higher = more vegetation
    """
    return safe_divide(nir - red, nir + red)


def compute_ndbi(swir1, nir):
    """
    Normalized Difference Built-up Index.
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Range: [-1, 1], higher = more built-up area
    """
    return safe_divide(swir1 - nir, swir1 + nir)


def compute_nbi(red, swir1, nir):
    """
    New Built-up Index (bare/new construction detection).
    NBI = (Red * SWIR1) / NIR
    Higher values indicate bare soil or new construction.
    """
    return safe_divide(red * swir1, nir)


def compute_all_indices(stacked_path, output_dir):
    """
    Compute all spectral indices from a stacked GeoTIFF.
    
    Input band order: B02(0), B03(1), B04(2), B08(3), B11(4), B12(5), CloudMask(6)
    """
    with rasterio.open(stacked_path) as src:
        data = src.read()  # shape: (bands, height, width)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    
    # Extract bands (0-indexed)
    red = data[2]    # B04
    nir = data[3]    # B08
    swir1 = data[4]  # B11
    cloud_mask = data[6] if data.shape[0] > 6 else np.ones_like(red)
    
    # Compute indices
    ndvi = compute_ndvi(nir, red)
    ndbi = compute_ndbi(swir1, nir)
    nbi = compute_nbi(red, swir1, nir)
    
    # Apply cloud mask (set cloudy pixels to NaN)
    ndvi = np.where(cloud_mask > 0.5, ndvi, np.nan)
    ndbi = np.where(cloud_mask > 0.5, ndbi, np.nan)
    nbi = np.where(cloud_mask > 0.5, nbi, np.nan)
    
    # Save each index as separate GeoTIFF
    os.makedirs(output_dir, exist_ok=True)
    
    index_profile = profile.copy()
    index_profile.update(count=1, dtype="float32")
    
    indices = {"NDVI": ndvi, "NDBI": ndbi, "NBI": nbi}
    results = {}
    
    for name, idx_data in indices.items():
        out_path = os.path.join(output_dir, f"{name}.tif")
        with rasterio.open(out_path, "w", **index_profile) as dst:
            dst.write(idx_data, 1)
            dst.descriptions = (name,)
        
        # Compute stats (ignoring NaN)
        valid = idx_data[~np.isnan(idx_data)]
        stats = {
            "min": float(np.min(valid)) if len(valid) > 0 else 0,
            "max": float(np.max(valid)) if len(valid) > 0 else 0,
            "mean": float(np.mean(valid)) if len(valid) > 0 else 0,
            "std": float(np.std(valid)) if len(valid) > 0 else 0,
        }
        results[name] = {"path": out_path, "stats": stats}
        print(f"    {name}: mean={stats['mean']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    # Save stats as JSON
    stats_path = os.path.join(output_dir, "index_stats.json")
    stats_out = {k: v["stats"] for k, v in results.items()}
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    
    return results


def compute_indices_all(base_dir):
    """Compute indices for all zones and periods."""
    processed_base = os.path.join(base_dir, "data", "processed")
    zones = ["peenya", "whitefield"]
    periods = ["T1_2020", "T2_2024"]
    
    all_results = {}
    
    for zone in zones:
        for period in periods:
            stacked_path = os.path.join(processed_base, zone, f"{period}_stacked.tif")
            if not os.path.exists(stacked_path):
                print(f"  [SKIP] {zone}/{period} - no stacked data")
                continue
            
            print(f"\n  Computing indices for {zone}/{period}...")
            output_dir = os.path.join(processed_base, zone, f"{period}_indices")
            results = compute_all_indices(stacked_path, output_dir)
            all_results[f"{zone}/{period}"] = results
            print(f"  [OK] {zone}/{period} indices computed")
    
    return all_results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  SPECTRAL INDEX COMPUTATION")
    print("=" * 60)
    results = compute_indices_all(base_dir)
    print(f"\n  Computed indices for {len(results)} datasets")
