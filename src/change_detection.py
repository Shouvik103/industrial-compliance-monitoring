"""
Change Detection module.
Bitemporal differencing of NDVI/NDBI to detect vegetation loss
and new construction between T1 (2020) and T2 (2024).
"""

import os
import numpy as np
import rasterio
import json


def load_index(index_path):
    with rasterio.open(index_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def compute_change_map(t1_path, t2_path):
    t1, profile = load_index(t1_path)
    t2, _ = load_index(t2_path)
    min_h = min(t1.shape[0], t2.shape[0])
    min_w = min(t1.shape[1], t2.shape[1])
    t1 = t1[:min_h, :min_w]
    t2 = t2[:min_h, :min_w]
    diff = t2 - t1
    valid_mask = ~(np.isnan(t1) | np.isnan(t2))
    diff = np.where(valid_mask, diff, np.nan)
    return diff, t1, t2, profile, valid_mask


def detect_vegetation_loss(ndvi_diff, threshold=-0.15):
    return ((ndvi_diff < threshold) & ~np.isnan(ndvi_diff)).astype(np.uint8)


def detect_new_construction(ndbi_diff, threshold=0.10):
    return ((ndbi_diff > threshold) & ~np.isnan(ndbi_diff)).astype(np.uint8)


def classify_land_cover(ndvi, ndbi):
    """0=Other, 1=DenseVeg, 2=SparseVeg, 3=BuiltUp, 4=BareSoil, 5=Water"""
    c = np.zeros_like(ndvi, dtype=np.uint8)
    valid = ~(np.isnan(ndvi) | np.isnan(ndbi))
    c = np.where(valid & (ndvi < 0.0), 5, c)
    c = np.where(valid & (ndvi >= 0.0) & (ndvi < 0.1) & (ndbi < 0.0), 4, c)
    c = np.where(valid & (ndbi > 0.0) & (ndvi < 0.2), 3, c)
    c = np.where(valid & (ndvi >= 0.2) & (ndvi <= 0.4), 2, c)
    c = np.where(valid & (ndvi > 0.4), 1, c)
    return c


def compute_change_statistics(veg_loss, new_construction, ndvi_t1, ndvi_t2, valid_mask):
    total_valid = valid_mask.sum()
    if total_valid == 0:
        return {}
    veg_t1 = ((ndvi_t1 > 0.2) & valid_mask).sum()
    veg_t2 = ((ndvi_t2 > 0.2) & valid_mask).sum()
    gc_t1 = (veg_t1 / total_valid) * 100
    gc_t2 = (veg_t2 / total_valid) * 100
    gc_change = gc_t2 - gc_t1
    vl_pct = (veg_loss.sum() / total_valid) * 100
    nc_pct = (new_construction.sum() / total_valid) * 100

    if gc_change < -30:
        risk = "HIGH"
    elif gc_change < -15:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "total_valid_pixels": int(total_valid),
        "green_cover_t1_pct": round(gc_t1, 2),
        "green_cover_t2_pct": round(gc_t2, 2),
        "green_cover_change_pct": round(gc_change, 2),
        "vegetation_loss_pixels": int(veg_loss.sum()),
        "vegetation_loss_pct": round(vl_pct, 2),
        "new_construction_pixels": int(new_construction.sum()),
        "new_construction_pct": round(nc_pct, 2),
        "risk_level": risk,
    }


def run_change_detection(base_dir, zone):
    processed = os.path.join(base_dir, "data", "processed")
    t1_dir = os.path.join(processed, zone, "T1_2020_indices")
    t2_dir = os.path.join(processed, zone, "T2_2024_indices")
    out_dir = os.path.join(processed, zone, "change_detection")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  --- Change Detection: {zone} ---")

    ndvi_diff, ndvi_t1, ndvi_t2, profile, valid = compute_change_map(
        os.path.join(t1_dir, "NDVI.tif"), os.path.join(t2_dir, "NDVI.tif"))
    ndbi_diff, ndbi_t1, ndbi_t2, _, _ = compute_change_map(
        os.path.join(t1_dir, "NDBI.tif"), os.path.join(t2_dir, "NDBI.tif"))

    h = min(ndvi_diff.shape[0], ndbi_diff.shape[0])
    w = min(ndvi_diff.shape[1], ndbi_diff.shape[1])
    ndvi_diff, ndbi_diff = ndvi_diff[:h,:w], ndbi_diff[:h,:w]
    ndvi_t1, ndvi_t2 = ndvi_t1[:h,:w], ndvi_t2[:h,:w]
    ndbi_t1, ndbi_t2 = ndbi_t1[:h,:w], ndbi_t2[:h,:w]
    valid = valid[:h,:w]

    veg_loss = detect_vegetation_loss(ndvi_diff)
    new_const = detect_new_construction(ndbi_diff)
    lc_t1 = classify_land_cover(ndvi_t1, ndbi_t1)
    lc_t2 = classify_land_cover(ndvi_t2, ndbi_t2)

    change_mask = np.zeros_like(veg_loss, dtype=np.uint8)
    change_mask = np.where(veg_loss > 0, 1, change_mask)
    change_mask = np.where(new_const > 0, 2, change_mask)
    change_mask = np.where((veg_loss > 0) & (new_const > 0), 3, change_mask)

    stats = compute_change_statistics(veg_loss, new_const, ndvi_t1, ndvi_t2, valid)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    fp = profile.copy()
    fp.update(count=1, dtype="float32", height=h, width=w)
    up = fp.copy()
    up.update(dtype="uint8")

    for name, data, prof in [
        ("ndvi_diff", ndvi_diff, fp), ("ndbi_diff", ndbi_diff, fp),
        ("veg_loss", veg_loss, up), ("new_construction", new_const, up),
        ("change_mask", change_mask, up),
        ("lc_t1", lc_t1, up), ("lc_t2", lc_t2, up),
    ]:
        with rasterio.open(os.path.join(out_dir, f"{name}.tif"), "w", **prof) as dst:
            dst.write(data, 1)

    with open(os.path.join(out_dir, "change_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def run_all_change_detection(base_dir):
    results = {}
    for zone in ["peenya", "whitefield"]:
        try:
            results[zone] = run_change_detection(base_dir, zone)
        except Exception as e:
            print(f"  [FAIL] {zone}: {e}")
            import traceback; traceback.print_exc()
    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  CHANGE DETECTION PIPELINE")
    print("=" * 60)
    stats = run_all_change_detection(base_dir)
    print("\n  SUMMARY")
    for z, s in stats.items():
        print(f"  {z}: Risk={s['risk_level']}, Green change={s['green_cover_change_pct']:.1f}%")
