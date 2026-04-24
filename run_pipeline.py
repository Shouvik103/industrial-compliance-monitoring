"""
Main pipeline runner. Executes all stages in sequence:
1. Preprocessing (band stacking, cloud masking)
2. Index computation (NDVI, NDBI, NBI)
3. Change detection (bitemporal differencing)
4. Visualization (maps, charts)
"""
import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocess import preprocess_all
from src.indices import compute_indices_all
from src.change_detection import run_all_change_detection
from src.visualize import generate_all_visualizations


def main():
    start = time.time()
    print("=" * 60)
    print("  INDUSTRIAL COMPLIANCE MONITORING PIPELINE")
    print("  ML Course Project - Statement 3")
    print("=" * 60)

    # Stage 1
    print("\n\n>>> STAGE 1: PREPROCESSING")
    print("=" * 60)
    preprocess_all(BASE_DIR)

    # Stage 2
    print("\n\n>>> STAGE 2: SPECTRAL INDEX COMPUTATION")
    print("=" * 60)
    compute_indices_all(BASE_DIR)

    # Stage 3
    print("\n\n>>> STAGE 3: CHANGE DETECTION")
    print("=" * 60)
    stats = run_all_change_detection(BASE_DIR)

    # Stage 4
    print("\n\n>>> STAGE 4: VISUALIZATION")
    print("=" * 60)
    generate_all_visualizations(BASE_DIR)

    elapsed = time.time() - start
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60)
    print("\n  COMPLIANCE SUMMARY:")
    for zone, s in stats.items():
        print(f"    {zone.upper()}: Risk={s['risk_level']}, "
              f"Green cover: {s['green_cover_t1_pct']:.1f}% -> {s['green_cover_t2_pct']:.1f}% "
              f"(change: {s['green_cover_change_pct']:+.1f}%)")
    print("\n  Output: frontend/assets/")
    print("  Next: Run 'python app.py' to launch the dashboard")


if __name__ == "__main__":
    main()
