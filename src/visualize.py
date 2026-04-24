"""
Visualization module. Generates PNG maps for the frontend and reports.
"""
import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json


def normalize_band(data, pmin=2, pmax=98):
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return np.zeros_like(data)
    lo = np.percentile(valid, pmin)
    hi = np.percentile(valid, pmax)
    if hi == lo:
        return np.zeros_like(data)
    out = (data - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def make_rgb(stacked_path):
    """Create RGB composite from stacked bands (B04=Red, B03=Green, B02=Blue)."""
    with rasterio.open(stacked_path) as src:
        data = src.read()
    red = normalize_band(data[2])  # B04
    green = normalize_band(data[1])  # B03
    blue = normalize_band(data[0])  # B02
    rgb = np.stack([red, green, blue], axis=-1)
    return rgb


def make_false_color(stacked_path):
    """NIR-Red-Green false color for vegetation highlighting."""
    with rasterio.open(stacked_path) as src:
        data = src.read()
    nir = normalize_band(data[3])
    red = normalize_band(data[2])
    green = normalize_band(data[1])
    return np.stack([nir, red, green], axis=-1)


def plot_index_map(index_path, title, output_path, cmap="RdYlGn", vmin=-1, vmax=1):
    with rasterio.open(index_path) as src:
        data = src.read(1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=title.split("(")[0].strip())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_change_map(change_path, title, output_path):
    with rasterio.open(change_path) as src:
        data = src.read(1)
    colors = ["#2d2d2d", "#ff4444", "#ff8800", "#ff0088"]
    labels = ["No Change", "Vegetation Loss", "New Construction", "Both"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_landcover(lc_path, title, output_path):
    with rasterio.open(lc_path) as src:
        data = src.read(1)
    colors = ["#333333", "#228B22", "#90EE90", "#CD853F", "#DEB887", "#4169E1"]
    labels = ["Other", "Dense Veg", "Sparse Veg", "Built-up", "Bare Soil", "Water"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0,1,2,3,4,5])
    cbar.ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_comparison(stacked_t1, stacked_t2, zone, output_path):
    """Side-by-side RGB comparison."""
    rgb1 = make_rgb(stacked_t1)
    rgb2 = make_rgb(stacked_t2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(rgb1)
    ax1.set_title(f"{zone.title()} - 2020 (Baseline)", fontsize=14, fontweight="bold")
    ax1.axis("off")
    ax2.imshow(rgb2)
    ax2.set_title(f"{zone.title()} - 2024 (Recent)", fontsize=14, fontweight="bold")
    ax2.axis("off")
    plt.suptitle(f"Satellite Imagery Comparison - {zone.title()}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_dashboard(stats, zone, output_path):
    """Compliance dashboard summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Green cover bar chart
    ax = axes[0]
    bars = ax.bar(["2020", "2024"],
                  [stats["green_cover_t1_pct"], stats["green_cover_t2_pct"]],
                  color=["#228B22", "#ff6b35"], width=0.5)
    ax.set_ylabel("Green Cover %")
    ax.set_title("Green Cover Change", fontweight="bold")
    ax.set_ylim(0, 100)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.1f}%",
                ha="center", fontweight="bold")

    # Risk gauge
    ax = axes[1]
    risk = stats["risk_level"]
    risk_colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color=risk_colors.get(risk, "#666"),
                             transform=ax.transAxes))
    ax.text(0.5, 0.5, risk, transform=ax.transAxes, fontsize=28, fontweight="bold",
            ha="center", va="center", color="white")
    ax.text(0.5, 0.15, "Risk Level", transform=ax.transAxes, fontsize=14,
            ha="center", va="center", fontweight="bold")
    ax.axis("off")

    # Change summary
    ax = axes[2]
    ax.axis("off")
    text = (
        f"Zone: {zone.title()}\n\n"
        f"Vegetation Loss: {stats['vegetation_loss_pct']:.1f}%\n"
        f"New Construction: {stats['new_construction_pct']:.1f}%\n"
        f"Green Cover Change: {stats['green_cover_change_pct']:.1f}%\n\n"
        f"Total Analyzed Pixels: {stats['total_valid_pixels']:,}"
    )
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=14,
            fontfamily="monospace", verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#ccc"))

    plt.suptitle(f"Compliance Dashboard - {zone.title()}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_all_visualizations(base_dir):
    processed = os.path.join(base_dir, "data", "processed")
    viz_dir = os.path.join(base_dir, "frontend", "assets")
    os.makedirs(viz_dir, exist_ok=True)

    for zone in ["peenya", "whitefield"]:
        print(f"\n  Generating visuals for {zone}...")
        zone_viz = os.path.join(viz_dir, zone)
        os.makedirs(zone_viz, exist_ok=True)

        t1_stacked = os.path.join(processed, zone, "T1_2020_stacked.tif")
        t2_stacked = os.path.join(processed, zone, "T2_2024_stacked.tif")
        t1_idx = os.path.join(processed, zone, "T1_2020_indices")
        t2_idx = os.path.join(processed, zone, "T2_2024_indices")
        cd_dir = os.path.join(processed, zone, "change_detection")

        if os.path.exists(t1_stacked) and os.path.exists(t2_stacked):
            plot_comparison(t1_stacked, t2_stacked, zone,
                           os.path.join(zone_viz, "rgb_comparison.png"))

        for period, idx_dir in [("T1_2020", t1_idx), ("T2_2024", t2_idx)]:
            ndvi_path = os.path.join(idx_dir, "NDVI.tif")
            if os.path.exists(ndvi_path):
                plot_index_map(ndvi_path, f"NDVI - {zone.title()} ({period})",
                              os.path.join(zone_viz, f"ndvi_{period}.png"))

        if os.path.exists(cd_dir):
            cm = os.path.join(cd_dir, "change_mask.tif")
            if os.path.exists(cm):
                plot_change_map(cm, f"Change Detection - {zone.title()}",
                               os.path.join(zone_viz, "change_map.png"))
            for p, label in [("lc_t1", "2020"), ("lc_t2", "2024")]:
                lc = os.path.join(cd_dir, f"{p}.tif")
                if os.path.exists(lc):
                    plot_landcover(lc, f"Land Cover {label} - {zone.title()}",
                                  os.path.join(zone_viz, f"landcover_{label}.png"))

            stats_path = os.path.join(cd_dir, "change_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)
                plot_dashboard(stats, zone, os.path.join(zone_viz, "dashboard.png"))

        print(f"  [OK] {zone} visualizations saved")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)
    generate_all_visualizations(base_dir)
