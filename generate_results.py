"""
Generate ALL result images and compliance report for the project.
- Confusion matrix heatmaps
- Feature importance bar charts
- Learning curves
- Annotated satellite imagery with violation markers
- Before/after comparison with annotations
- Compliance report (HTML)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import rasterio

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- 1. Confusion Matrix Heatmaps ----
def plot_confusion_matrices():
    print("  Generating confusion matrix heatmaps...")
    classes = ["Other", "Dense Veg", "Sparse Veg", "Built-up", "Bare Soil", "Water"]
    for zone in ["peenya", "whitefield"]:
        path = os.path.join(BASE, "models", f"rf_{zone}_metrics.json")
        if not os.path.exists(path): continue
        with open(path) as f: m = json.load(f)
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=classes[:cm.shape[0]], yticklabels=classes[:cm.shape[0]])
        ax.set_xlabel("Predicted", fontsize=13)
        ax.set_ylabel("Actual", fontsize=13)
        ax.set_title(f"Confusion Matrix - {zone.title()}\n"
                     f"Accuracy: {m['accuracy']*100:.1f}% | F1: {m['f1_score']*100:.1f}%",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{zone}.png"), dpi=150)
        plt.close()
    print("  [OK] Confusion matrices saved")

# ---- 2. Feature Importance ----
def plot_feature_importance():
    print("  Generating feature importance charts...")
    for zone in ["peenya", "whitefield"]:
        path = os.path.join(BASE, "models", f"rf_{zone}_metrics.json")
        if not os.path.exists(path): continue
        with open(path) as f: m = json.load(f)
        imp = m["feature_importance"]
        bands = list(imp.keys())
        vals = list(imp.values())
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(bands, vals, color=colors[:len(bands)])
        ax.set_xlabel("Importance", fontsize=13)
        ax.set_title(f"Random Forest Feature Importance - {zone.title()}", fontsize=14, fontweight="bold")
        for b, v in zip(bars, vals):
            ax.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"feature_importance_{zone}.png"), dpi=150)
        plt.close()
    print("  [OK] Feature importance charts saved")

# ---- 3. Learning Curve (RF accuracy vs n_estimators) ----
def plot_learning_curves():
    print("  Generating learning curves...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    sys.path.insert(0, BASE)
    from src.ml_classifier import prepare_features, generate_training_labels

    for zone in ["peenya", "whitefield"]:
        stacked = os.path.join(BASE, "data", "processed", zone, "T1_2020_stacked.tif")
        if not os.path.exists(stacked): continue
        X_all, X_valid_all, mask, h, w, _ = prepare_features(stacked)
        labels = generate_training_labels(X_valid_all)
        X_raw = X_valid_all[:, :6]
        # Subsample for speed
        idx = np.random.RandomState(42).choice(len(X_raw), min(50000, len(X_raw)), replace=False)
        X_sub, y_sub = X_raw[idx], labels[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.3, random_state=42)

        n_trees_list = [5, 10, 20, 50, 100, 150, 200]
        train_accs, test_accs = [], []
        for n in n_trees_list:
            rf = RandomForestClassifier(n_estimators=n, max_depth=15, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            train_accs.append(accuracy_score(y_tr, rf.predict(X_tr)))
            test_accs.append(accuracy_score(y_te, rf.predict(X_te)))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(n_trees_list, train_accs, "o-", label="Training Accuracy", color="#2ecc71", linewidth=2)
        ax.plot(n_trees_list, test_accs, "s-", label="Validation Accuracy", color="#e74c3c", linewidth=2)
        ax.set_xlabel("Number of Trees", fontsize=13)
        ax.set_ylabel("Accuracy", fontsize=13)
        ax.set_title(f"Learning Curve - {zone.title()}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=12)
        ax.set_ylim(0.85, 1.01)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"learning_curve_{zone}.png"), dpi=150)
        plt.close()
    print("  [OK] Learning curves saved")

# ---- 4. Annotated Satellite Imagery with Violation Markers ----
def normalize(data, pmin=2, pmax=98):
    v = data[~np.isnan(data)]
    if len(v) == 0: return np.zeros_like(data)
    lo, hi = np.percentile(v, pmin), np.percentile(v, pmax)
    if hi == lo: return np.zeros_like(data)
    return np.clip((data - lo) / (hi - lo), 0, 1)

def plot_annotated_imagery():
    print("  Generating annotated satellite imagery...")
    for zone in ["peenya", "whitefield"]:
        t2_path = os.path.join(BASE, "data", "processed", zone, "T2_2024_stacked.tif")
        viol_path = os.path.join(BASE, "data", "processed", zone, "change_detection", "violations.json")
        change_path = os.path.join(BASE, "data", "processed", zone, "change_detection", "change_mask.tif")
        if not all(os.path.exists(p) for p in [t2_path, viol_path, change_path]): continue

        # Load RGB
        with rasterio.open(t2_path) as src:
            data = src.read()
            transform = src.transform
            crs = src.crs
        rgb = np.stack([normalize(data[2]), normalize(data[1]), normalize(data[0])], axis=-1)
        # Set KGIS-masked pixels (band 7 = mask) to white
        if data.shape[0] >= 7:
            mask = data[6] < 0.5
            rgb[mask] = 1.0

        # Load change mask
        with rasterio.open(change_path) as src:
            change = src.read(1)
        h, w = min(rgb.shape[0], change.shape[0]), min(rgb.shape[1], change.shape[1])
        rgb, change = rgb[:h, :w], change[:h, :w]

        # Create overlay
        overlay = rgb.copy()
        overlay[change == 1] = [1, 0.2, 0.2]  # Red for veg loss
        overlay[change == 2] = [1, 0.5, 0]    # Orange for new construction
        overlay[change == 3] = [1, 0, 0.5]    # Pink for both
        blended = rgb * 0.5 + overlay * 0.5

        # Load violations
        with open(viol_path) as f: violations = json.load(f)

        # Convert violation lat/lon to pixel coordinates
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(blended)

        for v in violations[:20]:  # Top 20
            mx, my = t.transform(v["lon"], v["lat"])
            px = (mx - transform.c) / transform.a
            py = (my - transform.f) / transform.e
            color = {"vegetation_loss": "#ff4444", "new_construction": "#ff8800", "both": "#ff0088"}
            marker = ax.plot(px, py, "v", color=color.get(v["type"], "white"), markersize=10,
                           markeredgecolor="white", markeredgewidth=1.5)[0]
            ax.annotate(f"{v['area_hectares']}ha", (px, py), textcoords="offset points",
                       xytext=(8, 8), fontsize=8, color="white", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#ff4444", label="Vegetation Loss"),
            Patch(facecolor="#ff8800", label="New Construction"),
            Patch(facecolor="#ff0088", label="Both"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=11,
                 facecolor="black", edgecolor="white", labelcolor="white")
        ax.set_title(f"Violation Detection - {zone.title()} (2024)\nAnnotated Satellite Imagery with Geo-tagged Violations",
                    fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"annotated_violations_{zone}.png"), dpi=150, facecolor="black")
        plt.close()
    print("  [OK] Annotated imagery saved")

# ---- 5. Before/After with Change Overlay ----
def plot_before_after():
    print("  Generating before/after comparisons...")
    for zone in ["peenya", "whitefield"]:
        t1 = os.path.join(BASE, "data", "processed", zone, "T1_2020_stacked.tif")
        t2 = os.path.join(BASE, "data", "processed", zone, "T2_2024_stacked.tif")
        cd = os.path.join(BASE, "data", "processed", zone, "change_detection")
        if not all(os.path.exists(p) for p in [t1, t2, cd]): continue

        with rasterio.open(t1) as s: d1 = s.read()
        with rasterio.open(t2) as s: d2 = s.read()
        rgb1 = np.stack([normalize(d1[2]), normalize(d1[1]), normalize(d1[0])], axis=-1)
        rgb2 = np.stack([normalize(d2[2]), normalize(d2[1]), normalize(d2[0])], axis=-1)
        # Set KGIS-masked pixels to white
        if d1.shape[0] >= 7:
            m1 = d1[6] < 0.5
            rgb1[m1] = 1.0
        if d2.shape[0] >= 7:
            m2 = d2[6] < 0.5
            rgb2[m2] = 1.0

        # NDVI comparison
        ndvi1_p = os.path.join(BASE, "data", "processed", zone, "T1_2020_indices", "NDVI.tif")
        ndvi2_p = os.path.join(BASE, "data", "processed", zone, "T2_2024_indices", "NDVI.tif")
        with rasterio.open(ndvi1_p) as s: ndvi1 = s.read(1)
        with rasterio.open(ndvi2_p) as s: ndvi2 = s.read(1)

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes[0,0].imshow(rgb1); axes[0,0].set_title("RGB - 2020 (Baseline)", fontsize=13, fontweight="bold"); axes[0,0].axis("off")
        axes[0,1].imshow(rgb2); axes[0,1].set_title("RGB - 2024 (Recent)", fontsize=13, fontweight="bold"); axes[0,1].axis("off")
        h = min(ndvi1.shape[0], ndvi2.shape[0]); w = min(ndvi1.shape[1], ndvi2.shape[1])
        im1 = axes[1,0].imshow(ndvi1[:h,:w], cmap="RdYlGn", vmin=-0.2, vmax=0.8); axes[1,0].set_title("NDVI - 2020", fontsize=13, fontweight="bold"); axes[1,0].axis("off")
        plt.colorbar(im1, ax=axes[1,0], fraction=0.046)
        im2 = axes[1,1].imshow(ndvi2[:h,:w], cmap="RdYlGn", vmin=-0.2, vmax=0.8); axes[1,1].set_title("NDVI - 2024", fontsize=13, fontweight="bold"); axes[1,1].axis("off")
        plt.colorbar(im2, ax=axes[1,1], fraction=0.046)

        # Load stats
        sp = os.path.join(cd, "change_stats.json")
        with open(sp) as f: stats = json.load(f)
        fig.suptitle(f"Temporal Change Analysis - {zone.title()}\n"
                    f"Green Cover: {stats['green_cover_t1_pct']:.1f}% -> {stats['green_cover_t2_pct']:.1f}% "
                    f"(Change: {stats['green_cover_change_pct']:+.1f}%) | Risk: {stats['risk_level']}",
                    fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"before_after_{zone}.png"), dpi=150)
        plt.close()
    print("  [OK] Before/after comparisons saved")

# ---- 6. Land Cover Classification Maps ----
def plot_landcover_comparison():
    print("  Generating land cover classification maps...")
    classes = ["Outside KGIS Boundary", "Dense Veg", "Sparse Veg", "Built-up", "Bare Soil", "Water"]
    colors = ["#f5f5f5", "#228B22", "#90EE90", "#CD853F", "#DEB887", "#4169E1"]
    cmap = ListedColormap(colors)

    for zone in ["peenya", "whitefield"]:
        lc1 = os.path.join(BASE, "data", "processed", zone, "change_detection", "lc_t1.tif")
        lc2 = os.path.join(BASE, "data", "processed", zone, "change_detection", "lc_t2.tif")
        if not all(os.path.exists(p) for p in [lc1, lc2]): continue
        with rasterio.open(lc1) as s: d1 = s.read(1)
        with rasterio.open(lc2) as s: d2 = s.read(1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.imshow(d1, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
        ax1.set_title("Land Cover 2020", fontsize=14, fontweight="bold"); ax1.axis("off")
        im = ax2.imshow(d2, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
        ax2.set_title("Land Cover 2024", fontsize=14, fontweight="bold"); ax2.axis("off")
        from matplotlib.patches import Patch
        legend = [Patch(facecolor=c, label=l) for c, l in zip(colors[1:], classes[1:])]
        fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=12)
        fig.suptitle(f"ML-Based Land Cover Classification - {zone.title()}", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(RESULTS_DIR, f"landcover_comparison_{zone}.png"), dpi=150)
        plt.close()
    print("  [OK] Land cover maps saved")

# ---- 7. Compliance Report (HTML) ----
def generate_compliance_report():
    print("  Generating compliance reports...")
    for zone in ["peenya", "whitefield"]:
        stats_p = os.path.join(BASE, "data", "processed", zone, "change_detection", "change_stats.json")
        viol_p = os.path.join(BASE, "data", "processed", zone, "change_detection", "violations.json")
        metrics_p = os.path.join(BASE, "models", f"rf_{zone}_metrics.json")
        if not os.path.exists(stats_p): continue
        with open(stats_p) as f: stats = json.load(f)
        violations = []
        if os.path.exists(viol_p):
            with open(viol_p) as f: violations = json.load(f)
        ml_metrics = {}
        if os.path.exists(metrics_p):
            with open(metrics_p) as f: ml_metrics = json.load(f)

        risk_color = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(stats["risk_level"], "#666")

        viol_rows = ""
        for i, v in enumerate(violations, 1):
            viol_rows += f"<tr><td>{i}</td><td>{v['lat']:.6f}, {v['lon']:.6f}</td><td>{v['type'].replace('_',' ').title()}</td><td>{v['area_hectares']}</td></tr>\n"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Compliance Report - {zone.title()}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; color: #333; }}
h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
h2 {{ color: #1e40af; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #f0f4ff; font-weight: bold; }}
.risk-badge {{ display: inline-block; padding: 8px 24px; border-radius: 8px; color: white; font-size: 24px; font-weight: bold; background: {risk_color}; }}
.metric {{ display: inline-block; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px 25px; margin: 5px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; color: #1e40af; }}
.metric .label {{ font-size: 12px; color: #64748b; }}
</style></head><body>
<h1>Environmental Compliance Report</h1>
<p><strong>Zone:</strong> {zone.title()} Industrial Area, Bengaluru, Karnataka</p>
<p><strong>Analysis Period:</strong> March 2020 (Baseline) to March 2024 (Recent)</p>
<p><strong>Data Source:</strong> Sentinel-2 L2A (ESA Copernicus) | 10m Resolution</p>
<p><strong>Generated:</strong> Automated Pipeline v1.0</p>

<h2>Risk Assessment</h2>
<p>Overall Risk Level: <span class="risk-badge">{stats['risk_level']}</span></p>
<div>
<div class="metric"><div class="value">{stats['green_cover_t1_pct']:.1f}%</div><div class="label">Green Cover 2020</div></div>
<div class="metric"><div class="value">{stats['green_cover_t2_pct']:.1f}%</div><div class="label">Green Cover 2024</div></div>
<div class="metric"><div class="value" style="color:#ef4444">{stats['green_cover_change_pct']:+.1f}%</div><div class="label">Change</div></div>
<div class="metric"><div class="value">{stats['vegetation_loss_pct']:.1f}%</div><div class="label">Vegetation Loss</div></div>
<div class="metric"><div class="value">{stats['new_construction_pct']:.1f}%</div><div class="label">New Construction</div></div>
</div>

<h2>ML Model Performance</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Algorithm</td><td>Random Forest (100 trees, max_depth=15)</td></tr>
<tr><td>Training Samples</td><td>{ml_metrics.get('n_train', 'N/A'):,}</td></tr>
<tr><td>Test Samples</td><td>{ml_metrics.get('n_test', 'N/A'):,}</td></tr>
<tr><td>Accuracy</td><td>{ml_metrics.get('accuracy', 0)*100:.2f}%</td></tr>
<tr><td>Precision (weighted)</td><td>{ml_metrics.get('precision', 0)*100:.2f}%</td></tr>
<tr><td>Recall (weighted)</td><td>{ml_metrics.get('recall', 0)*100:.2f}%</td></tr>
<tr><td>F1-Score (weighted)</td><td>{ml_metrics.get('f1_score', 0)*100:.2f}%</td></tr></table>

<h2>Violation Alerts</h2>
<p>Total violations detected: <strong>{len(violations)}</strong></p>
<table><tr><th>#</th><th>Coordinates (Lat, Lon)</th><th>Type</th><th>Area (ha)</th></tr>
{viol_rows}</table>

<h2>Methodology</h2>
<ol>
<li><strong>Data Acquisition:</strong> KGIS Taluk boundaries (KSRSAC) + Sentinel-2 L2A imagery via STAC API</li>
<li><strong>Preprocessing:</strong> Cloud masking (SCL), KGIS boundary masking, band resampling (20m to 10m)</li>
<li><strong>Spectral Analysis:</strong> NDVI (vegetation), NDBI (built-up), NBI (new construction) index computation</li>
<li><strong>ML Classification:</strong> Random Forest trained on 6 spectral bands, validated with 70/30 split</li>
<li><strong>Change Detection:</strong> Bitemporal differencing of spectral indices with adaptive thresholds</li>
<li><strong>Violation Extraction:</strong> Connected component analysis with geo-coordinate conversion</li>
</ol>

<p style="color:#888; font-size:12px; margin-top:40px; border-top:1px solid #eee; padding-top:10px;">
Industrial Environmental Compliance Monitoring System | ML Course Project | IIIT Dharwad</p>
</body></html>"""

        out = os.path.join(BASE, "reports", f"compliance_report_{zone}.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f: f.write(html)
        print(f"  [OK] Report: {out}")

# ---- RUN ALL ----
if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING ALL RESULTS & VISUALIZATIONS")
    print("=" * 60)
    plot_confusion_matrices()
    plot_feature_importance()
    plot_learning_curves()
    plot_annotated_imagery()
    plot_before_after()
    plot_landcover_comparison()
    generate_compliance_report()
    print("\n" + "=" * 60)
    print(f"  ALL DONE! Check: {RESULTS_DIR}")
    print(f"  Reports: {os.path.join(BASE, 'reports')}")
    print("=" * 60)
