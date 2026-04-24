"""
Flask backend serving real ML pipeline results.
Endpoints return actual computed data from the pipeline.
"""
import os, json, glob
from flask import Flask, jsonify, send_from_directory, send_file
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@app.route("/")
def index():
    return send_file(os.path.join(BASE, "frontend", "index.html"))

@app.route("/api/zones")
def get_zones():
    zones = []
    for zone in ["peenya", "whitefield"]:
        cd = os.path.join(BASE, "data", "processed", zone, "change_detection")
        stats = load_json(os.path.join(cd, "change_stats.json"))
        violations = load_json(os.path.join(cd, "violations.json")) if os.path.exists(os.path.join(cd, "violations.json")) else []
        metrics = load_json(os.path.join(BASE, "models", f"rf_{zone}_metrics.json"))
        t1_idx = load_json(os.path.join(BASE, "data", "processed", zone, "T1_2020_indices", "index_stats.json"))
        t2_idx = load_json(os.path.join(BASE, "data", "processed", zone, "T2_2024_indices", "index_stats.json"))

        bbox = {"peenya": [13.01, 77.48, 13.07, 77.56], "whitefield": [12.95, 77.72, 13.01, 77.80]}
        center = {"peenya": [13.04, 77.52], "whitefield": [12.98, 77.76]}

        zones.append({
            "name": zone,
            "label": zone.title() + " Industrial Area",
            "center": center[zone],
            "bbox": bbox[zone],
            "change_stats": stats,
            "ml_metrics": {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "n_train": metrics.get("n_train", 0),
                "n_test": metrics.get("n_test", 0),
                "feature_importance": metrics.get("feature_importance", {}),
                "confusion_matrix": metrics.get("confusion_matrix", []),
            },
            "index_stats": {"T1_2020": t1_idx, "T2_2024": t2_idx},
            "violations": violations if isinstance(violations, list) else [],
        })
    return jsonify(zones)

@app.route("/api/images/<zone>")
def get_images(zone):
    imgs = {}
    for folder, prefix in [("results", ""), ("frontend/assets/" + zone, "asset_")]:
        d = os.path.join(BASE, folder)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".png") and (zone in f or prefix == "asset_"):
                key = prefix + os.path.splitext(f)[0]
                imgs[key] = f"/img/{folder}/{f}"
    return jsonify(imgs)

@app.route("/img/<path:filepath>")
def serve_image(filepath):
    full = os.path.join(BASE, filepath)
    return send_file(full, mimetype="image/png")

if __name__ == "__main__":
    print(f"  Dashboard: http://localhost:5000")
    app.run(debug=False, port=5000)
