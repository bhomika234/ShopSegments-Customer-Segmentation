"""
Export the per-method segment data used by the dashboard's "Segmentation method" picker.

Reads the single-file dashboard (ShopSegments_Dashboard.html, which embeds the K-Means, log K-Means, GMM
and DBSCAN results) and writes two small files the Streamlit app loads:
  app_data/methods.json         per method: description, silhouette, insights, segment cards, monthly revenue
  app_data/method_labels.csv    Customer ID + segment index for each method

Run:  python src/export_methods_data.py
"""
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
html = (BASE / "ShopSegments_Dashboard.html").read_text(encoding="utf-8")
D = json.loads(re.search(r"const D=(\{.*?\});\nconst e=React", html, re.S).group(1))

labels = pd.DataFrame({"Customer ID": [p[3] for p in D["pts"]]})
methods = []
for m in D["methods"]:
    labels[m["id"]] = m["lab"]
    methods.append({k: m[k] for k in ("id", "label", "desc", "sil", "ins")} | {
        "seg": [{k: s[k] for k in ("n", "short", "col", "cust", "rec", "freq", "mon", "rev", "act", "top")} for s in m["seg"]],
        "trend": m["trend"],
    })

labels.to_csv(BASE / "app_data" / "method_labels.csv", index=False)
(BASE / "app_data" / "methods.json").write_text(
    json.dumps({"months": D["months"], "methods": methods}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("Saved app_data/methods.json and app_data/method_labels.csv")
