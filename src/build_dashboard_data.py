"""
Build small, fast-loading files for the Streamlit dashboard.

Why: cleaned_retail.csv is ~39 MB. The live app should not load it, so we
pre-compute everything the dashboard needs once and save it to app_data/.

Outputs (in app_data/):
  customers.csv         RFM + segment + UMAP x/y for every customer
  segments.csv          one row per segment (size, avg spend, action, top products)
  segment_products.csv  top products per segment
  products.csv          StockCode -> Description, popularity
  item_similarity.csv   "customers who bought this also bought" (top 10 / item)
  als_recs.csv          personalised top-10 per customer (from ALS)
  popular.csv           cold-start fallback (most popular products)
  metrics.csv           Precision@10 etc. for the metrics table
  purchases.csv         what each customer already bought (to show history)

Run:  python src/build_dashboard_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "app_data"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ load
seg = pd.read_csv(DATA / "customer_segments_named.csv")
log_scaled = pd.read_csv(DATA / "customer_rfm_log_scaled.csv")  # same row order as customer_rfm
tx = pd.read_csv(
    DATA / "cleaned_retail.csv",
    usecols=["Invoice", "StockCode", "Description", "Quantity", "Customer ID", "TotalAmount", "InvoiceDate"],
)
tx = tx[tx["Quantity"] > 0].dropna(subset=["Customer ID"])
tx["StockCode"] = tx["StockCode"].astype(str)
tx["Customer ID"] = tx["Customer ID"].astype(int)
seg["Customer ID"] = seg["Customer ID"].astype(int)

# ------------------------------------------------------------------ 1. UMAP map
print("UMAP ...")
coords = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=42).fit_transform(log_scaled.values)
seg["UMAP_1"], seg["UMAP_2"] = coords[:, 0], coords[:, 1]
seg.to_csv(OUT / "customers.csv", index=False)

# ------------------------------------------------------------------ 2. products
desc = (
    tx.groupby(["StockCode", "Description"]).size().reset_index(name="n")
    .sort_values("n", ascending=False).drop_duplicates("StockCode")[["StockCode", "Description"]]
)
pop = tx.groupby("StockCode").agg(
    Units=("Quantity", "sum"), Revenue=("TotalAmount", "sum"), Buyers=("Customer ID", "nunique")
).reset_index()
# Rank by number of distinct buyers (not units) so one giant bulk order cannot look "popular"
products = desc.merge(pop, on="StockCode").sort_values(["Buyers", "Units"], ascending=False)
products["Description"] = products["Description"].str.strip().str.title()
products.to_csv(OUT / "products.csv", index=False)

# ------------------------------------------------------------------ 3. segment cards
seg_name = seg.set_index("Customer ID")["Segment_Name"]
tx["Segment"] = tx["Customer ID"].map(seg_name)
sp = (
    tx.groupby(["Segment", "StockCode"])
    .agg(Units=("Quantity", "sum"), Revenue=("TotalAmount", "sum"))
    .reset_index()
    .sort_values(["Segment", "Units"], ascending=[True, False])
    .groupby("Segment").head(5)
    .merge(desc, on="StockCode")
)
sp["Description"] = sp["Description"].str.strip().str.title()
sp.to_csv(OUT / "segment_products.csv", index=False)

summary = seg.groupby("Segment_Name").agg(
    Customers=("Customer ID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum"),
    Marketing_Action=("Marketing_Action", "first"),
).reset_index()
summary.to_csv(OUT / "segments.csv", index=False)

# ------------------------------------------------------------------ 4. item-item similarity
# Binary customer x item matrix on the 2,000 most popular items (keeps the file small)
print("Item similarity ...")
top_items = products.head(2000)["StockCode"].tolist()
t = tx[tx["StockCode"].isin(top_items)][["Customer ID", "StockCode"]].drop_duplicates()
cust_idx = {c: i for i, c in enumerate(t["Customer ID"].unique())}
item_idx = {s: i for i, s in enumerate(top_items)}
M = csr_matrix(
    (np.ones(len(t)), (t["Customer ID"].map(cust_idx), t["StockCode"].map(item_idx))),
    shape=(len(cust_idx), len(item_idx)),
)
Mn = normalize(M.T.tocsr())            # items x customers, L2-normalised -> cosine
sim = (Mn @ Mn.T).toarray()
np.fill_diagonal(sim, 0)
rows = []
for i, code in enumerate(top_items):
    for r, j in enumerate(np.argsort(-sim[i])[:10], 1):
        if sim[i, j] > 0:
            rows.append((code, top_items[j], round(float(sim[i, j]), 4), r))
pd.DataFrame(rows, columns=["StockCode", "Also_Bought", "Similarity", "Rank"]).to_csv(
    OUT / "item_similarity.csv", index=False
)

# ------------------------------------------------------------------ 5. recs & metrics
als = pd.read_csv(DATA / "als_recommendations.csv")
als["Customer ID"] = als["Customer ID"].astype(int)
als["Recommended StockCode"] = als["Recommended StockCode"].astype(str)
als[["Customer ID", "Recommended StockCode", "Rank"]].to_csv(OUT / "als_recs.csv", index=False)

popular = products.head(10)[["StockCode"]].copy()
popular["Rank"] = range(1, 11)
popular.to_csv(OUT / "popular.csv", index=False)

pd.read_csv(DATA / "recommender_evaluation.csv").to_csv(OUT / "metrics.csv", index=False)

hist = (
    tx.groupby(["Customer ID", "StockCode"]).agg(Units=("Quantity", "sum")).reset_index()
    .sort_values(["Customer ID", "Units"], ascending=[True, False])
    .groupby("Customer ID").head(15)
)
hist.to_csv(OUT / "purchases.csv", index=False)

# monthly revenue per segment (for the trend chart)
tx["Month"] = pd.to_datetime(tx["InvoiceDate"]).dt.to_period("M").astype(str)
tx.groupby(["Month", "Segment"])["TotalAmount"].sum().reset_index().to_csv(OUT / "monthly_revenue.csv", index=False)

print("Done ->", OUT)
