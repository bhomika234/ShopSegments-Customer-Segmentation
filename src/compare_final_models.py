
import pandas as pd
from sklearn.metrics import silhouette_score

# Load original RFM data and both feature sets
rfm = pd.read_csv("data/customer_rfm.csv")
original = pd.read_csv("data/customer_rfm_scaled.csv")
log_scaled = pd.read_csv("data/customer_rfm_log_scaled.csv")

# Load saved cluster assignments
original_segments = pd.read_csv("data/customer_segments.csv")
log_segments = pd.read_csv("data/log_customer_segments.csv")

# Calculate silhouette scores for k=4
original_score = silhouette_score(
    original[["Recency", "Frequency", "Monetary"]],
    original_segments["Cluster"]
)

log_score = silhouette_score(
    log_scaled[["Recency", "Frequency", "Monetary"]],
    log_segments["Log_Cluster"]
)

print("Final K-Means comparison (k=4)")
print(f"Original RFM silhouette: {original_score:.4f}")
print(f"Log-transformed silhouette: {log_score:.4f}")

# Build a combined summary using original RFM values
original_profile = rfm[["Customer ID", "Recency", "Frequency", "Monetary"]].copy()
original_profile["Cluster"] = original_segments["Cluster"]
original_profile["Model"] = "Original RFM"

log_profile = rfm[["Customer ID", "Recency", "Frequency", "Monetary"]].copy()
log_profile["Cluster"] = log_segments["Log_Cluster"]
log_profile["Model"] = "Log-transformed RFM"

combined = pd.concat([original_profile, log_profile], ignore_index=True)

summary = combined.groupby(["Model", "Cluster"]).agg(
    Customers=("Customer ID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
).round(2)

print("\nCluster profile comparison:")
print(summary.to_string())

summary.to_csv("data/final_model_comparison.csv")
print("\nSaved to data/final_model_comparison.csv")
