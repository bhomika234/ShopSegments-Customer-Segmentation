
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

# Load original RFM data and scaled features
rfm = pd.read_csv("data/customer_rfm.csv")
scaled = pd.read_csv("data/customer_rfm_scaled.csv")

features = ["Recency", "Frequency", "Monetary"]
X = scaled[features]

# Compare selected cluster counts
for k in [2, 4, 5]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    rfm[f"Cluster_k{k}"] = model.fit_predict(X)

    # Summarize customer behaviour in each cluster
    profile = rfm.groupby(f"Cluster_k{k}").agg(
        Customers=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    ).round(2)

    print(f"\n--- Customer Profiles: k={k} ---")
    print(profile.to_string())

# Save profiles with cluster assignments
output_path = Path("data/kmeans_cluster_profiles.csv")
rfm.to_csv(output_path, index=False)

print("\nCluster assignments saved to:", output_path)
