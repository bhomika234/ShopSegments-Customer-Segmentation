
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

# Load original RFM data and scaled features
rfm = pd.read_csv("data/customer_rfm.csv")
scaled = pd.read_csv("data/customer_rfm_scaled.csv")

features = ["Recency", "Frequency", "Monetary"]
X = scaled[features]

# Fit the candidate K-Means model
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = model.fit_predict(X)

# Create a customer profile for each cluster
profile = rfm.groupby("Cluster").agg(
    Customers=("Customer ID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
).round(2)

print("K-Means customer profiles (k=4):")
print(profile.to_string())

# Save customer assignments and cluster profiles
rfm.to_csv("data/customer_segments.csv", index=False)
profile.to_csv("data/customer_segment_summary.csv")

print("\nCustomer segments saved to data/customer_segments.csv")
print("Segment summary saved to data/customer_segment_summary.csv")
