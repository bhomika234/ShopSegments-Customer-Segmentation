
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load scaled customer features
df = pd.read_csv("data/customer_rfm_scaled.csv")
X = df[["Recency", "Frequency", "Monetary"]]

# Test a small range of DBSCAN parameters
results = []

for eps in [0.3, 0.5, 0.7, 1.0]:
    for min_samples in [5, 10]:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        # Exclude noise points when calculating the silhouette score
        non_noise = labels != -1
        cluster_labels = labels[non_noise]
        unique_clusters = set(cluster_labels)

        if len(unique_clusters) >= 2 and non_noise.sum() > len(unique_clusters):
            score = silhouette_score(
                X[non_noise],
                cluster_labels
            )
        else:
            score = None

        results.append({
            "eps": eps,
            "min_samples": min_samples,
            "clusters_found": len(unique_clusters),
            "noise_points": int((labels == -1).sum()),
            "silhouette_score": score
        })

# Display and save results
results_df = pd.DataFrame(results)
print("DBSCAN evaluation results:")
print(results_df.round(4).to_string(index=False))

output_path = Path("data/dbscan_evaluation.csv")
results_df.to_csv(output_path, index=False)
print("\nEvaluation saved to:", output_path)
