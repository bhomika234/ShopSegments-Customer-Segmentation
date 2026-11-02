
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load scaled RFM features
file_path = Path("data/customer_rfm_scaled.csv")
df = pd.read_csv(file_path)

# Test different numbers of clusters
X = df[["Recency", "Frequency", "Monetary"]]
k_values = range(2, 9)

results = []

for k in k_values:
    # Train K-Means
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X)

    # Evaluate clustering quality
    score = silhouette_score(X, labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": score
    })

# Display evaluation results
results_df = pd.DataFrame(results)

print("K-Means evaluation results:")
print(results_df.round(4).to_string(index=False))

# Save results for later comparison
output_path = Path("data/kmeans_evaluation.csv")
results_df.to_csv(output_path, index=False)

print("\nEvaluation saved to:", output_path)
