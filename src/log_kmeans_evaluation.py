
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load log-transformed and scaled RFM features
df = pd.read_csv("data/customer_rfm_log_scaled.csv")
X = df[["Recency", "Frequency", "Monetary"]]

results = []

# Evaluate K-Means for different cluster counts
for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": score
    })

# Display and save evaluation results
results_df = pd.DataFrame(results)
print("Log-transformed K-Means evaluation:")
print(results_df.round(4).to_string(index=False))

output_path = Path("data/log_kmeans_evaluation.csv")
results_df.to_csv(output_path, index=False)
print("\nEvaluation saved to:", output_path)
