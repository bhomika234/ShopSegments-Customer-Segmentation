
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load scaled customer features
df = pd.read_csv("data/customer_rfm_scaled.csv")
X = df[["Recency", "Frequency", "Monetary"]]

# Evaluate different numbers of Gaussian components
results = []

for k in range(2, 9):
    model = GaussianMixture(
        n_components=k,
        random_state=42,
        n_init=3,
        reg_covar=1e-5
    )

    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    results.append({
        "components": k,
        "silhouette_score": score,
        "AIC": model.aic(X),
        "BIC": model.bic(X)
    })

# Display and save evaluation results
results_df = pd.DataFrame(results)
print("GMM evaluation results:")
print(results_df.round(4).to_string(index=False))

output_path = Path("data/gmm_evaluation.csv")
results_df.to_csv(output_path, index=False)
print("\nEvaluation saved to:", output_path)
