
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "cleaned_retail.csv"
OUTPUT_FILE = DATA_DIR / "item_based_recommendations.csv"

# Load required columns only
df = pd.read_csv(
    INPUT_FILE,
    usecols=["Customer ID", "StockCode", "Quantity"]
)

# Remove missing values
df = df.dropna(subset=["Customer ID", "StockCode", "Quantity"])

# Aggregate customer purchases by product
df = df.groupby(
    ["Customer ID", "StockCode"]
)["Quantity"].sum().reset_index()

# Keep only positive purchase quantities
df = df[df["Quantity"] > 0]

# Create customer-item interaction matrix
interaction_matrix = df.pivot(
    index="Customer ID",
    columns="StockCode",
    values="Quantity"
).fillna(0)

print("Customer-item matrix shape:", interaction_matrix.shape)

# Convert matrix to sparse format
sparse_matrix = csr_matrix(interaction_matrix.values)

# Calculate item-item cosine similarity
item_similarity = cosine_similarity(
    sparse_matrix.T,
    dense_output=False
)

# Generate recommendations for each customer
recommendations = []

customer_ids = interaction_matrix.index
product_codes = interaction_matrix.columns

for customer_index, customer_id in enumerate(customer_ids):

    customer_vector = interaction_matrix.iloc[customer_index].values

    # Find products already purchased
    purchased_items = customer_vector > 0

    # Calculate recommendation scores
    scores = item_similarity.dot(customer_vector)

    # Exclude already purchased products
    scores[purchased_items] = 0

    # Select top 10 recommended products
    top_indices = scores.argsort()[::-1][:10]

    for rank, item_index in enumerate(top_indices, start=1):

        if scores[item_index] > 0:
            recommendations.append({
                "Customer ID": customer_id,
                "Recommended StockCode": product_codes[item_index],
                "Score": scores[item_index],
                "Rank": rank
            })

# Save recommendations
recommendations_df = pd.DataFrame(recommendations)

recommendations_df.to_csv(OUTPUT_FILE, index=False)

print("\nItem-Based Collaborative Filtering completed!")
print("Recommendations saved to:", OUTPUT_FILE)
print("\nSample recommendations:")
print(recommendations_df.head(10))
