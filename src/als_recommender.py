
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "cleaned_retail.csv"
OUTPUT_FILE = DATA_DIR / "als_recommendations.csv"

# Load purchase data
df = pd.read_csv(
    INPUT_FILE,
    usecols=["Customer ID", "StockCode", "Quantity"]
)

# Clean data
df = df.dropna()
df = df[df["Quantity"] > 0]

# Aggregate purchases
df = df.groupby(
    ["Customer ID", "StockCode"]
)["Quantity"].sum().reset_index()

# Create customer and item mappings
customer_ids = df["Customer ID"].unique()
item_ids = df["StockCode"].unique()

customer_map = {
    customer_id: index
    for index, customer_id in enumerate(customer_ids)
}

item_map = {
    item_id: index
    for index, item_id in enumerate(item_ids)
}

# Convert IDs into matrix indices
df["customer_index"] = df["Customer ID"].map(customer_map)
df["item_index"] = df["StockCode"].map(item_map)

# Build sparse customer-item matrix
user_item_matrix = csr_matrix(
    (
        df["Quantity"].astype(float),
        (df["customer_index"], df["item_index"])
    ),
    shape=(len(customer_ids), len(item_ids))
)

# Train ALS model
model = AlternatingLeastSquares(
    factors=50,
    regularization=0.1,
    iterations=20,
    random_state=42
)

# Train using item-user matrix
model.fit(user_item_matrix)

print("ALS model training completed!")

# Generate top 10 recommendations for each customer
recommendations = []

for user_index, customer_id in enumerate(customer_ids):

    user_items = user_item_matrix[user_index]

    item_indices, item_scores = model.recommend(
        userid=user_index,
        user_items=user_items,
        N=10,
        filter_already_liked_items=True
    )

    for rank, (item_index, score) in enumerate(
        zip(item_indices, item_scores), start=1
    ):
        recommendations.append({
            "Customer ID": customer_id,
            "Recommended StockCode": item_ids[item_index],
            "Score": float(score),
            "Rank": rank
        })

# Save recommendations
recommendations_df = pd.DataFrame(recommendations)

recommendations_df.to_csv(OUTPUT_FILE, index=False)

print("\nALS recommendations saved successfully!")
print("Output:", OUTPUT_FILE)

print("\nSample recommendations:")
print(recommendations_df.head(10))
