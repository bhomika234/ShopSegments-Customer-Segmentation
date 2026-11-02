
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "cleaned_retail.csv"
OUTPUT_FILE = DATA_DIR / "recommender_evaluation.csv"

# Load purchase data
df = pd.read_csv(
    INPUT_FILE,
    usecols=["Invoice", "InvoiceDate", "Customer ID", "StockCode", "Quantity"]
)

# Clean data
df = df.dropna()
df = df[df["Quantity"] > 0]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Find each customer's latest invoice
latest_invoices = (
    df.groupby("Customer ID")["InvoiceDate"]
    .max()
    .reset_index()
)

# Separate training and testing data
test = df.merge(
    latest_invoices,
    on=["Customer ID", "InvoiceDate"]
)

train = df[
    ~df.set_index(["Customer ID", "InvoiceDate"]).index.isin(
        test.set_index(["Customer ID", "InvoiceDate"]).index
    )
].copy()

# Aggregate customer-item purchases
train = train.groupby(
    ["Customer ID", "StockCode"]
)["Quantity"].sum().reset_index()

test = test.groupby(
    ["Customer ID", "StockCode"]
)["Quantity"].sum().reset_index()

# Create mappings using training data
customer_ids = train["Customer ID"].unique()
item_ids = train["StockCode"].unique()

customer_map = {
    customer: index
    for index, customer in enumerate(customer_ids)
}

item_map = {
    item: index
    for index, item in enumerate(item_ids)
}

# Keep only test items known to the training data
test = test[
    test["Customer ID"].isin(customer_map)
    & test["StockCode"].isin(item_map)
]

# Build training interaction matrix
train["user_index"] = train["Customer ID"].map(customer_map)
train["item_index"] = train["StockCode"].map(item_map)

user_item_matrix = csr_matrix(
    (
        train["Quantity"].astype(float),
        (train["user_index"], train["item_index"])
    ),
    shape=(len(customer_ids), len(item_ids))
)

# Prepare actual test purchases
actual_items = {}

for customer_id, group in test.groupby("Customer ID"):
    actual_items[customer_map[customer_id]] = set(
        item_map[item]
        for item in group["StockCode"]
    )

# Calculate Item-Based CF similarity
item_similarity = cosine_similarity(
    user_item_matrix.T,
    dense_output=False
)

# Train ALS model
als_model = AlternatingLeastSquares(
    factors=50,
    regularization=0.1,
    iterations=20,
    random_state=42
)
als_model.fit(user_item_matrix)

print("Models trained for evaluation!")

# Evaluate Precision@10
item_cf_scores = []
als_scores = []

for user_index, actual in actual_items.items():

    user_vector = user_item_matrix[user_index]

    # Item-Based CF recommendations
    scores = item_similarity.dot(user_vector.toarray().ravel())
    scores[user_vector.indices] = 0

    top_cf = np.argsort(scores)[::-1][:10]
    hits_cf = len(set(top_cf) & actual)

    item_cf_scores.append(hits_cf / 10)

    # ALS recommendations
    recommended_indices, recommended_scores = als_model.recommend(
        userid=user_index,
        user_items=user_vector,
        N=10,
        filter_already_liked_items=True
    )

    hits_als = len(set(recommended_indices) & actual)

    als_scores.append(hits_als / 10)

# Calculate average Precision@10
item_cf_precision = (
    np.mean(item_cf_scores) if item_cf_scores else 0
)

als_precision = (
    np.mean(als_scores) if als_scores else 0
)

# Save evaluation results
results = pd.DataFrame({
    "Model": [
        "Item-Based Collaborative Filtering",
        "ALS Recommender"
    ],
    "Precision@10": [
        item_cf_precision,
        als_precision
    ]
})

results.to_csv(OUTPUT_FILE, index=False)

print("\nPrecision@10 Evaluation Results:")
print(results.to_string(index=False))

print("\nEvaluation saved successfully!")
print("Output:", OUTPUT_FILE)
print("Evaluated customers:", len(actual_items))
