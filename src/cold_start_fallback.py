import pandas as pd
from pathlib import Path

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "cleaned_retail.csv"
CF_FILE = DATA_DIR / "item_based_recommendations.csv"
ALS_FILE = DATA_DIR / "als_recommendations.csv"

OUTPUT_FILE = DATA_DIR / "cold_start_recommendations.csv"

# Load purchase data
df = pd.read_csv(
    INPUT_FILE,
    usecols=["Customer ID", "StockCode", "Quantity"]
)

# Remove invalid records
df = df.dropna()
df = df[df["Quantity"] > 0]

# Calculate product popularity
popularity = (
    df.groupby("StockCode")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

popular_items = popularity.index.tolist()

# Load existing recommendations
cf = pd.read_csv(CF_FILE)
als = pd.read_csv(ALS_FILE)

# Select ALS recommendations as the primary model
recommendations = als.copy()

# Get existing customer purchase history
customer_purchases = (
    df.groupby("Customer ID")["StockCode"]
    .apply(set)
    .to_dict()
)

# Get all customers
customers = df["Customer ID"].unique()

final_rows = []

# Generate recommendations for every customer
for customer_id in customers:

    # Get existing recommendations
    customer_recs = recommendations[
        recommendations["Customer ID"] == customer_id
    ]

    recommended_items = customer_recs[
        "Recommended StockCode"
    ].tolist()

    # Get customer's purchased products
    purchased_items = customer_purchases.get(
        customer_id, set()
    )

    # Remove already purchased products
    final_items = [
        item for item in recommended_items
        if item not in purchased_items
    ]

    # Fill missing recommendations using popular products
    for item in popular_items:

        if len(final_items) >= 10:
            break

        if item not in purchased_items and item not in final_items:
            final_items.append(item)

    # Save top 10 recommendations
    for rank, item in enumerate(final_items[:10], start=1):

        if item in recommended_items:
            source = "ALS"
        else:
            source = "Popularity Fallback"

        final_rows.append({
            "Customer ID": customer_id,
            "Recommended StockCode": item,
            "Rank": rank,
            "Recommendation Source": source
        })

# Create recommendations for a new customer
new_customer_id = "NEW_CUSTOMER"

new_customer_items = popular_items[:10]

for rank, item in enumerate(new_customer_items, start=1):

    final_rows.append({
        "Customer ID": new_customer_id,
        "Recommended StockCode": item,
        "Rank": rank,
        "Recommendation Source": "Popularity Fallback"
    })

# Save final recommendations
final_df = pd.DataFrame(final_rows)

final_df.to_csv(OUTPUT_FILE, index=False)

print("Cold-start fallback completed!")

print("\nSample recommendations:")
print(final_df.head(15).to_string(index=False))

print("\nNew customer recommendations:")
print(
    final_df[
        final_df["Customer ID"] == new_customer_id
    ].to_string(index=False)
)

print("\nOutput saved successfully!")
print("File:", OUTPUT_FILE)