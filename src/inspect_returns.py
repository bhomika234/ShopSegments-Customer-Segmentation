
import pandas as pd
from pathlib import Path

# Define input path
input_path = Path("data/online_retail_II.csv")

# Load only required columns to reduce memory usage
columns = ["Invoice", "Quantity", "Price"]

df = pd.read_csv(
    input_path,
    encoding="ISO-8859-1",
    usecols=columns
)

# Identify cancelled invoices (invoice numbers starting with C)
invoice_text = df["Invoice"].astype("string").str.strip()
cancelled_mask = invoice_text.str.startswith("C", na=False)

# Identify negative quantities and negative prices
negative_quantity_mask = df["Quantity"] < 0
negative_price_mask = df["Price"] < 0

# Display inspection results
print("Total transaction rows:", len(df))
print("Cancelled invoice rows:", cancelled_mask.sum())
print("Unique cancelled invoices:", invoice_text[cancelled_mask].nunique())
print("Negative quantity rows:", negative_quantity_mask.sum())
print("Negative price rows:", negative_price_mask.sum())

print("\nSample cancelled invoice rows:")
print(df.loc[cancelled_mask].head(10).to_string(index=False))

print("\nSample negative quantity rows:")
print(df.loc[negative_quantity_mask].head(10).to_string(index=False))
