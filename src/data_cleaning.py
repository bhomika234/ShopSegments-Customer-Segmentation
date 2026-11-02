
import pandas as pd
from pathlib import Path

# Define file paths
input_path = Path("data/online_retail_II.csv")
output_path = Path("data/cleaned_retail.csv")

df = pd.read_csv(
    input_path,
    encoding="ISO-8859-1",

    
)
print("Original dataset shape:", df.shape)

# 1. Remove duplicate rows
df = df.drop_duplicates().copy()


# 2. Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%m/%d/%y %H:%M",
    errors="coerce"
)

print("Invalid dates:", df["InvoiceDate"].isna().sum())

print("Invalid dates:", df["InvoiceDate"].isna().sum())


# 3. Remove rows with invalid dates
df = df.dropna(subset=["InvoiceDate"])

# 4. Remove rows with missing Customer ID
# Required for customer-level RFM analysis
df = df.dropna(subset=["Customer ID"]).copy()

# 5. Remove rows with missing descriptions
df = df.dropna(subset=["Description"]).copy()

# 6. Remove rows with zero or negative prices
df = df[df["Price"] > 0].copy()

# 7. Identify negative quantities
# These may represent returns or cancellations
negative_quantity = df[df["Quantity"] < 0]

print("\nNegative quantity rows:", len(negative_quantity))

# 8. Keep only positive quantities for the initial
# purchase-based RFM analysis
# Returns are counted separately for later review
df = df[df["Quantity"] > 0].copy()

# 9. Create total transaction value
df["TotalAmount"] = df["Quantity"] * df["Price"]

# Save cleaned data
df.to_csv(output_path, index=False)

# Display cleaning results
print("\nCleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nCleaned data saved to:", output_path)
print("\nFirst 5 cleaned rows:")
print(df.head())
