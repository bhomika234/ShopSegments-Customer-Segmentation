
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the RFM dataset
df = pd.read_csv("data/customer_rfm.csv")

# Create output folder for graphs
Path("data/plots").mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Recency distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Recency"], bins=30, kde=True)
plt.title("Customer Recency Distribution")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/plots/recency.png")
plt.show()

# 2. Frequency distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Frequency"], bins=30)
plt.title("Customer Purchase Frequency")
plt.xlabel("Number of Unique Invoices")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/plots/frequency.png")
plt.show()

# 3. Monetary distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Monetary"], bins=40)
plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/plots/monetary.png")
plt.show()

# 4. Correlation between RFM features
plt.figure(figsize=(6, 5))
sns.heatmap(
    df[["Recency", "Frequency", "Monetary"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("RFM Feature Correlation")
plt.tight_layout()
plt.savefig("data/plots/rfm_correlation.png")
plt.show()

print("EDA completed. Graphs saved in data/plots/")
