# ShopSegments: Customer Segmentation & Product Recommender

Online stores often send the same promotion to everyone, which wastes money. **ShopSegments** groups customers by behaviour
(RFM) and recommends the right products to each customer, in a dashboard a marketing person can use without technical knowledge.

**Live demo:** _add your Streamlit Cloud / Hugging Face URL here_  
**Demo video:** _add your Loom / YouTube link here_

## Dataset
Online Retail II (UCI): ~541k transactions of a UK online gift shop (Dec 2010 - Dec 2011).
After cleaning: 392,693 purchase rows and **4,338 customers**.

## Approach
1. **Cleaning + RFM**: removed missing customer IDs, bad prices/dates, duplicates and returns. Built Recency, Frequency, Monetary per customer.
2. **Clustering**: StandardScaler (and a log-transform variant), then K-Means, DBSCAN and Gaussian Mixture; k chosen with silhouette, elbow **and** business sense.
3. **Segments named** for marketing: VIP Champions, Loyal High-Value, Regular, At-Risk, each with a suggested action.
4. **2D map**: UMAP projection of log-scaled RFM, drawn with Plotly.
5. **Recommenders**: item-based collaborative filtering and ALS (`implicit`), evaluated with Precision@10 on a hold-out set (each customer's latest order).
6. **Cold start**: new customers get the best-selling products (ranked by number of distinct buyers).
7. **Dashboard**: Streamlit app reading small pre-computed files, so it loads fast.

## Results
| Item | Result |
|---|---|
| Customers | 4,338 |
| K-Means k=4 silhouette | 0.616 (k=2 is 0.896 but too coarse for marketing) |
| Log-RFM K-Means k=4 silhouette | 0.338 |
| DBSCAN best | 2 clusters, silhouette 0.674, 107 noise points |
| GMM best | 3 components, silhouette 0.541 |
| Item-based CF Precision@10 | 1.99% |
| **ALS Precision@10** | **4.17%** |

| Segment | Customers | Avg spend | Avg orders | Last purchase | Action |
|---|---:|---:|---:|---:|---|
| VIP Champions | 13 | £127,188 | 82.5 | 7 days ago | Exclusive rewards, premium offers |
| Loyal High-Value | 204 | £12,691 | 22.3 | 16 days ago | Early access, bundles, loyalty rewards |
| Regular | 3,054 | £1,354 | 3.7 | 44 days ago | Loyalty points, personalised discounts |
| At-Risk | 1,067 | £479 | 1.6 | 248 days ago | Win-back emails, discounts |

## Dashboard features
The Streamlit app (`app/app.py`) and the single-file `ShopSegments_Dashboard.html` have the same layout and content.

**Customer Segments tab**
- KPI cards, **segmentation method picker** (K-Means, log K-Means, Gaussian Mixture, DBSCAN) with description and silhouette score
- Segment filter, interactive customer map with two views (Similarity / Orders x spend), segment profiles, segment sizes, top customers, insights
- Segment cards: size, average spend, share of revenue, top products, suggested marketing action
- Revenue by month and segment, CSV download of the segment list

**Product Recommendations tab**
- "Customers who bought this also bought" for 2,000 popular products
- Personalised top-10 for any customer (with what they already bought), and a best-seller list for a new customer (cold start)

**Model Quality tab**
- Precision@10, elbow and silhouette charts, methods compared, known limitations

The HTML file also has the dataset facts and the data-exploration plots (Recency, Frequency, Monetary, correlation).

## How to run
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
To rebuild from raw data (optional): `pip install -r requirements-dev.txt`, run the scripts in `src/` in order
(`data_cleaning.py` ... `evaluate_recommender.py`), then `python src/build_dashboard_data.py`. The multi-method files come from `python src/export_methods_data.py`.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo (big raw files in `data/` are git-ignored and not needed by the app).
2. On share.streamlit.io choose the repo, main file `app/app.py`, then Deploy.

## Project structure
```
app/app.py                   Streamlit dashboard
app_data/                    small pre-computed files used by the app
                             (methods.json + method_labels.csv hold the 4 segmentation methods)
ShopSegments_Dashboard.html  same dashboard as one file, opens in any browser (needs internet for React + font)
src/                         cleaning, EDA, clustering, recommenders, evaluation
src/build_dashboard_data.py  builds app_data/ (UMAP, similarity, product names)
src/export_methods_data.py   rebuilds app_data/methods.json and method_labels.csv from the HTML dashboard
data/                        intermediate CSVs and plots
```

## Limitations
- K-Means on raw RFM puts a few extreme spenders into a tiny VIP group (13 customers).
- One UK shop, one year of data; results may not transfer to other stores.
- Precision@10 is low in absolute terms, which is typical for sparse retail data.
- Segment movement over time and market basket analysis (bonus) are not included yet.

## Earlier work log

## Day 1: Data Cleaning, EDA, and Returns Inspection

### 1. Data Cleaning
- Original dataset contains 541,910 rows and 8 columns.
- Invoice dates were converted to datetime using the format `%m/%d/%y %H:%M`.
- Rows with missing customer IDs or descriptions were removed.
- Rows with invalid dates and non-positive prices were excluded.
- Duplicate rows were removed.
- Negative quantities were excluded from the initial purchase-based RFM dataset.
- The cleaned dataset contains 392,693 rows and 9 columns.
- No missing values remain in the cleaned dataset.
- The cleaned file was saved as `data/cleaned_retail.csv`.

### 2. Returns and Cancellations Inspection
Inspection of the original dataset produced these results:

- Total transaction rows: 541,910
- Rows with invoice numbers starting with `C`: 9,288
- Unique cancellation invoice numbers: 3,836
- Rows with negative quantities: 10,624
- Rows with negative prices: 2

Invoice numbers starting with `C` were treated as cancellation indicators. Negative quantities may represent returns or cancellations and should not automatically be treated as confirmed returns.

### 3. Exploratory Data Analysis
Four plots were generated and saved in `data/plots/`:

- `recency.png`
- `frequency.png`
- `monetary.png`
- `rfm_correlation.png`

The Recency distribution is right-skewed: many customers have relatively low recency values, while some have much higher values. Lower recency indicates a more recent purchase; higher recency indicates a longer time since the last purchase.

### 4. RFM Summary
The RFM analysis contains 4,338 customers.

- Mean recency: 92.54 days
- Median recency: 51 days
- Mean frequency: 4.27 invoices
- Median frequency: 2 invoices
- Mean monetary value: 2,048.69
- Median monetary value: 668.57

Frequency represents the number of distinct invoices, and Monetary represents the sum of quantity multiplied by price in the cleaned purchase dataset.

### 5. Day 1 Outcome
Data cleaning, initial RFM analysis, EDA plotting, and returns/cancellations inspection were completed. The next step is to review the remaining plots and proceed with customer segmentation.
## Day 2 — Feature Scaling and Clustering

### Feature Scaling
- Standardized Recency, Frequency, and Monetary features using `StandardScaler`.
- Verified that the scaled features have approximately zero means and unit standard deviations.
- Applied `log1p` transformation followed by standardization to evaluate the effect of skewed RFM distributions.

### K-Means Evaluation
- Evaluated cluster counts from 2 to 8 using inertia and silhouette score.
- The highest silhouette score on the original scaled RFM data was **0.8958 at k=2**.
- Selected **k=4 as a business-oriented candidate** for more detailed customer segmentation.

### K-Means Customer Segments (k=4)

| Cluster | Customers | Average Recency | Average Frequency | Average Monetary |
|---|---:|---:|---:|---:|
| 0 | 3,054 | 43.70 | 3.68 | 1,353.63 |
| 1 | 1,067 | 248.08 | 1.55 | 478.85 |
| 2 | 13 | 7.38 | 82.54 | 127,187.96 |
| 3 | 204 | 15.50 | 22.33 | 12,690.50 |

### Model Comparison
- Original RFM K-Means (k=4) silhouette score: **0.6162**
- Log-transformed RFM K-Means (k=4) silhouette score: **0.3375**
- DBSCAN best tested configuration: `eps=0.3`, `min_samples=5`; 2 clusters, 107 noise points, silhouette score 0.6743.
- GMM highest tested silhouette score: **0.5410 with 3 components**.

### Initial Model Decision
Original RFM K-Means with four clusters is retained as the primary candidate because it offers interpretable customer groups and a higher silhouette score than the log-transformed k=4 model. This is a provisional business-oriented choice, not proof that it is universally the best model. Cluster sizes, outliers, and business usefulness should be considered alongside silhouette scores.

### Generated Outputs
- `data/customer_rfm_scaled.csv`
- `data/customer_rfm_log_scaled.csv`
- `data/kmeans_evaluation.csv`
- `data/kmeans_cluster_profiles.csv`
- `data/customer_segments.csv`
- `data/customer_segment_summary.csv`
- `data/dbscan_evaluation.csv`
- `data/gmm_evaluation.csv`
- `data/log_kmeans_evaluation.csv`
- `data/log_customer_segments.csv`
- `data/log_customer_segment_summary.csv`
- `data/final_model_comparison.csv`