"""
ShopSegments - Customer Segmentation & Product Recommender (Streamlit dashboard)

Same layout as ShopSegments_Dashboard.html: method picker, segment filter, map view toggle,
segment cards, revenue trend, recommendations and model quality.
Reads the small pre-computed files in app_data/ (built by src/build_dashboard_data.py + methods.json / method_labels.csv).

Run:  streamlit run app/app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DATA = Path(__file__).resolve().parent.parent / "app_data"

st.set_page_config(page_title="ShopSegments", page_icon="🛍️", layout="wide", initial_sidebar_state="collapsed")

PLAIN = {  # one-line plain-English description for non-technical readers
    "VIP Champions": "Very few customers, but they buy constantly and spend the most.",
    "Loyal High-Value Customers": "Buy often and spend a lot. The backbone of revenue.",
    "Regular Customers": "The largest group. Steady, average buyers with room to grow.",
    "At-Risk Customers": "Have not bought for a long time. Likely to be lost soon.",
}

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1440px; }
    div[data-testid="stMetric"] { background:#fff; padding:14px 18px; border-radius:10px; border:1px solid #E2E8F0; text-align:center; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color:#0F172A; justify-content:center; }
    div[data-testid="stMetricLabel"] { justify-content:center; }
    .card { background:#fff; border:1px solid #E2E8F0; border-radius:12px; padding:16px 18px; height:100%; }
    .card h4 { margin:0 0 2px 0; }
    .card .sub { color:#64748B; font-size:0.88rem; margin-bottom:10px; }
    .card .big { font-size:1.5rem; font-weight:700; }
    .action { background:#F1F5F9; border-radius:8px; padding:8px 10px; font-size:0.9rem; margin-top:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    d = {n: pd.read_csv(APP_DATA / f"{n}.csv") for n in [
        "customers", "segments", "segment_products", "products", "item_similarity",
        "als_recs", "popular", "metrics", "purchases", "monthly_revenue", "method_labels"]}
    for k in ("StockCode", "Also_Bought", "Recommended StockCode"):
        for df in d.values():
            if k in df.columns:
                df[k] = df[k].astype(str)
    d["methods"] = json.loads((APP_DATA / "methods.json").read_text(encoding="utf-8"))
    return d


D = load()
cust, segs, products = D["customers"], D["segments"].set_index("Segment_Name"), D["products"]
METHODS, MONTHS = D["methods"]["methods"], D["methods"]["months"]
name_of = products.set_index("StockCode")["Description"].to_dict()
price_ok = lambda codes: [name_of.get(c, f"Item {c}") for c in codes]

# ------------------------------------------------------------------ header
st.title("ShopSegments")
st.caption("Customer segmentation and product recommendations: see who your best customers are and what to offer them.")

tab1, tab2, tab3 = st.tabs(["📊 Customer Segments", "🎯 Product Recommendations", "🧪 Model Quality"])

# ================================================================== TAB 1
with tab1:
    kpi = st.container()                      # filled after the filters below are known
    left, mid, right = st.columns([1.05, 3, 1.5], gap="medium")

    with left:
        with st.container(border=True):
            mi = st.selectbox("Segmentation method", range(len(METHODS)), format_func=lambda i: METHODS[i]["label"])
            M = METHODS[mi]
            st.caption(f"{M['desc']}  \nSilhouette score: **{M['sil']:.3f}**")
            st.markdown("**Segments**")
            chosen = [i for i, s in enumerate(M["seg"])
                      if st.checkbox(f"{s['short']} ({s['cust']:,})", True, key=f"seg_{M['id']}_{i}")]
            st.markdown("**Map view**")
            vmode = st.radio("Map view", ["Similarity", "Orders × spend"], horizontal=True, label_visibility="collapsed")
            st.markdown("**How to read**")
            st.caption("Recency = days since last order (lower is better). Frequency = number of orders. Monetary = total spend in £.")

    if not chosen:
        st.warning("Pick at least one segment, showing all of them instead.")
        chosen = list(range(len(M["seg"])))

    SEG = M["seg"]
    short = [s["short"] for s in SEG]
    CMAP = {s["short"]: s["col"] for s in SEG}
    df = cust.merge(D["method_labels"][["Customer ID", M["id"]]].rename(columns={M["id"]: "Seg"}), on="Customer ID")
    df["Segment"] = df["Seg"].map(lambda i: short[i])
    view = df[df["Seg"].isin(chosen)]
    order = [short[i] for i in chosen]

    k1, k2, k3, k4 = kpi.columns(4)
    k1.metric("Total customers", f"{len(view):,}")
    k2.metric("Segments", f"{len(chosen)}")
    k3.metric("Average spend", f"£{view['Monetary'].mean():,.0f}")
    k4.metric("Average orders", f"{view['Frequency'].mean():.1f}")

    with mid:
        with st.container(border=True):
            st.subheader("Customer map")
            if vmode == "Similarity":
                fig = px.scatter(view, x="UMAP_1", y="UMAP_2", color="Segment", color_discrete_map=CMAP,
                                 category_orders={"Segment": order}, opacity=0.75,
                                 hover_data={"Customer ID": True, "Recency": True, "Frequency": True,
                                             "Monetary": ":,.0f", "UMAP_1": False, "UMAP_2": False, "Segment": False})
                fig.update_xaxes(showticklabels=False, title="")
                fig.update_yaxes(showticklabels=False, title="")
                st.caption("Dots close together behave alike (UMAP projection of Recency, Frequency, Monetary).")
            else:
                fig = px.scatter(view, x="Frequency", y="Monetary", color="Segment", color_discrete_map=CMAP,
                                 category_orders={"Segment": order}, opacity=0.75, log_x=True, log_y=True,
                                 hover_data={"Customer ID": True, "Recency": True, "Frequency": True, "Monetary": ":,.0f", "Segment": False},
                                 labels={"Frequency": "More orders (log scale) →", "Monetary": "Total spend £ (log scale) →"})
                st.caption("Top right = frequent and big spenders.")
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
                              legend_title_text="", legend=dict(orientation="h", y=-0.12))
            st.plotly_chart(fig, width="stretch")

        with st.container(border=True):
            st.subheader("Segment profiles")
            rows = []
            for i in chosen:
                s = SEG[i]
                for lab, key in (("Avg spend", "mon"), ("Avg orders", "freq"), ("Recency (days)", "rec")):
                    mx = max(x[key] for x in SEG)
                    rows.append({"Segment": s["short"], "Metric": lab, "Scaled (0-100)": s[key] / mx * 100, "Value": s[key]})
            pb = px.bar(pd.DataFrame(rows), x="Segment", y="Scaled (0-100)", color="Metric", barmode="group",
                        color_discrete_map={"Avg spend": "#E8735A", "Avg orders": "#4F6FE8", "Recency (days)": "#5BBF9F"},
                        hover_data={"Value": ":,.1f", "Scaled (0-100)": False})
            pb.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", legend_title_text="",
                             legend=dict(orientation="h", y=-0.2), xaxis_title="", yaxis_title="")
            st.plotly_chart(pb, width="stretch")
            st.caption("Each metric is scaled 0-100 against the highest segment. Hover a bar for the real value.")

    with right:
        with st.container(border=True):
            st.subheader("Segment sizes")
            pie = px.pie(view, names="Segment", color="Segment", color_discrete_map=CMAP, hole=0.5,
                         category_orders={"Segment": order})
            pie.update_traces(textinfo="percent")
            pie.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.1), legend_title_text="")
            st.plotly_chart(pie, width="stretch")
        with st.container(border=True):
            st.subheader("Top customers")
            top = (view.sort_values("Monetary", ascending=False).head(8)
                   [["Customer ID", "Segment", "Monetary", "Frequency"]]
                   .rename(columns={"Monetary": "Spend (£)", "Frequency": "Orders"}))
            top["Customer ID"] = top["Customer ID"].astype(int)
            st.dataframe(top, hide_index=True, width="stretch",
                         column_config={"Spend (£)": st.column_config.NumberColumn(format="£%,.0f")})
        with st.container(border=True):
            st.subheader("Insights")
            st.markdown("\n".join(f"- {t}" for t in M["ins"]))

    # ---- segment cards
    cols = st.columns(len(chosen))
    tot_rev = sum(s["rev"] for s in SEG)
    for col, i in zip(cols, chosen):
        s = SEG[i]
        top_html = "".join(f"<li>{t}</li>" for t in s["top"])
        plain = PLAIN.get(s["n"], "")
        col.markdown(
            f"""<div class="card" style="border-top:5px solid {s['col']}">
            <h4>{s['n']}</h4><div class="sub">{plain}</div>
            <div class="big">{s['cust']:,} <span style="font-size:0.9rem;font-weight:400">customers</span></div>
            <div>Avg spend <b>£{s['mon']:,.0f}</b> · {s['rev'] / tot_rev:.0%} of revenue</div>
            <div>Last order <b>{s['rec']:.0f}</b> days ago · <b>{s['freq']:.1f}</b> orders</div>
            <div style="margin-top:8px"><b>Top products</b><ul style="margin:2px 0 0 18px;padding:0">{top_html}</ul></div>
            <div class="action"><b>Action:</b> {s['act']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("Revenue by month and segment")
    tr = pd.DataFrame([{"Month": m, "Segment": short[i], "Revenue (£)": v}
                       for i in chosen for m, v in zip(MONTHS, M["trend"][i])])
    ln = px.line(tr, x="Month", y="Revenue (£)", color="Segment", color_discrete_map=CMAP, category_orders={"Segment": order})
    ln.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", legend_title_text="")
    st.plotly_chart(ln, width="stretch")

    with st.expander("Download segment list (CSV)"):
        out = view[["Customer ID", "Segment", "Recency", "Frequency", "Monetary"]]
        st.download_button("Download customers + segments", out.to_csv(index=False), "customer_segments.csv", "text/csv")
        st.dataframe(out.head(200), hide_index=True, width="stretch")

# ================================================================== TAB 2
with tab2:
    a, b = st.columns(2, gap="large")

    # ---------- Customers who bought this also bought
    with a:
        st.subheader("Customers who bought this also bought")
        sim_items = set(D["item_similarity"]["StockCode"])
        cat = products[products["StockCode"].isin(sim_items)].sort_values("Buyers", ascending=False)
        labels = dict(zip(cat["StockCode"], cat["Description"] + "  (" + cat["StockCode"] + ")"))
        pick = st.selectbox("Choose a product (type to search)", list(labels), format_func=labels.get, index=0)
        rel = D["item_similarity"].query("StockCode == @pick").sort_values("Rank").head(10)
        if rel.empty:
            st.info("No related products found for this item.")
        else:
            out = pd.DataFrame({"#": rel["Rank"].values,
                                "Product": price_ok(rel["Also_Bought"]),
                                "Match strength": rel["Similarity"].values})
            st.dataframe(out, hide_index=True, width="stretch",
                         column_config={"Match strength": st.column_config.ProgressColumn(
                             min_value=0, max_value=float(out["Match strength"].max()), format="%.2f")})
            st.caption("Match strength = how often the two products appear in the same customers' baskets (cosine similarity).")

    # ---------- Personalised for a customer
    with b:
        st.subheader("Top-10 picks for a customer")
        mode = st.radio("Who is the customer?", ["Existing customer", "New customer (no history)"], horizontal=True)

        if mode == "New customer (no history)":
            st.info("We know nothing about this person yet, so we show the **best-selling products** overall. "
                    "This is the *cold-start* fallback.")
            pop = D["popular"].sort_values("Rank")
            st.dataframe(pd.DataFrame({"#": pop["Rank"], "Product": price_ok(pop["StockCode"])}),
                         hide_index=True, width="stretch")
        else:
            order = cust.sort_values("Monetary", ascending=False)
            ids = order["Customer ID"].astype(int).tolist()
            seg_of = cust.set_index(cust["Customer ID"].astype(int))["Segment_Name"].to_dict()
            cid = st.selectbox("Customer ID", ids, format_func=lambda i: f"{i}  ·  {seg_of[i]}")
            row = cust[cust["Customer ID"].astype(int) == cid].iloc[0]
            st.markdown(
                f"**{row['Segment_Name']}** · spent £{row['Monetary']:,.0f} · {int(row['Frequency'])} orders · "
                f"last bought {int(row['Recency'])} days ago"
            )
            st.markdown(f"*Suggested action:* {row['Marketing_Action']}")

            recs = D["als_recs"][D["als_recs"]["Customer ID"] == cid].sort_values("Rank").head(10)
            if recs.empty:
                st.warning("No personal model output for this customer, showing best-sellers instead.")
                recs = D["popular"].rename(columns={"StockCode": "Recommended StockCode"})
            st.dataframe(pd.DataFrame({"#": recs["Rank"].values,
                                       "Recommended product": price_ok(recs["Recommended StockCode"])}),
                         hide_index=True, width="stretch")
            with st.expander("What this customer already bought"):
                h = D["purchases"][D["purchases"]["Customer ID"] == cid].head(10)
                st.dataframe(pd.DataFrame({"Product": price_ok(h["StockCode"]), "Units": h["Units"].values}),
                             hide_index=True, width="stretch")

# ================================================================== TAB 3
with tab3:
    st.subheader("How good are the recommendations?")
    m = D["metrics"].copy()
    m["Precision@10"] = (m["Precision@10"] * 100).round(2)
    st.dataframe(m.rename(columns={"Precision@10": "Precision@10 (%)"}), hide_index=True, width="stretch")
    st.caption("Tested on a hold-out set: each customer's most recent order was hidden, the model recommended 10 products, "
               "and we counted how many of them the customer really bought next. ALS is about 2x better than item-based filtering.")

    st.subheader("Why 4 segments?")
    ev = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "kmeans_evaluation.csv")
    e1, e2 = st.columns(2)
    f1 = px.line(ev, x="k", y="inertia", markers=True, title="Elbow (lower = tighter groups)")
    f2 = px.line(ev, x="k", y="silhouette_score", markers=True, title="Silhouette score (higher = better separated)")
    for f in (f1, f2):
        f.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="white")
    e1.plotly_chart(f1, width="stretch")
    e2.plotly_chart(f2, width="stretch")
    st.markdown(
        "- Silhouette is highest at k=2, but two groups (\"normal\" vs \"a few huge buyers\") is too coarse for marketing.\n"
        "- k=4 keeps a good silhouette (0.62) **and** gives four groups a marketing team can act on.\n"
        "- DBSCAN and Gaussian Mixture were also tested; they gave fewer or less useful groups."
    )

    st.subheader("Known limitations")
    st.markdown(
        "- Segments come from K-Means on raw RFM, so a few extreme spenders form their own tiny VIP group.\n"
        "- Data is one UK online store from 2010-2011; results will differ on other shops.\n"
        "- Precision@10 is low in absolute terms (typical for sparse retail data)."
    )

    st.subheader("Segmentation methods compared")
    st.dataframe(pd.DataFrame([{"Method": m["label"], "Segments": len(m["seg"]), "Silhouette": m["sil"], "What it does": m["desc"]}
                               for m in METHODS]), hide_index=True, width="stretch")
