import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Analytics Platform", layout="wide")
st.title("Customer Behavior & Sales Analytics Platform")
st.caption("Upload a transactions CSV. The app will auto-detect and map common column names.")

uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize: lowercase + trim + replace spaces/dashes with underscores
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    # Collapse repeated underscores
    df.columns = df.columns.str.replace(r"__+", "_", regex=True)
    return df


def auto_map_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common dataset column names to the schema we need:
    order_id, customer_id, order_date, category, net_sales
    """
    rename_map = {
        # order id variants
        "orderid": "order_id",
        "order": "order_id",
        "invoice_id": "order_id",
        "invoice": "order_id",
        "transaction_id": "order_id",
        "txn_id": "order_id",

        # customer id variants
        "customerid": "customer_id",
        "cust_id": "customer_id",
        "user_id": "customer_id",
        "client_id": "customer_id",

        # date variants
        "orderdate": "order_date",
        "date": "order_date",
        "transaction_date": "order_date",
        "txn_date": "order_date",
        "purchase_date": "order_date",

        # category variants
        "product_category": "category",
        "productcategory": "category",
        "cat": "category",
        "department": "category",
        "segment": "category",

        # net sales variants
        "netsales": "net_sales",
        "sales": "net_sales",
        "sales_amount": "net_sales",
        "amount": "net_sales",
        "revenue": "net_sales",
        "net_revenue": "net_sales",
        "total_sales": "net_sales",
        "order_value": "net_sales",
        "price": "net_sales",
    }

    # Only rename columns that appear in df
    to_rename = {}
    for c in df.columns:
        key = c.replace("__", "_")
        if key in rename_map:
            to_rename[c] = rename_map[key]
    return df.rename(columns=to_rename)


def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit/pyarrow fails if df has duplicate column names.
    This function resolves duplicates by keeping the version with fewer missing values.
    """
    if not df.columns.duplicated().any():
        return df

    dup_names = df.columns[df.columns.duplicated()].tolist()
    st.warning(f"Duplicate columns detected and fixed: {sorted(set(dup_names))}")

    new_df = pd.DataFrame(index=df.index)

    for col in pd.unique(df.columns):
        same = df.loc[:, df.columns == col]
        if same.shape[1] == 1:
            new_df[col] = same.iloc[:, 0]
        else:
            # choose the column with fewer missing values
            miss_counts = same.isna().sum().values
            best_idx = int(np.argmin(miss_counts))
            new_df[col] = same.iloc[:, best_idx]

    return new_df


def find_best_kmeans(X: np.ndarray, k_min=2, k_max=6, random_state=42):
    best_k, best_score = None, -1
    scores = {}
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score, scores


if not uploaded_file:
    st.info("Upload a CSV file to start.")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Normalize + map + fix duplicates
df = normalize_columns(df)
df = auto_map_schema(df)
df = fix_duplicate_columns(df)

# Debug section (optional)
with st.expander("Debug: Detected columns & preview"):
    st.write(df.columns.tolist())
    st.dataframe(df.head(10))

required_cols = ["order_id", "customer_id", "order_date", "category", "net_sales"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(
        "CSV schema mismatch.\n\n"
        f"Missing required columns: {missing}\n\n"
        f"Required: {required_cols}\n\n"
        "Fix by renaming columns in your CSV, or tell me your exact headers and I’ll map them."
    )
    st.stop()

# Basic cleanup
df = df.drop_duplicates()

# Parse dates
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.dropna(subset=["order_date"])

# Ensure sales numeric
df["net_sales"] = pd.to_numeric(df["net_sales"], errors="coerce")
df = df.dropna(subset=["net_sales"])

# KPIs
total_revenue = float(df["net_sales"].sum())
total_orders = int(df["order_id"].nunique())
total_customers = int(df["customer_id"].nunique())
avg_order_value = total_revenue / total_orders if total_orders else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"${total_revenue:,.2f}")
k2.metric("Total Orders", f"{total_orders:,}")
k3.metric("Total Customers", f"{total_customers:,}")
k4.metric("Avg Order Value (ATV)", f"${avg_order_value:,.2f}")

st.divider()

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("Monthly Revenue Trend")
    monthly = (
        df.assign(month=df["order_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["net_sales"].sum()
        .sort_index()
    )
    st.line_chart(monthly)

with c2:
    st.subheader("Revenue by Category")
    cat_sales = df.groupby("category")["net_sales"].sum().sort_values(ascending=False)
    st.bar_chart(cat_sales)

st.divider()

# RFM + clustering
st.subheader("Customer Segmentation (RFM + KMeans)")

snapshot_date = df["order_date"].max()

rfm = df.groupby("customer_id").agg(
    Recency=("order_date", lambda x: (snapshot_date - x.max()).days),
    Frequency=("order_id", "count"),
    Monetary=("net_sales", "sum"),
)

if rfm.shape[0] < 10:
    st.warning("Not enough unique customers to run segmentation (need at least ~10). Showing RFM table only.")
    st.dataframe(rfm.head(50))
    st.stop()

# Scale for KMeans
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# Find best K
best_k, best_sil, sil_scores = find_best_kmeans(X, k_min=2, k_max=6, random_state=42)

l, r = st.columns([1, 1])
with l:
    st.write("Silhouette score by K:")
    st.json(sil_scores)
with r:
    st.write(f"Selected K: **{best_k}**")
    st.write(f"Best silhouette score: **{best_sil:.3f}**")

# Fit final model
model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
rfm["Cluster"] = model.fit_predict(X)

# Cluster summary
summary = rfm.groupby("Cluster").agg(
    Customers=("Cluster", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum"),
).sort_values("Total_Revenue", ascending=False)

s1, s2 = st.columns([1.2, 0.8])
with s1:
    st.write("Cluster Summary")
    st.dataframe(summary)
with s2:
    st.write("Customers per Cluster")
    st.bar_chart(summary["Customers"])

st.divider()

# Export segments
st.subheader("Export Results")
segments_out = rfm.reset_index()
st.download_button(
    label="Download customer_segments.csv",
    data=segments_out.to_csv(index=False).encode("utf-8"),
    file_name="customer_segments.csv",
    mime="text/csv",
)

st.caption("Tip: Import customer_segments.csv into Power BI and relate it using customer_id.")
