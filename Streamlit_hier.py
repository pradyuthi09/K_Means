import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mall Customer Insight Dashboard",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ AI Customer Segmentation & Marketing Recommendation System")
st.write("Hierarchical Clustering based mall customer analysis dashboard.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

@st.cache_data
def load_clustered_data():
    return pd.read_csv("Mall_hierarchi_clustered.csv")

df = load_data()
clustered_df = load_clustered_data()

# ---------------- CLEAN COLUMN NAMES ----------------
def clean_columns(dataframe):
    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.replace("(k$)", "", regex=False)
        .str.replace("(1-100)", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    return dataframe

df = clean_columns(df)
clustered_df = clean_columns(clustered_df)

# ---------------- FIND IMPORTANT COLUMNS ----------------
def find_column(dataframe, keyword):
    for col in dataframe.columns:
        if keyword.lower() in col.lower():
            return col
    return None

income_col_df = find_column(df, "income")
spend_col_df = find_column(df, "spending")
income_col_cluster = find_column(clustered_df, "income")
spend_col_cluster = find_column(clustered_df, "spending")

# detect cluster column
cluster_col = [c for c in clustered_df.columns if "cluster" in c.lower() or "label" in c.lower()][0]

st.sidebar.success("Dataset cleaned & validated ✔")

# ---------------- CLUSTER CENTERS ----------------
@st.cache_data
def get_cluster_centers():

    numeric_df = clustered_df.select_dtypes(include=np.number)
    features_df = numeric_df.loc[:, numeric_df.columns != cluster_col]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df)

    scaled_df = pd.DataFrame(scaled, columns=features_df.columns)
    scaled_df['Cluster'] = clustered_df[cluster_col].values

    centers = scaled_df.groupby('Cluster').mean()

    return centers.values, scaler, features_df.columns.tolist()

centers, scaler, feature_columns = get_cluster_centers()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🧍 Enter New Customer Details")

age = st.sidebar.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 30)
income = st.sidebar.slider("Annual Income", int(df[income_col_df].min()), int(df[income_col_df].max()), 60)
spending = st.sidebar.slider("Spending Score", 1, 100, 50)

# create input dataframe
input_data = pd.DataFrame([[age, income, spending]], columns=feature_columns)

scaled_input = scaler.transform(input_data)

# ---------------- PREDICT CLUSTER ----------------
distances = np.linalg.norm(centers - scaled_input, axis=1)
cluster = np.argmin(distances)

st.subheader("🎯 Cluster Prediction")
st.success(f"This customer belongs to Cluster {cluster}")

# ---------------- CUSTOMER TYPE & STRATEGY ----------------
def describe_cluster(income, spending):

    if income > 70 and spending > 70:
        return "💎 Premium Customer", "Offer VIP membership, premium lounge access and exclusive deals"

    elif income > 70 and spending < 40:
        return "🧠 Careful Rich Customer", "Provide personalized luxury discounts and email marketing"

    elif income < 40 and spending > 70:
        return "🛒 Impulsive Buyer", "Attract using festival sales, combo packs and limited offers"

    elif income < 40 and spending < 40:
        return "💰 Budget Customer", "Provide coupons, cashback and seasonal offers"

    else:
        return "🙂 Average Customer", "Regular advertisements and loyalty points"

cust_type, strategy = describe_cluster(income, spending)

st.info(f"Customer Type: {cust_type}")
st.warning(f"Recommended Marketing Strategy: {strategy}")

# ---------------- CUSTOMER PROFILE CARD ----------------
st.subheader("👤 Customer Profile")

c1, c2, c3 = st.columns(3)
c1.metric("Age", age)
c2.metric("Income", income)
c3.metric("Spending Score", spending)

# ---------------- CLUSTER DISTRIBUTION ----------------
st.subheader("📊 Cluster Distribution")

cluster_counts = clustered_df[cluster_col].value_counts().sort_index()

fig = go.Figure(data=[go.Pie(
    labels=[f"Cluster {i}" for i in cluster_counts.index],
    values=cluster_counts.values,
    textinfo='label+percent'
)])

st.plotly_chart(fig, use_container_width=True)

# ---------------- SCATTER PLOT ----------------
st.subheader("📈 Customer Segments Visualization")

fig2 = px.scatter(
    clustered_df,
    x=income_col_cluster,
    y=spend_col_cluster,
    color=clustered_df[cluster_col].astype(str),
    title="Customer Segments"
)

# new customer point
fig2.add_scatter(
    x=[income],
    y=[spending],
    mode="markers",
    marker=dict(color="red", size=14, symbol="diamond"),
    name="New Customer"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- CLUSTER COMPARISON ----------------
st.subheader("📊 Cluster Behavior Comparison")

cluster_summary = clustered_df.groupby(cluster_col).mean(numeric_only=True)
st.dataframe(cluster_summary)

fig_bar = px.bar(
    cluster_summary,
    x=cluster_summary.index,
    y=[income_col_cluster, spend_col_cluster],
    barmode="group",
    title="Average Income & Spending per Cluster"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- DENDROGRAM ----------------
st.subheader("🌳 Hierarchical Relationship (Dendrogram)")

sample_df = clustered_df[[income_col_cluster, spend_col_cluster]].sample(40, random_state=1)

linked = linkage(sample_df, method="ward")

fig_d, ax = plt.subplots(figsize=(10,5))
dendrogram(linked, ax=ax)
plt.xlabel("Customers")
plt.ylabel("Distance")

st.pyplot(fig_d)

# ---------------- DATASET STATS ----------------
st.subheader("📌 Dataset Statistics")

d1, d2, d3 = st.columns(3)
d1.metric("Total Customers", len(df))
d2.metric("Average Age", round(df["Age"].mean(), 1))
d3.metric("Average Income", round(df[income_col_df].mean(), 1))

st.markdown("---")
st.caption("Powered by Hierarchical Clustering (Agglomerative - Ward Linkage)")
