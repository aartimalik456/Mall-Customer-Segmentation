import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("👥 Customer Segmentation")

# Load dataset
data = pd.read_csv("mall_customer_segmentation.csv")

# Select features
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Elbow Method
# -------------------------
st.subheader("📉 Elbow Method")

inertia = []

for i in range(1,11):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia.append(model.inertia_)

fig = px.line(
    x=range(1,11),
    y=inertia,
    markers=True,
    labels={
        "x":"Number of Clusters",
        "y":"Inertia"
    },
    title="Elbow Method"
)

st.plotly_chart(fig, use_container_width=True)

st.info("The elbow point suggests using **5 clusters**.")

# -------------------------
# Train Final Model
# -------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------
# Cluster Scatter Plot
# -------------------------

st.subheader("📊 Customer Segments")

fig = px.scatter(
    data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=data["Cluster"].astype(str),
    hover_data=["Age"],
    size="Age",
    title="Customer Segments"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Cluster Statistics
# -------------------------

st.subheader("📋 Cluster Analysis")

cluster_summary = data.groupby("Cluster").mean(numeric_only=True)

st.dataframe(cluster_summary, use_container_width=True)

# -------------------------
# Customers in Each Cluster
# -------------------------

st.subheader("📌 Customers per Cluster")

cluster_count = data["Cluster"].value_counts().sort_index()

st.bar_chart(cluster_count)

# -------------------------
# Download Button
# -------------------------

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Segmented Dataset",
    csv,
    "Mall_Customers_Segmented.csv",
    "text/csv"
)
st.markdown("---")
st.subheader("📌 Business Insights")

st.info("""
✔ Customers with high income and high spending are ideal targets for premium memberships.

✔ Customers with high income but low spending can be encouraged through loyalty programs.

✔ Customers with low income and high spending respond well to discounts and offers.

✔ Average customers should receive personalized recommendations.
""")