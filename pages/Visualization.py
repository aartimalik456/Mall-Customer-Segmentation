import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("📈 Data Visualization")

# Load Dataset
data = pd.read_csv("mall_customer_segmentation.csv")

# -----------------------------
# Age Distribution
# -----------------------------
st.subheader("Age Distribution")

fig = px.histogram(
    data,
    x="Age",
    nbins=20,
    color_discrete_sequence=["royalblue"]
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Income Distribution
# -----------------------------
st.subheader("Annual Income Distribution")

fig = px.histogram(
    data,
    x="Annual Income (k$)",
    nbins=20,
    color_discrete_sequence=["green"]
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Spending Score Distribution
# -----------------------------
st.subheader("Spending Score Distribution")

fig = px.histogram(
    data,
    x="Spending Score (1-100)",
    nbins=20,
    color_discrete_sequence=["red"]
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("Correlation Heatmap")

numeric_data = data.select_dtypes(include=["number"])

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)