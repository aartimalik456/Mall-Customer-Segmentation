import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("mall_customer_segmentation.csv")

st.title("🛍 Mall Customer Segmentation Dashboard")

st.markdown("""
### Welcome!

This project uses **Machine Learning (K-Means Clustering)** to divide mall customers into different groups based on:

- 👤 Age
- 💰 Annual Income
- 🎯 Spending Score

The segmentation helps businesses understand customer behavior and design targeted marketing strategies.
""")

st.markdown("---")

# KPI Cards
col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Customers", len(df))
col2.metric("💰 Average Income", f"{df['Annual Income (k$)'].mean():.2f} k$")
col3.metric("⭐ Average Spending", f"{df['Spending Score (1-100)'].mean():.2f}")

st.markdown("---")

st.subheader("🎯 Project Workflow")

st.write("""
1. Load the dataset
2. Clean the data
3. Select features
4. Scale the data
5. Apply K-Means Clustering
6. Visualize customer segments
7. Predict customer segment
""")

st.markdown("---")

st.success("Use the sidebar to explore the dashboard.")