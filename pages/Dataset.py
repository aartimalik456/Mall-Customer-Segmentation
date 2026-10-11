import streamlit as st
import pandas as pd

st.title("📊 Dataset Explorer")

# Load Dataset
data = pd.read_csv("mall_customer_segmentation.csv")

st.subheader("Dataset Preview")
st.dataframe(data, use_container_width=True)

st.subheader("Dataset Shape")
col1, col2 = st.columns(2)
col1.metric("Rows", data.shape[0])
col2.metric("Columns", data.shape[1])

st.subheader("Column Names")
st.write(data.columns.tolist())

st.subheader("Statistical Summary")
st.dataframe(data.describe(), use_container_width=True)

# Download Button
csv = data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name="Mall_Customers.csv",
    mime="text/csv"
)