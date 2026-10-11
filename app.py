import streamlit as st

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🛍 Mall Customer Segmentation Dashboard")
st.sidebar.image("assets/logo.png", use_container_width=True)

st.markdown("""
Welcome to the **Mall Customer Segmentation Dashboard**.

This application uses **Machine Learning (K-Means Clustering)** to divide customers into different groups based on:

- 👤 Age
- 💰 Annual Income
- 🎯 Spending Score

Use the sidebar to explore different pages.
""")

st.sidebar.success("📂 Select a page")