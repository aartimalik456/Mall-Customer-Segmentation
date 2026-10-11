import streamlit as st
import joblib
import numpy as np

st.title("🎯 Predict Customer Segment")

# Load saved model
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.write("Enter the customer details below:")

# Input fields
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=1,
    max_value=200,
    value=50
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Customer Segment"):

    customer = np.array([[age, income, score]])

    customer_scaled = scaler.transform(customer)

    cluster = int(kmeans.predict(customer_scaled)[0])

    st.success(f"Customer belongs to Cluster {cluster}")

    # Cluster descriptions
    descriptions = {
        0: ("Premium Customer", "Offer exclusive memberships and premium products."),
        1: ("Budget Customer", "Recommend affordable products and discounts."),
        2: ("Regular Customer", "Maintain engagement with seasonal offers."),
        3: ("High Income - Low Spending", "Encourage spending using loyalty rewards."),
        4: ("Young Active Customer", "Promote trending products and personalized recommendations.")
    }

    title, recommendation = descriptions.get(
        cluster,
        ("Customer", "No recommendation available.")
    )

    st.subheader(title)
    st.info(recommendation)