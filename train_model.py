import os
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load Dataset
# ----------------------------

data = pd.read_csv("mall_customer_segmentation.csv")

# Remove duplicates
data.drop_duplicates(inplace=True)

# Features
X = data[['Age',
          'Annual Income (k$)',
          'Spending Score (1-100)']]

# ----------------------------
# Scale Features
# ----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train KMeans
# ----------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# ----------------------------
# Save Models
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(kmeans, "models/kmeans_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print("Model Saved Successfully")