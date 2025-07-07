import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Page setup
st.set_page_config(page_title="🧠 Customer Segment Predictor", layout="wide")
st.title("🧠 Customer Segment Predictor using KMeans")

#Load data directly (no upload)
csv_path = "Mall_Customers.csv"
if not os.path.exists(csv_path):
    st.error("❌ 'Mall_Customers.csv' not found in the app directory.")
    st.stop()

df = pd.read_csv(csv_path)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
st.success("✅ Dataset loaded successfully.")

#Feature selection
features = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train KMeans model
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Sidebar: New customer input
st.sidebar.header("🧍 Enter New Customer Details")
age = st.sidebar.slider("Age", 15, 70, 30)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 60)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

#Prepare input for prediction
gender_num = 0 if gender == "Male" else 1
new_customer = np.array([[age, gender_num, income, score]])
new_customer_scaled = scaler.transform(new_customer)
predicted_cluster = kmeans.predict(new_customer_scaled)[0]

#Output: Prediction
st.subheader("🎯 Predicted Segment")
st.success(f"New customer belongs to **Cluster {predicted_cluster}**")

#Cluster summary
st.subheader("📊 Cluster Profiles")
summary = df.groupby('Cluster')[features].mean().reset_index()
st.dataframe(summary.style.highlight_max(axis=0))

#Strategy output
st.subheader("💡 Suggested Marketing Strategy")

def strategy_for_cluster(cluster_id):
    return {
        0: "🎁 Send discount offers and loyalty rewards.",
        1: "💼 Upsell luxury or premium services.",
        2: "📣 Offer exclusive bundles and perks.",
        3: "🧪 A/B test offers and monitor response.",
        4: "📍 Focus on brand awareness and trust."
    }.get(cluster_id, "No strategy available.")

st.markdown(f"**Strategy for Cluster {predicted_cluster}:** {strategy_for_cluster(predicted_cluster)}")

# Plot clusters
st.subheader("📍 Visualizing Customer Segments")
fig, ax = plt.subplots()
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2')
plt.scatter(income, score, c='black', s=200, marker='X', label='New Customer')
plt.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built with l❤️ve by group 10")
