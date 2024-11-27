import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

# Cluster Labels
def get_cluster_label(cluster):
    cluster_labels = {
        0: "Win-Back Campaign",
        1: "Minimal Effort Group",
        2: "Loyal VIPs",
        3: "Upsell Potential",
        6: "High Value Retention Targets",
        5: "Re-Engagement Campaigns",
        4: "Dormant Customers"
    }
    return cluster_labels.get(cluster, "Unknown")

# Load model dynamically
def load_model(model_path):
    return joblib.load(model_path)

# App Title
st.title("Online Retail Customer Segmentation and Classification")

# Upload data section
uploaded_file = st.file_uploader("Upload Online Retail Dataset", type=["csv"])
if uploaded_file is not None:
    # Read and display the uploaded data
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.write("### Uploaded Dataset Preview:")
    st.dataframe(data.head())

    # Preprocessing and feature engineering
    st.write("### Preprocessing Data...")
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
    data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

    aggregated_df = data.groupby('CustomerID').agg(
        MonetaryValue=('TotalSpend', 'sum'),
        Frequency=('InvoiceNo', 'nunique'),
        LastInvoiceDate=('InvoiceDate', 'max')
    ).reset_index()

    max_invoice_date = aggregated_df['LastInvoiceDate'].max()
    aggregated_df['Recency'] = (max_invoice_date - aggregated_df['LastInvoiceDate']).dt.days
    aggregated_df = aggregated_df.drop(columns=['LastInvoiceDate'])

    st.write("### Processed Customer Metrics:")
    st.dataframe(aggregated_df.head())

    # Scaling features
    scaler = StandardScaler()
    features = ['MonetaryValue', 'Frequency', 'Recency']
    aggregated_df_scaled = aggregated_df.copy()
    aggregated_df_scaled[features] = scaler.fit_transform(aggregated_df[features])

    # Model selection
    model_files = {
        "Decision Tree": "models/abc_clf_model.pkl",
        "Random Forest": "models/bagging_clf_model.pkl",
        "SVM": "models/svc_clf_model.pkl",
        "Gradient Boosting": "models/xgbC_model.pkl",
        "MLP Classifier": "models/mlp_clf_model.pkl",
        "Naive Bayes": "models/nbc_model.pkl",
        "KNN": "models/neigh_model.pkl",
        "Random Forest Classifier":"models/rfc_clf_model.pkl",
        "AdaBoost": "models/abc_clf_model.pkl",
        "Bagging": "models/bagging_clf_model.pkl",
        "Stacking": "models/stack_clf_model.pkl"
    }

    model_choice = st.selectbox("Select a Classification Model", list(model_files.keys()))
    selected_model_path = model_files[model_choice]
    model = load_model(selected_model_path)

    # Predictions
    st.write("### Predicting Customer Segments...")
    aggregated_df['Cluster'] = model.predict(aggregated_df_scaled[features])
    aggregated_df['Segment'] = aggregated_df['Cluster'].apply(get_cluster_label)

    st.write("### Predicted Segments:")
    st.dataframe(aggregated_df[['CustomerID', 'Cluster', 'Segment']].head())

    # Visualization
    st.write("### Segment Distribution:")
    fig = px.pie(aggregated_df, names='Segment', title='Customer Segment Distribution')
    st.plotly_chart(fig)

    # Download results
    st.write("### Download Segmentation Results:")
    csv = aggregated_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "customer_segments.csv", "text/csv")
