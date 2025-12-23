# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# -------------------------
# Load Models & Data
# -------------------------
rf_model = joblib.load('rf_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
rfm = pd.read_csv('rfm_data.csv')
forecast = pd.read_csv('sales_forecast.csv')
data_clean = pd.read_csv('data_clean.csv')  # Optional: full cleaned dataset

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Smart Retail Analytics", layout="wide")
st.title("Smart Retail Analytics & Prediction System")

# -------------------------
# Sidebar Navigation
# -------------------------
menu = ["Dashboard", "Customer Segmentation", "Churn Prediction", "Sales Forecast"]
choice = st.sidebar.selectbox("Select Section", menu)

# -------------------------
# Dashboard Section
# -------------------------
if choice == "Dashboard":
    st.header("Business Dashboard")
    
    total_revenue = data_clean['TotalPrice'].sum()
    total_customers = data_clean['Customer ID'].nunique()
    total_transactions = data_clean['Invoice'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Customers", total_customers)
    col3.metric("Total Transactions", total_transactions)
    
    # Monthly Sales Chart
    monthly_sales = data_clean.set_index('InvoiceDate')['TotalPrice'].resample('M').sum()
    st.subheader("Monthly Revenue Trend")
    st.line_chart(monthly_sales)
    
    # Top Products
    st.subheader("Top 10 Best-Selling Products")
    top_products = data_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

# -------------------------
# Customer Segmentation
# -------------------------
elif choice == "Customer Segmentation":
    st.header("👤 Customer Segmentation")
    
    st.write("RFM Cluster Summary:")
    st.dataframe(rfm.groupby('Cluster').mean())
    
    st.subheader("Cluster Distribution")
    st.bar_chart(rfm['Cluster'].value_counts())

# -------------------------
# Churn Prediction Tool
# -------------------------
elif choice == "Churn Prediction":
    st.header("Customer Churn Prediction")
    
    st.write("Select customer features:")
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=100, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=10000.0, value=100.0)
    cluster = st.selectbox("Cluster", rfm['Cluster'].unique())
    
    if st.button("Predict Churn"):
        X_new = np.array([[recency, frequency, monetary, cluster]])
        pred = rf_model.predict(X_new)
        prob = rf_model.predict_proba(X_new)[0][1]
        st.write(f"Churn Prediction: {'Yes' if pred[0]==1 else 'No'}")
        st.write(f"Probability of Churn: {prob:.2f}")

# -------------------------
# Sales Forecast Section
# -------------------------
elif choice == "Sales Forecast":
    st.header("Sales Forecast")
    
    st.subheader("Next 12 Months Forecast")
    forecast_display = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12)
    forecast_display.set_index('ds', inplace=True)
    st.line_chart(forecast_display['yhat'])
    st.write(forecast_display)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("Developed by Muhammed – Smart Retail Analytics & Prediction System")