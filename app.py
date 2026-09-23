import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="AI Churn Analytics", layout="wide")

st.title("AI-Powered Customer Churn & Revenue Analytics")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output"

st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to:", ["Business Insights", "Key Performance Indicators", "Visualizations", "Data explorer"])

if menu == "Business Insights":
    st.header("Business Insights")
    insights_file = OUTPUT / "ai_business_insights.txt"
    if insights_file.exists():
        text = insights_file.read_text(encoding="utf-8")
        st.text(text)
    else:
        st.warning("Insights file not found. Please run the analytics script first.")

elif menu == "Key Performance Indicators":
    st.header("KPI Summary")
    kpi_file = OUTPUT / "kpi_summary.csv"
    if kpi_file.exists():
        kpi_df = pd.read_csv(kpi_file)
        st.dataframe(kpi_df, use_container_width=True)
    else:
        st.warning("KPI file not found.")

elif menu == "Visualizations":
    st.header("Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Rate by Contract Type")
        img1 = OUTPUT / "churn_by_contract.png"
        if img1.exists():
            st.image(Image.open(img1), use_column_width=True)
            
        st.subheader("Customers by Risk Level")
        img2 = OUTPUT / "customer_risk_distribution.png"
        if img2.exists():
            st.image(Image.open(img2), use_column_width=True)
            
    with col2:
        st.subheader("Top AI Model Drivers of Churn")
        img3 = OUTPUT / "ai_feature_importance.png"
        if img3.exists():
            st.image(Image.open(img3), use_column_width=True)

elif menu == "Data explorer":
    st.header("Data Explorer")
    data_file = OUTPUT / "customer_churn_risk_scores.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Data file not found.")
