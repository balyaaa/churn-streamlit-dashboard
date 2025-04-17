
import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load("churn_model_pipeline.joblib")

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.title("🔍 Customer Churn Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload customer data (.csv)", type=["csv"])

if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    st.write("### Preview Data")
    st.dataframe(df_new.head())

    # Predict
    predictions = model.predict(df_new.drop(columns=["CustomerId", "Surname"], errors='ignore'))
    df_new["Churn Prediction"] = predictions

    churn_count = df_new["Churn Prediction"].value_counts()
    churn_percent = churn_count / len(df_new) * 100

    st.write("### Prediction Results")
    st.dataframe(df_new[["CustomerId"] + (["Surname"] if "Surname" in df_new.columns else []) + ["Churn Prediction"]])

    st.write("### Churn Summary")
    st.metric("Churned Customers", churn_count.get(1, 0))
    st.metric("Churn Rate", f"{churn_percent.get(1, 0):.2f}%")

    st.bar_chart(df_new["Churn Prediction"].value_counts())

else:
    st.info("Silakan upload file CSV yang berisi data pelanggan untuk diprediksi.")
