import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Models
# ---------------------------

@st.cache_resource
def load_models():
    freight_model = joblib.load("C:/Users/User/Desktop/ML project/model/best_freight_model.pkl")
    flag_model = joblib.load("C:/Users/User/Desktop/ML project/model/predict_flag_invoice.pkl")
    return freight_model, flag_model

freight_model, flag_model = load_models()

# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.title("⚙️ Prediction Settings")

st.sidebar.write("""
Select the prediction model you want to use.

📦 **Freight Prediction** – Estimate transportation or shipping cost of goods.

⚠️ **Invoice Risk Prediction** – Detect invoices that may be flagged due to unusual patterns.
""")

model_option = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Freight Prediction", "Invoice Risk Prediction"]
)

# ---------------------------
# App Title
# ---------------------------

st.title("📦 Invoice Risk & Freight Prediction")

st.info("""
This AI-powered application analyzes invoice data using Machine Learning.

📦 **Freight Cost Prediction**  
Freight cost is the **transportation or shipping charge** required to move goods from a supplier to a warehouse, store, or customer.  
This model predicts the expected freight cost based on the **invoice value**.

⚠️ **Invoice Risk Prediction**  
This model analyzes invoice details and predicts whether an invoice may be **flagged for issues** such as unusual quantities, delays, or mismatched values.

Select a model from the **sidebar**, enter the required information, and click **Predict**.
""")

# ---------------------------
# Freight Prediction
# ---------------------------

if model_option == "Freight Prediction":

    st.header("📦 Freight Cost Prediction")

    dollars = st.number_input("Invoice Dollars ($)", min_value=0.0)
    quantity = st.number_input("Invoice Quantity", min_value=0)

    if st.button("Predict Freight Cost"):

        freight_input = pd.DataFrame({
            "Dollars": [dollars]
        })

        freight_pred = freight_model.predict(freight_input)[0]

        st.subheader("Prediction Result")

        st.success(f"📦 Predicted Freight Cost: ${round(freight_pred,2)}")

# ---------------------------
# Invoice Risk Prediction
# ---------------------------

if model_option == "Invoice Risk Prediction":

    st.header("⚠️ Invoice Risk Prediction")

    quantity = st.number_input("Invoice Quantity", min_value=0)
    dollars = st.number_input("Invoice Dollars ($)", min_value=0.0)
    freight = st.number_input("Freight Cost ($)", min_value=0.0)

    total_item_quantity = st.number_input("Total Item Quantity", min_value=0)
    total_item_dollars = st.number_input("Total Item Dollars ($)", min_value=0.0)
    avg_receiving_delay = st.number_input("Average Receiving Delay (Days)", min_value=0)

    if st.button("Predict Invoice Risk"):

        flag_input = pd.DataFrame({
            "invioce_quantity": [quantity],   # keep same name used during training
            "invoice_dollar": [dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars],
            "avg_receiving_delay": [avg_receiving_delay]
        })

        flag_pred = flag_model.predict(flag_input)[0]

        st.subheader("Prediction Result")

        if flag_pred == 1:
            st.error("⚠️ Invoice Likely to be Flagged")
        else:
            st.success("✅ Invoice Looks Safe")