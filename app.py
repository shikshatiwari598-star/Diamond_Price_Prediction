
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("diamond_price_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# App title
st.title("💎 Diamond Price Prediction")
st.write("Predict the estimated price of a diamond using Machine Learning.")

st.divider()

# User input
carat = st.number_input(
    "Enter Diamond Carat",
    min_value=0.2,
    max_value=5.0,
    value=1.0,
    step=0.1
)

# Prediction button
if st.button("Predict Diamond Price", use_container_width=True):

    new_diamond = pd.DataFrame({
        "carat": [carat]
    })

    predicted_price = model.predict(new_diamond)[0]

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Diamond Price",
        value=f"${predicted_price:,.2f}"
    )

    st.info(
        "This is a model-based estimate, not a guaranteed market price."
    )

st.divider()
st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
