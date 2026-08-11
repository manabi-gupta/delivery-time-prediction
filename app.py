
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load trained model and preprocessing pipeline
model = tf.keras.models.load_model("delivery_time_model.keras")
preprocessor = joblib.load("delivery_preprocessor.pkl")

st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Delivery Time Predictor")
st.write(
    "AI-powered delivery time prediction using a Neural Network."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------

market_id = st.number_input(
    "Market ID",
    min_value=0,
    value=1
)

category = st.selectbox(
    "Restaurant Category",
    [
        "american", "pizza", "mexican", "burger",
        "sandwich", "chinese", "japanese", "dessert",
        "fast", "indian", "thai", "italian",
        "vietnamese", "mediterranean", "breakfast"
    ]
)

order_protocol = st.number_input(
    "Order Protocol",
    min_value=0,
    max_value=10,
    value=3
)

total_items = st.number_input(
    "Total Items",
    min_value=1,
    value=2
)

subtotal = st.number_input(
    "Order Subtotal",
    min_value=0.0,
    value=500.0
)

num_distinct_items = st.number_input(
    "Number of Distinct Items",
    min_value=1,
    value=2
)

min_item_price = st.number_input(
    "Minimum Item Price",
    min_value=0.0,
    value=100.0
)

max_item_price = st.number_input(
    "Maximum Item Price",
    min_value=0.0,
    value=300.0
)

onshift = st.number_input(
    "Partners On Shift",
    min_value=0,
    value=20
)

busy = st.number_input(
    "Busy Partners",
    min_value=0,
    value=10
)

outstanding = st.number_input(
    "Outstanding Orders",
    min_value=0,
    value=20
)

hour = st.slider(
    "Order Hour",
    0,
    23,
    19
)

day_of_week = st.slider(
    "Day of Week (Monday = 0)",
    0,
    6,
    2
)

month = st.slider(
    "Month",
    1,
    12,
    2
)

# -----------------------------
# Feature Engineering
# -----------------------------

is_weekend = int(day_of_week >= 5)

available_partners = max(
    onshift - busy,
    0
)

partner_utilization = (
    busy / onshift
    if onshift > 0
    else 0
)

# Default for a store not previously seen
store_frequency = 0

# EXACT 18 FEATURES EXPECTED BY PREPROCESSOR
input_data = pd.DataFrame([{
    "market_id": market_id,
    "store_primary_category": category,
    "order_protocol": order_protocol,
    "total_items": total_items,
    "subtotal": subtotal,
    "num_distinct_items": num_distinct_items,
    "min_item_price": min_item_price,
    "max_item_price": max_item_price,
    "total_onshift_partners": onshift,
    "total_busy_partners": busy,
    "total_outstanding_orders": outstanding,
    "hour": hour,
    "day_of_week": day_of_week,
    "month": month,
    "is_weekend": is_weekend,
    "available_partners": available_partners,
    "partner_utilization": partner_utilization,
    "store_frequency": store_frequency
}])

# -----------------------------
# Prediction
# -----------------------------

if st.button("🚀 Predict Delivery Time"):

    try:
        processed_input = preprocessor.transform(input_data)

        prediction = model.predict(
            processed_input,
            verbose=0
        )[0][0]

        prediction = max(float(prediction), 0)

        st.success(
            f"Estimated Delivery Time: {prediction:.1f} minutes"
        )

    except Exception as e:

        st.error(
            f"Prediction error: {str(e)}"
        )
